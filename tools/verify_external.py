
#!/usr/bin/env python3
"""Compare our C++ voxelizer with RPG E2Depth's NumPy implementation.

The published RPG function is loaded from a separate Git checkout.
Only two compatibility adaptations are applied:

1. Use our recording's explicit time interval.
2. Replace deprecated np.int with np.int64.

RPG's interpolation and accumulation operations remain unchanged.
"""

import argparse
import ast
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np


MAGIC = b"EVXEVT01"
EXPECTED_RPG_REVISION = "17d453c175d94a6e4e666b4dafb64b072eb2df5d"
EXPECTED_RPG_SHA256 = "dfb1c40092ea02cdd861e212d22bef8e256994234dab08e7ab35f8606ec8bdd8"

HEADER_DTYPE = np.dtype([
    ("magic", "S8"),
    ("width", "<u4"),
    ("height", "<u4"),
    ("count", "<u8"),
    ("start", "<i8"),
    ("end", "<i8"),
])

EVENT_DTYPE = np.dtype({
    "names": ["x", "y", "p", "reserved", "t"],
    "formats": ["<u2", "<u2", "<i2", "<u2", "<i8"],
    "offsets": [0, 2, 4, 6, 8],
    "itemsize": 16,
})


def sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def git_revision(repository: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        capture_output=True,
        text=True,
        check=False,
    )

    return result.stdout.strip() if result.returncode == 0 else None


def source_is_modified(repository: Path, source: Path) -> bool:
    result = subprocess.run(
        ["git", "status", "--porcelain", "--", str(source)],
        cwd=repository,
        capture_output=True,
        text=True,
        check=False,
    )

    return result.returncode != 0 or bool(result.stdout.strip())


def load_events(path: Path):
    """Read our versioned binary format independently of C++."""

    if path.stat().st_size < HEADER_DTYPE.itemsize:
        raise ValueError("Binary file is too small")

    header = np.fromfile(
        path,
        dtype=HEADER_DTYPE,
        count=1,
    )[0]

    if header["magic"] != MAGIC:
        raise ValueError("Incorrect binary format identifier")

    width = int(header["width"])
    height = int(header["height"])
    count = int(header["count"])
    start = int(header["start"])
    end = int(header["end"])

    if width <= 0 or height <= 0 or count <= 0:
        raise ValueError("Invalid recording dimensions or event count")

    if start < 0 or end <= start:
        raise ValueError("Invalid recording time interval")

    expected_size = (
        HEADER_DTYPE.itemsize
        + count * EVENT_DTYPE.itemsize
    )

    if path.stat().st_size != expected_size:
        raise ValueError("Event count and binary file size disagree")

    events = np.memmap(
        path,
        dtype=EVENT_DTYPE,
        mode="r",
        offset=HEADER_DTYPE.itemsize,
        shape=(count,),
    )

    if np.any(events["reserved"] != 0):
        raise ValueError("Nonzero reserved event bytes")

    if np.any(events["x"] >= width) or np.any(events["y"] >= height):
        raise ValueError("Events contain invalid coordinates")

    if np.any((events["p"] != 0) & (events["p"] != 1)):
        raise ValueError("Events contain invalid polarities")

    timestamps = events["t"]

    if np.any(timestamps < start) or np.any(timestamps >= end):
        raise ValueError("Events outside the recording time interval")

    if np.any(timestamps[1:] < timestamps[:-1]):
        raise ValueError("Event timestamps are not ordered")

    return events, {
        "width": width,
        "height": height,
        "event_count": count,
        "start_us": start,
        "end_us": end,
    }


class ReferenceAdapter(ast.NodeTransformer):
    """Apply narrowly scoped changes to RPG's published function."""

    def __init__(self):
        self.boundary_changes = set()
        self.numpy_compatibility_changes = 0

    def visit_Assign(self, node):
        if len(node.targets) == 1:
            target = node.targets[0]

            if isinstance(target, ast.Name):
                replacements = {
                    "first_stamp": "REFERENCE_START",
                    "last_stamp": "REFERENCE_END",
                }

                if target.id in replacements:
                    # Reject unexpected source revisions that
                    # assign these variables more than once.
                    if target.id in self.boundary_changes:
                        raise ValueError(
                            f"Repeated assignment to {target.id}"
                        )

                    self.boundary_changes.add(target.id)

                    node.value = ast.copy_location(
                        ast.Name(
                            id=replacements[target.id],
                            ctx=ast.Load(),
                        ),
                        node.value,
                    )

        return self.generic_visit(node)

    def visit_Attribute(self, node):
        node = self.generic_visit(node)

        if (
            isinstance(node.value, ast.Name)
            and node.value.id == "np"
            and node.attr == "int"
        ):
            node.attr = "int64"
            self.numpy_compatibility_changes += 1

        return node


def load_rpg_function(
    source: Path,
    duration_us: int,
):
    """Compile only RPG's original NumPy voxelization function."""

    source_text = source.read_text(encoding="utf-8")
    tree = ast.parse(source_text, filename=str(source))

    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "events_to_voxel_grid"
    ]

    if len(functions) != 1:
        raise ValueError(
            "Expected exactly one RPG events_to_voxel_grid function"
        )

    adapter = ReferenceAdapter()

    function = adapter.visit(functions[0])

    if adapter.boundary_changes != {
        "first_stamp",
        "last_stamp",
    }:
        raise ValueError(
            "Unexpected RPG source: temporal boundaries changed"
        )

    if adapter.numpy_compatibility_changes == 0:
        raise ValueError(
            "Unexpected RPG source: np.int compatibility "
            "adjustment was not applied"
        )

    module = ast.fix_missing_locations(
        ast.Module(
            body=[function],
            type_ignores=[],
        )
    )

    namespace = {
        "np": np,
        "REFERENCE_START": 0.0,
        "REFERENCE_END": float(duration_us),
    }

    # Execute only the selected function, not RPG's
    # module imports or deep-learning components.
    #
    # As with any executable third-party source,
    # use a trusted, pinned repository revision.
    exec(
        compile(module, str(source), "exec"),
        namespace,
    )

    return namespace["events_to_voxel_grid"], {
        "boundary_adaptations": sorted(
            adapter.boundary_changes
        ),
        "numpy_compatibility_changes": (
            adapter.numpy_compatibility_changes
        ),
    }


def calculate_external_reference(
    events: np.memmap,
    metadata: dict,
    bins: int,
    reference_function,
) -> np.ndarray:
    """Prepare RPG's required [timestamp, x, y, polarity] input."""

    duration = (
        metadata["end_us"]
        - metadata["start_us"]
    )

    # A float64 value represents all integers exactly
    # only up to 2**53.
    if duration > 2**53:
        raise ValueError(
            "Recording duration exceeds exact float64 "
            "integer precision"
        )

    # RPG requires a dense N x 4 NumPy array.
    # Subtract the integer origin before converting to
    # float64 to preserve timestamp precision.
    rpg_events = np.empty(
        (len(events), 4),
        dtype=np.float64,
    )

    rpg_events[:, 0] = (
        events["t"] - metadata["start_us"]
    ).astype(np.float64)

    rpg_events[:, 1] = events["x"]
    rpg_events[:, 2] = events["y"]
    rpg_events[:, 3] = events["p"]

    # The RPG implementation modifies its input array.
    # This array is independent of our memory-mapped file.
    reference = reference_function(
        rpg_events,
        bins,
        metadata["width"],
        metadata["height"],
    )

    expected_shape = (
        bins,
        metadata["height"],
        metadata["width"],
    )

    if reference.shape != expected_shape:
        raise ValueError("Unexpected external reference shape")

    if not np.isfinite(reference).all():
        raise ValueError("External reference contains invalid values")

    return reference


def compare(
    actual: np.ndarray,
    reference: np.ndarray,
    atol: float,
    rtol: float,
) -> dict:
    """Evaluate numerical agreement at every voxel."""

    if actual.shape != reference.shape:
        raise ValueError("CPU and reference tensor shapes disagree")

    if not np.isfinite(actual).all():
        raise ValueError("CPU output contains NaN or infinity")

    # Use float64 for the error calculation itself.
    error = np.abs(
        actual.astype(np.float64)
        - reference.astype(np.float64)
    )

    allowed = (
        atol
        + rtol * np.abs(reference.astype(np.float64))
    )

    failed = error > allowed

    worst = np.unravel_index(
        int(np.argmax(error)),
        error.shape,
    )

    failed_count = int(np.count_nonzero(failed))

    return {
        "passed": failed_count == 0,
        "voxel_count": int(error.size),
        "failed_voxels": failed_count,
        "max_absolute_error": float(error.max()),
        "mean_absolute_error": float(error.mean()),
        "rmse": float(
            np.sqrt(np.mean(np.square(error)))
        ),
        "worst_voxel": {
            "bin": int(worst[0]),
            "y": int(worst[1]),
            "x": int(worst[2]),
            "cpu_value": float(actual[worst]),
            "reference_value": float(reference[worst]),
        },
        "tolerances": {
            "absolute": atol,
            "relative": rtol,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify CPU voxelization using RPG E2Depth"
    )

    parser.add_argument(
        "--events",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--voxels",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--rpg-repo",
        type=Path,
        default=Path("external/rpg_e2depth"),
    )

    parser.add_argument(
        "--report",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--cpu-executable",
        type=Path,
        help="Executable that produced the supplied NumPy voxel file",
    )

    parser.add_argument(
        "--expected-rpg-sha256",
        type=str,
        default=EXPECTED_RPG_SHA256,
        help="Required RPG source SHA-256 (defaults to the pinned revision)",
    )
    parser.add_argument(
        "--expected-rpg-revision",
        default=EXPECTED_RPG_REVISION,
        help="Required RPG Git revision",
    )

    parser.add_argument(
        "--atol",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-4,
    )

    args = parser.parse_args()

    if (
        not np.isfinite(args.atol)
        or not np.isfinite(args.rtol)
        or args.atol < 0
        or args.rtol < 0
    ):
        parser.error("Tolerances must be finite and nonnegative")

    source = (
        args.rpg_repo
        / "utils"
        / "event_tensor_utils.py"
    )

    if not source.is_file():
        parser.error(f"RPG source not found: {source}")

    source_hash = sha256(source)

    if source_hash != args.expected_rpg_sha256.lower():
        parser.error("RPG source SHA-256 does not match")
    revision = git_revision(args.rpg_repo.resolve())
    if revision != args.expected_rpg_revision:
        parser.error("RPG Git revision does not match")
    if source_is_modified(args.rpg_repo.resolve(), source.resolve()):
        parser.error("RPG source has local modifications")

    events, metadata = load_events(args.events)

    actual = np.load(
        args.voxels,
        allow_pickle=False,
    )

    if actual.ndim != 3:
        raise ValueError("CPU output must be a 3D tensor")

    if actual.dtype != np.dtype("float32"):
        raise ValueError("CPU output must contain float32 values")

    bins, height, width = actual.shape

    if bins < 2:
        raise ValueError("At least two temporal bins are required")

    if (
        width != metadata["width"]
        or height != metadata["height"]
    ):
        raise ValueError("CPU output and recording geometry disagree")

    reference_function, adaptations = load_rpg_function(
        source,
        metadata["end_us"] - metadata["start_us"],
    )

    reference = calculate_external_reference(
        events,
        metadata,
        bins,
        reference_function,
    )

    metrics = compare(
        actual,
        reference,
        args.atol,
        args.rtol,
    )

    report = {
        "result": metrics,
        "recording": {
            **metadata,
            "path": str(args.events.resolve()),
            "sha256": sha256(args.events),
        },
        "cpu_output": {
            "path": str(args.voxels.resolve()),
            "sha256": sha256(args.voxels),
            "shape": list(actual.shape),
        },
        "cpu_executable": (
            {
                "path": str(args.cpu_executable.resolve()),
                "sha256": sha256(args.cpu_executable),
            }
            if args.cpu_executable is not None
            else None
        ),
        "external_reference": {
            "repository": (
                "https://github.com/uzh-rpg/rpg_e2depth"
            ),
            "git_revision": revision,
            "source_sha256": source_hash,
            "source_modified": source_is_modified(
                args.rpg_repo.resolve(),
                source.resolve(),
            ),
            "adaptations": adaptations,
            "numpy_version": np.__version__,
        },
    }

    args.report.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.report.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Events: {metadata['event_count']:,}")
    print(f"Voxel shape: {actual.shape}")
    print(f"RPG revision: {report['external_reference']['git_revision']}")
    print(f"Maximum error: {metrics['max_absolute_error']:.9g}")
    print(f"Mean error: {metrics['mean_absolute_error']:.9g}")
    print(f"RMSE: {metrics['rmse']:.9g}")
    print(
        f"Failed voxels: "
        f"{metrics['failed_voxels']:,} / "
        f"{metrics['voxel_count']:,}"
    )
    print(f"Result: {'PASS' if metrics['passed'] else 'FAIL'}")
    print(f"Report: {args.report}")

    if not metrics["passed"]:
        print(f"Worst voxel: {metrics['worst_voxel']}")

    return 0 if metrics["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
