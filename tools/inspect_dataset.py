
#!/usr/bin/env python3
"""Inspect original Prophesee HDF5 event recordings.

Examples:
    python3 tools/inspect_dataset.py --input data/raw
    python3 tools/inspect_dataset.py --input data/raw/sparklers.hdf5
    python3 tools/inspect_dataset.py --input data/raw --force

Run after configuring the Prophesee ECF HDF5 plugin.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path

import h5py
import numpy as np


SCHEMA_VERSION = 1
CHUNK_SIZE = 100_000
SUPPORTED_EXTENSIONS = {".h5", ".hdf5"}

REQUIRED_EVENT_FIELDS = {"x", "y", "p", "t"}


# ---------------------------------------------------------
# 1. File discovery and identification
# ---------------------------------------------------------

def discover_recordings(source: Path) -> list[Path]:
    """Discover recordings in a file or directory."""

    if source.is_file():
        if source.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file: {source}")

        return [source.resolve()]

    if not source.is_dir():
        raise FileNotFoundError(source)

    return sorted(
        path.resolve()
        for path in source.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def sha256_file(path: Path) -> str:
    """Calculate a file fingerprint without loading it all."""

    digest = hashlib.sha256()

    with path.open("rb") as file:
        while block := file.read(1024 * 1024):
            digest.update(block)

    return digest.hexdigest()


def output_path(
    recording: Path,
    source: Path,
    output_dir: Path,
) -> Path:
    """Create an output path without losing subdirectories."""

    root = source.resolve()

    if root.is_file():
        root = root.parent

    relative = recording.relative_to(root)

    return (
        output_dir
        / relative.parent
        / f"{relative.stem}.metadata.json"
    )


# ---------------------------------------------------------
# 2. Recording metadata
# ---------------------------------------------------------

def decode_attribute(value):
    """Convert HDF5 metadata values into ordinary Python values."""

    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")

    if isinstance(value, np.ndarray):
        return [decode_attribute(item) for item in value.tolist()]

    if isinstance(value, np.generic):
        return decode_attribute(value.item())

    if isinstance(value, (str, int, float, bool)):
        return value

    return str(value)


def read_recording_metadata(file: h5py.File) -> dict:
    """Extract metadata relevant to our experiments."""

    attributes = {
        name: decode_attribute(value)
        for name, value in file.attrs.items()
        if name != "serial_number"
    }

    raw_geometry = attributes.get("geometry", "")

    geometry_match = re.fullmatch(
        r"\s*(\d+)\s*[xX]\s*(\d+)\s*",
        str(raw_geometry),
    )

    if geometry_match is None:
        raise ValueError(
            f"Unsupported or missing geometry: {raw_geometry!r}"
        )

    width = int(geometry_match.group(1))
    height = int(geometry_match.group(2))

    if width == 0 or height == 0:
        raise ValueError("Sensor dimensions must be positive")

    return {
        "width": width,
        "height": height,
        "timestamp_unit": "us",
        "polarity_encoding": {
            "0": "negative",
            "1": "positive",
        },
        "attributes": attributes,
    }


# ---------------------------------------------------------
# 3. Event inspection
# ---------------------------------------------------------

def update_range(
    statistics: dict,
    name: str,
    values: np.ndarray,
) -> None:
    """Update minimum and maximum observations."""

    if values.size == 0:
        return

    minimum = int(values.min())
    maximum = int(values.max())

    min_key = f"min_{name}"
    max_key = f"max_{name}"

    if statistics[min_key] is None:
        statistics[min_key] = minimum
        statistics[max_key] = maximum
    else:
        statistics[min_key] = min(
            statistics[min_key],
            minimum,
        )
        statistics[max_key] = max(
            statistics[max_key],
            maximum,
        )


def inspect_events(
    events: h5py.Dataset,
    width: int,
    height: int,
    chunk_size: int,
) -> dict:
    """Inspect all CD events incrementally."""

    if events.ndim != 1:
        raise ValueError("Expected a one-dimensional event dataset")

    fields = events.dtype.fields or {}

    if not REQUIRED_EVENT_FIELDS.issubset(fields):
        raise ValueError(
            "Missing event fields: "
            f"{sorted(REQUIRED_EVENT_FIELDS - fields.keys())}"
        )

    # Reject unexpected field types rather than silently
    # converting potentially incompatible recordings.
    expected_types = {
        "x": np.dtype("<u2"),
        "y": np.dtype("<u2"),
        "p": np.dtype("<i2"),
        "t": np.dtype("<i8"),
    }

    for name, expected in expected_types.items():
        actual = fields[name][0]

        if actual.newbyteorder("=") != expected.newbyteorder("="):
            raise ValueError(
                f"Unexpected dtype for {name}: {actual}"
            )

    statistics = {
        "event_count": 0,
        "min_x": None,
        "max_x": None,
        "min_y": None,
        "max_y": None,
        "min_timestamp_us": None,
        "max_timestamp_us": None,
        "first_timestamp_us": None,
        "last_timestamp_us": None,
        "positive_events": 0,
        "negative_events": 0,
        "invalid_polarities": 0,
        "invalid_coordinates": 0,
        "negative_timestamps": 0,
        "timestamp_order_violations": 0,
    }

    previous_timestamp = None

    for start in range(0, len(events), chunk_size):
        chunk = events[start : start + chunk_size]

        if len(chunk) == 0:
            continue

        x = chunk["x"]
        y = chunk["y"]
        polarity = chunk["p"]
        timestamps = chunk["t"]

        statistics["event_count"] += len(chunk)

        statistics["positive_events"] += int(
            np.count_nonzero(polarity == 1)
        )

        statistics["negative_events"] += int(
            np.count_nonzero(polarity == 0)
        )

        statistics["invalid_polarities"] += int(
            np.count_nonzero(
                (polarity != 0) & (polarity != 1)
            )
        )

        statistics["invalid_coordinates"] += int(
            np.count_nonzero(
                (x >= width) | (y >= height)
            )
        )

        statistics["negative_timestamps"] += int(
            np.count_nonzero(timestamps < 0)
        )

        update_range(statistics, "x", x)
        update_range(statistics, "y", y)
        update_range(
            statistics,
            "timestamp_us",
            timestamps,
        )

        if statistics["first_timestamp_us"] is None:
            statistics["first_timestamp_us"] = int(
                timestamps[0]
            )

        statistics["last_timestamp_us"] = int(
            timestamps[-1]
        )

        # Check ordering within the current chunk.
        statistics["timestamp_order_violations"] += int(
            np.count_nonzero(
                timestamps[1:] < timestamps[:-1]
            )
        )

        # Check ordering between consecutive chunks.
        if previous_timestamp is not None:
            if timestamps[0] < previous_timestamp:
                statistics["timestamp_order_violations"] += 1

        previous_timestamp = int(timestamps[-1])

    statistics["timestamps_nondecreasing"] = (
        statistics["timestamp_order_violations"] == 0
    )

    statistics["validation_passed"] = all((
        statistics["invalid_polarities"] == 0,
        statistics["invalid_coordinates"] == 0,
        statistics["negative_timestamps"] == 0,
        statistics["timestamps_nondecreasing"],
        statistics["event_count"] > 0,
    ))

    return statistics


# ---------------------------------------------------------
# 4. Complete recording inspection
# ---------------------------------------------------------

def inspect_recording(
    path: Path,
    chunk_size: int,
) -> dict:
    """Produce a validated report for one HDF5 recording."""

    before = path.stat()

    with h5py.File(path, "r") as file:
        metadata = read_recording_metadata(file)

        if "CD/events" not in file:
            raise ValueError("Missing CD/events dataset")

        events = file["CD/events"]

        statistics = inspect_events(
            events,
            metadata["width"],
            metadata["height"],
            chunk_size,
        )

        event_format = {
            "path": "/CD/events",
            "count": len(events),
            "record_bytes": events.dtype.itemsize,
            "fields": {
                name: str(events.dtype.fields[name][0])
                for name in REQUIRED_EVENT_FIELDS
            },
        }

    fingerprint = sha256_file(path)

    after = path.stat()

    if (
        before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
    ):
        raise RuntimeError(
            "Source recording changed during inspection"
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "status": (
            "valid"
            if statistics["validation_passed"]
            else "invalid"
        ),
        "source": {
            "path": str(path),
            "size_bytes": after.st_size,
            "mtime_ns": after.st_mtime_ns,
            "sha256": fingerprint,
        },
        "recording": metadata,
        "event_format": event_format,
        "statistics": statistics,
    }


# ---------------------------------------------------------
# 5. Reports, caching and command-line interface
# ---------------------------------------------------------

def is_cached(
    recording: Path,
    destination: Path,
) -> bool:
    """Skip previously validated, unchanged recordings."""

    if not destination.is_file():
        return False

    try:
        report = json.loads(
            destination.read_text(encoding="utf-8")
        )

        source = report["source"]
        current = recording.stat()

        return (
            report["schema_version"] == SCHEMA_VERSION
            and report["status"] == "valid"
            and source["path"] == str(recording)
            and source["size_bytes"] == current.st_size
            and source["mtime_ns"] == current.st_mtime_ns
            and bool(source.get("sha256"))
        )

    except (OSError, ValueError, KeyError, TypeError):
        return False


def save_report(
    destination: Path,
    report: dict,
) -> None:
    """Save one report without partially replacing an old one."""

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = destination.with_suffix(
        destination.suffix + ".tmp"
    )

    try:
        temporary.write_text(
            json.dumps(
                report,
                indent=2,
                sort_keys=True,
            ) + "\n",
            encoding="utf-8",
        )

        temporary.replace(destination)

    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect Prophesee HDF5 recordings"
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw"),
        help="One recording or a directory of recordings",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/metadata"),
        help="Directory for generated inspection reports",
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=CHUNK_SIZE,
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Reinspect recordings even when cached",
    )

    args = parser.parse_args()

    if args.chunk_size <= 0:
        parser.error("--chunk-size must be positive")

    recordings = discover_recordings(args.input)

    if not recordings:
        print("No HDF5 recordings found.")
        return 0

    failures = 0

    for path in recordings:
        destination = output_path(
            path,
            args.input,
            args.output,
        )

        if not args.force and is_cached(
            path,
            destination,
        ):
            print(f"SKIP: {path.name}")
            continue

        print(f"INSPECT: {path.name}", flush=True)

        try:
            report = inspect_recording(
                path,
                args.chunk_size,
            )

            save_report(destination, report)

            if report["status"] == "valid":
                print(
                    f"OK: {path.name} "
                    f"({report['statistics']['event_count']} events)"
                )
            else:
                failures += 1
                print(f"INVALID: {path.name}")

        except Exception as error:
            failures += 1
            print(
                f"FAILED: {path.name}: "
                f"{type(error).__name__}: {error}"
            )

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
