#!/usr/bin/env python3
"""Compare a voxel grid with the triangular definition in float64."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from verify_external import load_events


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reference_grid(events, metadata, bins):
    duration = metadata["end_us"] - metadata["start_us"]
    if duration > 2**53:
        raise ValueError("Duration exceeds exact float64 integer precision")
    shape = (bins, metadata["height"], metadata["width"])
    reference = np.zeros(shape, dtype=np.float64)
    for offset in range(0, len(events), 1_000_000):
        chunk = events[offset:offset + 1_000_000]
        relative = (chunk["t"] - metadata["start_us"]).astype(np.float64)
        position = (bins - 1) * relative / duration
        sign = np.where(chunk["p"] == 1, 1.0, -1.0)
        for bin_index in range(bins):
            weight = np.maximum(0.0, 1.0 - np.abs(bin_index - position))
            active = weight > 0
            np.add.at(reference[bin_index],
                      (chunk["y"][active], chunk["x"][active]),
                      sign[active] * weight[active])
    return reference


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--voxels", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--cpu-executable", type=Path)
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--rtol", type=float, default=1e-4)
    args = parser.parse_args()
    if (not np.isfinite(args.atol) or not np.isfinite(args.rtol) or
        args.atol < 0 or args.rtol < 0):
        parser.error("Tolerances must be finite and nonnegative")

    events, metadata = load_events(args.events)
    actual = np.load(args.voxels, allow_pickle=False)
    if (actual.dtype != np.float32 or actual.ndim != 3 or
        actual.shape[1:] != (metadata["height"], metadata["width"])):
        raise ValueError("Voxel shape or dtype differs from the binary recording")
    if actual.shape[0] < 2 or not np.isfinite(actual).all():
        raise ValueError("Invalid voxel output")
    reference = reference_grid(events, metadata, actual.shape[0])
    error = np.abs(actual.astype(np.float64) - reference)
    allowed = args.atol + args.rtol * np.abs(reference)
    failed = error > allowed
    worst = np.unravel_index(int(np.argmax(error)), error.shape)
    metrics = {
        "passed": not bool(np.any(failed)),
        "voxel_count": int(error.size),
        "failed_voxels": int(np.count_nonzero(failed)),
        "max_absolute_error": float(error.max()),
        "mean_absolute_error": float(error.mean()),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "worst_voxel": {
            "bin": int(worst[0]), "y": int(worst[1]), "x": int(worst[2]),
            "actual": float(actual[worst]), "reference": float(reference[worst]),
        },
        "tolerances": {"absolute": args.atol, "relative": args.rtol},
    }
    report = {
        "reference": "float64 triangular-weight accumulation",
        "result": metrics,
        "recording": {**metadata, "sha256": sha256(args.events)},
        "cpu_output": {"path": str(args.voxels.resolve()),
                       "sha256": sha256(args.voxels), "shape": list(actual.shape)},
        "numpy_version": np.__version__,
    }
    if args.cpu_executable:
        report["cpu_executable"] = {
            "path": str(args.cpu_executable.resolve(strict=True)),
            "sha256": sha256(args.cpu_executable),
        }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Precision reference: {'PASS' if metrics['passed'] else 'FAIL'}")
    print(f"Maximum error: {metrics['max_absolute_error']:.9g}")
    print(f"Failed voxels: {metrics['failed_voxels']} / {metrics['voxel_count']}")
    print(f"Report: {args.report}")
    return 0 if metrics["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
