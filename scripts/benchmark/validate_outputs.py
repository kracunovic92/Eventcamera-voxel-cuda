#!/usr/bin/env python3
"""Give a pass/fail check of saved CPU and CUDA arrays."""

import argparse
from pathlib import Path

from output_comparison import analyze_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-5)
    args = parser.parse_args()
    rows = analyze_run(args.run_dir, args.atol, args.rtol)
    failures = 0
    for row in rows:
        passed = row["passed"]
        failures += not passed
        print(f"{'PASS' if passed else 'FAIL'} {row['dataset']} "
              f"{row['event_count']} events {row['bins']} bins {row['strategy']}: "
              f"{row['failed_voxels']} / {row['voxel_count']} voxels")
    print(f"Validated {len(rows)} CPU/CUDA pairs; {failures} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
