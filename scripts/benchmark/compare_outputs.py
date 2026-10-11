#!/usr/bin/env python3
"""Write detailed CPU versus CUDA accuracy and run-metric CSV from saved arrays."""

import argparse
import csv
from pathlib import Path

from output_comparison import analyze_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-5)
    args = parser.parse_args()
    rows = analyze_run(args.run_dir, args.atol, args.rtol)
    destination = args.output or args.run_dir / "comparisons.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    failures = sum(not row["passed"] for row in rows)
    print(f"Compared {len(rows)} CPU/CUDA pairs: {failures} failed. CSV: {destination}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
