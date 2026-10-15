#!/usr/bin/env python3
"""Summarize CUDA measurements without changing the raw CSV."""

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("measurements", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.measurements.with_name("summary.csv")
    groups = defaultdict(list)
    with args.measurements.open(newline="") as source:
        for row in csv.DictReader(source):
            key = (row["strategy"], row["workload"], row["dataset"],
                   row["dataset_sha256"], row["event_count"], row["bins"])
            groups[key].append(float(row["end_to_end_ms"]))
    if not groups:
        raise ValueError("no measurements found")
    with output.open("w", newline="") as target:
        writer = csv.writer(target)
        writer.writerow(["strategy", "workload", "dataset", "dataset_sha256", "event_count", "bins", "samples",
                         "mean_ms", "median_ms", "stddev_ms", "min_ms", "max_ms", "p50_ms", "p95_ms"])
        for key, values in sorted(groups.items()):
            values.sort()
            p95_position = 0.95 * (len(values) - 1)
            lower = int(p95_position)
            p95 = values[lower] + (values[min(lower + 1, len(values) - 1)] - values[lower]) * (p95_position - lower)
            writer.writerow([*key, len(values), statistics.mean(values), statistics.median(values),
                             statistics.stdev(values) if len(values) > 1 else 0,
                             values[0], values[-1], statistics.median(values), p95])


if __name__ == "__main__":
    main()
