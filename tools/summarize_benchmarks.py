#!/usr/bin/env python3
"""Summarize process-level benchmark medians without pooling iterations."""

import argparse
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-dirty", action="store_true",
                        help="Include exploratory runs from uncommitted source")
    args = parser.parse_args()

    groups = defaultdict(list)
    for manifest_path in sorted(args.runs.rglob("manifest.json")):
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("schema_version") != 2:
            continue
        if manifest.get("status") != "completed":
            continue
        if manifest["source"]["dirty"] and not args.include_dirty:
            continue
        if not manifest.get("validations"):
            continue
        benchmark_path = manifest_path.parent / "benchmark.json"
        if not benchmark_path.is_file():
            continue
        if hashlib.sha256(benchmark_path.read_bytes()).hexdigest() != manifest.get(
            "benchmark_sha256"):
            raise ValueError(f"Benchmark JSON differs from its manifest: {benchmark_path}")
        for case in manifest["validations"].values():
            for record in case.values():
                path = Path(record["path"])
                if not path.is_absolute():
                    path = manifest_path.parent / path
                if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
                    raise ValueError(f"Validation report differs from manifest: {path}")
        output = json.loads(benchmark_path.read_text())
        by_case = defaultdict(list)
        for entry in output["benchmarks"]:
            if entry.get("run_type") == "iteration":
                by_case[(entry["name"], entry["time_unit"])].append(
                    entry["real_time"])
        source_digest = hashlib.sha256(json.dumps(
            manifest["source"]["file_sha256"], sort_keys=True
        ).encode()).hexdigest()
        settings = manifest["parameters"]
        matrix_sha = manifest.get("matrix", {}).get("config_sha256")
        for (name, unit), times in by_case.items():
            if len(times) != settings["repetitions"]:
                raise ValueError(f"Incomplete repetitions in {benchmark_path}: {name}")
            key = (
                manifest["dataset"]["sha256"],
                manifest["executable"]["sha256"], source_digest,
                matrix_sha or "",
                settings["min_time"], settings["warmup_seconds"],
                settings["repetitions"], name, unit,
            )
            groups[key].append({
                "run_id": manifest["run_id"],
                "repetitions": len(times),
                "median": statistics.median(times),
                "minimum": min(times),
                "maximum": max(times),
            })

    if not groups:
        parser.error("No completed benchmark runs found")
    report = {"schema_version": 1, "groups": []}
    for (dataset_sha, executable_sha, source_sha, matrix_sha, min_time,
         warmup, repetitions, name, unit), runs in sorted(groups.items()):
        process_medians = [run["median"] for run in runs]
        report["groups"].append({
            "dataset_sha256": dataset_sha,
            "executable_sha256": executable_sha,
            "source_tree_sha256": source_sha,
            "matrix_config_sha256": matrix_sha or None,
            "minimum_time": min_time,
            "warmup_seconds": warmup,
            "repetitions": repetitions,
            "benchmark": name,
            "time_unit": unit,
            "independent_runs": len(runs),
            "median_of_run_medians": statistics.median(process_medians),
            "min_run_median": min(process_medians),
            "max_run_median": max(process_medians),
            "runs": runs,
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Summary: {args.output}")
    for group in report["groups"]:
        print(group["benchmark"], group["independent_runs"],
              group["median_of_run_medians"], group["time_unit"])


if __name__ == "__main__":
    main()
