#!/usr/bin/env python3
"""Run a declared CPU benchmark matrix as independent processes."""

import argparse
import json
import random
import re
import subprocess
import sys
from pathlib import Path

from run_benchmark import sha256, validation_reports


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tools/run_benchmark.py"


def positive_integer(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def validate(config):
    if config.get("schema_version") != 1:
        raise ValueError("Unsupported benchmark matrix schema")
    positive_integer(config.get("independent_runs"), "independent_runs")
    positive_integer(config.get("repetitions"), "repetitions")
    if type(config.get("schedule_seed")) is not int:
        raise ValueError("schedule_seed must be an integer")
    if not re.fullmatch(r"[0-9]+(?:\.[0-9]+)?s", str(config.get("min_time"))):
        raise ValueError("min_time must be seconds, for example 2s")
    if not re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", str(config.get("warmup"))):
        raise ValueError("warmup must be seconds, for example 1")
    if type(config.get("require_validation")) is not bool:
        raise ValueError("require_validation must be true or false")
    if not isinstance(config.get("filter"), str) or not config["filter"]:
        raise ValueError("filter must be a nonempty benchmark expression")
    cases = config.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("cases must be a nonempty list")
    seen = set()
    for case in cases:
        dataset, window = case.get("dataset"), case.get("window_id")
        if not all(isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_-]+", value)
                   for value in (dataset, window)):
            raise ValueError("dataset and window_id must be simple names")
        if (dataset, window) in seen:
            raise ValueError(f"Duplicate case: {dataset}/{window}")
        seen.add((dataset, window))
        bins = case.get("bins")
        if (not isinstance(bins, list) or not bins or
            any(type(value) is not int or value < 2 for value in bins) or
            len(bins) != len(set(bins))):
            raise ValueError(f"Invalid bins for {dataset}/{window}")
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--executable", type=Path,
                        default=ROOT / "build/cpu-benchmark-release/event_voxel_bench_cpu")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    args.config = args.config.resolve(strict=True)
    config = json.loads(args.config.read_text())
    cases = validate(config)
    executable = args.executable.resolve(strict=True)
    cpu_sha = sha256(executable.parent / "event_voxel_run_cpu")
    for case in cases:
        dataset, window = case["dataset"], case["window_id"]
        prepared = ROOT / "data/processed" / dataset
        if window != "full":
            prepared /= window
        input_path = prepared / "events.bin"
        if not input_path.is_file():
            raise FileNotFoundError(f"Prepare {dataset}/{window} first: {input_path}")
        input_sha = sha256(input_path)
        metadata = json.loads((prepared / "metadata.json").read_text())
        if metadata.get("binary_sha256") != input_sha:
            raise ValueError(f"Prepared metadata differs from input: {input_path}")
        if config["require_validation"]:
            validation_reports(ROOT / "results/validation" / dataset / window,
                               case["bins"], input_sha, cpu_sha)
    if args.validate_only:
        print(f"Matrix ready: {len(cases)} cases, "
              f"{config['independent_runs']} processes per case")
        return

    generator = random.Random(config["schedule_seed"])
    order_index = 0
    for index in range(1, config["independent_runs"] + 1):
        round_cases = cases.copy()
        generator.shuffle(round_cases)
        for case in round_cases:
            order_index += 1
            dataset, window = case["dataset"], case["window_id"]
            prepared = ROOT / "data/processed" / dataset
            if window != "full":
                prepared /= window
            command = [
                sys.executable, str(RUNNER),
                "--input", str(prepared / "events.bin"),
                "--executable", str(executable),
                "--output-dir", str(ROOT / "results/benchmarks" / dataset / window),
                "--bins", ",".join(map(str, case["bins"])),
                "--filter", config["filter"],
                "--min-time", config["min_time"],
                "--warmup", str(config["warmup"]),
                "--repetitions", str(config["repetitions"]),
                "--matrix-config", str(args.config),
                "--matrix-case", f"{dataset}/{window}",
                "--matrix-run-index", str(index),
                "--matrix-order-index", str(order_index),
            ]
            if config["require_validation"]:
                command += ["--validation-dir", str(ROOT / "results/validation" /
                                                      dataset / window)]
            if args.allow_dirty:
                command.append("--allow-dirty")
            print(f"Benchmark case {dataset}/{window}, process {index}/"
                  f"{config['independent_runs']}", flush=True)
            subprocess.run(command, check=True, cwd=ROOT)


if __name__ == "__main__":
    main()
