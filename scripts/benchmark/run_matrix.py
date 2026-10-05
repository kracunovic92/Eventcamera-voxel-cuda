#!/usr/bin/env python3
"""Run a committed CUDA benchmark matrix and retain every measurement."""

import argparse
import csv
import itertools
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path


FIELDS = ["strategy", "workload", "event_count", "bins", "iteration", "end_to_end_ms"]
STRATEGIES = {"event-scatter", "pixel-gather", "sort-reduce"}
WORKLOADS = {"uniform", "collision-heavy"}


def command(*args):
    result = subprocess.run(args, text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def proc_value(path, prefix):
    try:
        for line in Path(path).read_text().splitlines():
            if line.startswith(prefix):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return None


def require_config(path):
    config = json.loads(path.read_text())
    if set(config) != {
        "warmup_iterations", "measurement_iterations", "strategies",
        "event_counts", "bins", "workloads",
    }:
        raise ValueError("config has missing or unknown fields")
    for name in ("warmup_iterations", "measurement_iterations"):
        value = config[name]
        if type(value) is not int or value < (0 if name == "warmup_iterations" else 1):
            raise ValueError(f"{name} must be a valid integer")
    for name, allowed, minimum in (
        ("strategies", STRATEGIES, None),
        ("workloads", WORKLOADS, None),
        ("event_counts", None, 1),
        ("bins", None, 2),
    ):
        values = config[name]
        if not isinstance(values, list) or not values or len(values) != len(set(values)):
            raise ValueError(f"{name} must be a nonempty list without duplicates")
        if allowed is not None and any(value not in allowed for value in values):
            raise ValueError(f"{name} contains an unsupported value")
        if minimum is not None and any(type(value) is not int or value < minimum for value in values):
            raise ValueError(f"{name} contains an invalid integer")
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--correctness-file", type=Path)
    parser.add_argument("--environment-file", type=Path)
    parser.add_argument("--run-path-file", type=Path)
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()

    config = require_config(args.config)
    executable = args.executable.resolve(strict=True)
    if not os.access(executable, os.X_OK):
        raise ValueError(f"benchmark is not executable: {executable}")
    commit = command("git", "rev-parse", "HEAD")
    if not commit:
        raise RuntimeError("run from a Git checkout")
    dirty = bool(command("git", "status", "--porcelain", "--untracked-files=normal"))
    if dirty and not args.allow_dirty:
        raise RuntimeError("working tree is dirty; commit changes or use --allow-dirty")

    gpu_name = command("nvidia-smi", "--query-gpu=name", "--format=csv,noheader")
    if not gpu_name:
        raise RuntimeError("nvidia-smi found no GPU")
    gpu_slug = "".join(char if char.isalnum() else "_" for char in gpu_name.splitlines()[0])[:40]
    run_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{gpu_slug}_{commit[:8]}_{uuid.uuid4().hex[:6]}"
    run_dir = args.output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    if args.run_path_file is not None:
        args.run_path_file.write_text(str(run_dir.resolve()) + "\n")
    shutil.copyfile(args.config, run_dir / "config.json")
    for source, target in ((args.correctness_file, "correctness.txt"),
                           (args.environment_file, "environment.txt")):
        if source is not None:
            shutil.copyfile(source, run_dir / target)

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "commit": commit,
        "branch": command("git", "branch", "--show-current"),
        "dirty": dirty,
        "gpu": command("nvidia-smi", "--query-gpu=name,uuid,memory.total,driver_version,compute_cap", "--format=csv,noheader"),
        "cpu": proc_value("/proc/cpuinfo", "model name") or platform.processor(),
        "cpu_count": os.cpu_count(),
        "memory_total": proc_value("/proc/meminfo", "MemTotal"),
        "platform": platform.platform(),
        "kernel": platform.release(),
        "nvcc": command("nvcc", "--version"),
        "cmake": command("cmake", "--version"),
        "compiler": command("c++", "--version"),
        "python": platform.python_version(),
        "build_type": "Release",
        "executable": str(executable),
        "status": "running",
    }
    metadata_path = run_dir / "metadata.json"

    def save_metadata():
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

    save_metadata()
    (run_dir / "nvidia-smi.txt").write_text(command("nvidia-smi", "-q") or "unavailable\n")
    case_count = 0
    try:
        with (run_dir / "measurements.csv").open("w", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=FIELDS)
            writer.writeheader()
            for strategy, workload, count, bins in itertools.product(
                config["strategies"], config["workloads"], config["event_counts"], config["bins"]
            ):
                args_list = [str(executable), strategy, str(count), str(bins),
                             str(config["warmup_iterations"]),
                             str(config["measurement_iterations"]), workload]
                result = subprocess.run(args_list, text=True, capture_output=True, check=False)
                if result.returncode != 0:
                    raise RuntimeError(f"benchmark failed for {args_list[1:]}: {result.stderr.strip()}")
                rows = list(csv.DictReader(result.stdout.splitlines()))
                if not rows or list(rows[0]) != FIELDS or len(rows) != config["measurement_iterations"]:
                    raise ValueError(f"invalid benchmark CSV for {args_list[1:]}")
                for iteration, row in enumerate(rows):
                    if (row["strategy"] != strategy or row["workload"] != workload or
                        int(row["event_count"]) != count or int(row["bins"]) != bins or
                        int(row["iteration"]) != iteration or
                        not math.isfinite(float(row["end_to_end_ms"])) or
                        float(row["end_to_end_ms"]) <= 0):
                        raise ValueError(f"invalid benchmark row for {args_list[1:]}")
                    writer.writerow(row)
                output.flush()
                case_count += 1
                print(f"{case_count}: {strategy} {workload} {count} events {bins} bins", flush=True)
        metadata["status"] = "complete"
        metadata["cases"] = case_count
        save_metadata()
        print(run_dir)
    except Exception as error:
        metadata["status"] = "failed"
        metadata["error"] = str(error)
        save_metadata()
        print(f"partial results retained in {run_dir}", file=sys.stderr)
        raise


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        sys.exit(f"matrix error: {error}")
