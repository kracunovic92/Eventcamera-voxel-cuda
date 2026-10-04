#!/usr/bin/env python3
"""Run one CPU benchmark process and save a reproducible run manifest."""

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import re
import shutil
import struct
import subprocess
import sys
import uuid
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True
    ).strip()


def source_hashes() -> dict[str, str]:
    paths = git("ls-files", "--cached", "--others", "--exclude-standard").splitlines()
    suffixes = {".cpp", ".hpp", ".h", ".cu", ".cuh", ".py", ".sh", ".cmake"}
    named = {"CMakeLists.txt", "CMakePresets.json", "Makefile", "requirements.txt",
             ".pre-commit-config.yaml"}
    selected = [
        path for path in paths
        if (Path(path).suffix in suffixes or Path(path).name in named or
            path.startswith(".github/workflows/")) and (ROOT / path).is_file()
    ]
    return {path: sha256(ROOT / path) for path in sorted(selected)}


def parse_bins(value: str) -> list[int]:
    try:
        bins = [int(part.strip()) for part in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("Bins must be comma-separated integers") from error
    if not bins or any(bin_count < 2 for bin_count in bins) or len(set(bins)) != len(bins):
        raise argparse.ArgumentTypeError("Bin counts must be unique integers >= 2")
    return bins


def prepared_header(path: Path) -> dict:
    with path.open("rb") as stream:
        raw = stream.read(40)
    if len(raw) != 40:
        raise ValueError("Prepared event header is incomplete")
    magic, width, height, count, start, end = struct.unpack("<8sIIQqq", raw)
    if magic != b"EVXEVT01" or count == 0 or end <= start:
        raise ValueError("Invalid prepared event header")
    return {
        "width": width, "height": height, "event_count": count,
        "start_us": start, "end_us": end,
    }


def validation_reports(directory: Path, bins: list[int], input_sha: str,
                       cpu_executable_sha: str) -> dict:
    reports = {}
    for bin_count in bins:
        case = {}
        output_sha = None
        for kind, path in (("external", directory / f"bins_{bin_count}.json"),
                           ("precision", directory / f"bins_{bin_count}.precision.json")):
            report = json.loads(path.read_text())
            if (not report["result"]["passed"] or
                report["recording"]["sha256"] != input_sha or
                report["cpu_output"]["shape"][0] != bin_count or
                not report.get("cpu_executable") or
                report["cpu_executable"]["sha256"] != cpu_executable_sha):
                raise ValueError(f"Validation does not match benchmark case: {path}")
            current_sha = report["cpu_output"]["sha256"]
            if output_sha is not None and output_sha != current_sha:
                raise ValueError(f"Validation outputs differ for bin count {bin_count}")
            output_sha = current_sha
            case[kind] = {"path": str(path.resolve()), "sha256": sha256(path)}
        reports[str(bin_count)] = case
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bins", type=parse_bins, default=[2, 5, 10])
    parser.add_argument("--filter", default="BM_CpuVoxelization")
    parser.add_argument("--min-time", default="2s")
    parser.add_argument("--warmup", default="1")
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--matrix-config", type=Path)
    parser.add_argument("--matrix-case")
    parser.add_argument("--matrix-run-index", type=int)
    parser.add_argument("--matrix-order-index", type=int)
    parser.add_argument("--validation-dir", type=Path,
                        help="Require passing external and precision validation for every bin count")
    args = parser.parse_args()

    if args.repetitions < 1:
        parser.error("--repetitions must be positive")
    if not re.fullmatch(r"[0-9]+(?:\.[0-9]+)?s", args.min_time):
        parser.error("--min-time must be seconds, for example 2s")
    if not re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", args.warmup):
        parser.error("--warmup must be seconds, for example 1")

    source_status = git("status", "--porcelain", "--untracked-files=all", "--",
                        ".", ":(exclude)results/**")
    if source_status and not args.allow_dirty:
        parser.error("Source tree is dirty; commit it or pass --allow-dirty for an exploratory run")
    input_path = args.input.resolve(strict=True)
    executable = args.executable.resolve(strict=True)
    input_sha = sha256(input_path)
    preparation_path = input_path.parent / "metadata.json"
    preparation = json.loads(preparation_path.read_text())
    if preparation.get("binary_sha256") != input_sha:
        raise ValueError("Prepared metadata does not match the benchmark input")
    cpu_executable = executable.parent / "event_voxel_run_cpu"
    cpu_executable_sha = sha256(cpu_executable)
    validations = (validation_reports(args.validation_dir, args.bins, input_sha,
                                      cpu_executable_sha)
                   if args.validation_dir else {})
    cache = executable.parent / "CMakeCache.txt"
    cache_values = {}
    if cache.is_file():
        for line in cache.read_text(errors="replace").splitlines():
            if line.startswith(("CMAKE_CXX_COMPILER:", "CMAKE_CXX_FLAGS_RELEASE:",
                                "CMAKE_BUILD_TYPE:")):
                key, value = line.split("=", 1)
                cache_values[key] = value
    cpu_info = subprocess.run(["lscpu"], capture_output=True, text=True)
    run_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    run_dir = args.output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    output_json = run_dir / "benchmark.json"
    if validations:
        (run_dir / "validation").mkdir()
        for bin_count, case in validations.items():
            for kind, record in case.items():
                suffix = ".precision.json" if kind == "precision" else ".json"
                copy = run_dir / "validation" / f"bins_{bin_count}{suffix}"
                shutil.copy2(record["path"], copy)
                record["path"] = str(copy.relative_to(run_dir))

    command = [
        str(executable),
        f"--benchmark_filter={args.filter}",
        f"--benchmark_min_time={args.min_time}",
        f"--benchmark_min_warmup_time={args.warmup}",
        f"--benchmark_repetitions={args.repetitions}",
        f"--benchmark_out={output_json.resolve()}",
        "--benchmark_out_format=json",
    ]
    manifest = {
        "schema_version": 2,
        "status": "running",
        "run_id": run_id,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source": {
            "git_revision": git("rev-parse", "HEAD"),
            "dirty": bool(source_status),
            "git_status": source_status.splitlines(),
            "file_sha256": source_hashes(),
        },
        "dataset": {"path": str(input_path), "sha256": input_sha,
                    **prepared_header(input_path), "preparation": preparation},
        "executable": {"path": str(executable), "sha256": sha256(executable)},
        "cpu_executable": {"path": str(cpu_executable),
                           "sha256": cpu_executable_sha},
        "validations": validations,
        "build": cache_values,
        "parameters": {
            "bins": args.bins,
            "filter": args.filter,
            "min_time": args.min_time,
            "warmup_seconds": float(args.warmup),
            "repetitions": args.repetitions,
        },
        "system": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
            "load_average_at_start": os.getloadavg(),
            "lscpu": cpu_info.stdout if cpu_info.returncode == 0 else None,
        },
        "command": command,
    }
    if args.matrix_config:
        manifest["matrix"] = {
            "config_path": str(args.matrix_config.resolve(strict=True)),
            "config_sha256": sha256(args.matrix_config),
            "case": args.matrix_case,
            "run_index": args.matrix_run_index,
            "order_index": args.matrix_order_index,
        }
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    environment = os.environ.copy()
    environment["EVENT_VOXEL_BENCH_INPUT"] = str(input_path)
    environment["EVENT_VOXEL_BENCH_BINS"] = ",".join(map(str, args.bins))
    result = subprocess.run(command, env=environment, capture_output=True, text=True)
    (run_dir / "benchmark.log").write_text(result.stdout + result.stderr)
    manifest["status"] = "completed" if result.returncode == 0 else "failed"
    manifest["exit_code"] = result.returncode
    manifest["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    manifest["system"]["load_average_at_end"] = os.getloadavg()
    if output_json.is_file():
        manifest["benchmark_sha256"] = sha256(output_json)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    print(f"Run folder: {run_dir}")
    if result.returncode:
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
