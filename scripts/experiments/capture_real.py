#!/usr/bin/env python3
"""Save CPU and, when requested, CUDA voxel arrays for every matrix case."""

import argparse
import json
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path

from run_matrix import require_config, sha256_file


def git(*arguments):
    result = subprocess.run(["git", *arguments], capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--cpu-executable", type=Path, required=True)
    parser.add_argument("--cuda-executable", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-path-file", type=Path)
    args = parser.parse_args()
    config = require_config(args.config)
    dataset_root = args.dataset_root.resolve(strict=True)
    cpu = args.cpu_executable.resolve(strict=True)
    cuda = args.cuda_executable.resolve(strict=True) if args.cuda_executable else None
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    run_dir = args.output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    if args.run_path_file:
        args.run_path_file.write_text(str(run_dir.resolve()) + "\n")
    manifest = {
        "schema_version": 1,
        "status": "running",
        "run_id": run_id,
        "config_sha256": sha256_file(args.config),
        "git_revision": git("rev-parse", "HEAD"),
        "git_status": git("status", "--porcelain", "--untracked-files=normal", "--",
                          ".", ":(exclude)results/**"),
        "cpu_executable_sha256": sha256_file(cpu),
        "cuda_executable_sha256": sha256_file(cuda) if cuda else None,
        "cases": [],
    }
    (run_dir / "config.json").write_text(args.config.read_text())

    def save():
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    def run(executable, arguments, case_dir, name, backend, strategy, count, bins):
        output = case_dir / f"{name}.npy"
        metrics = case_dir / f"{name}.json"
        subprocess.run([str(executable), *arguments[:2], str(output), *arguments[2:],
                        "--count", str(count), "--metrics", str(metrics)], check=True)
        details = json.loads(metrics.read_text())
        if (details["backend"] != backend or details["strategy"] != strategy or
                details["selected_events"] != count or details["bins"] != bins):
            raise ValueError(f"run metadata differs from requested case: {metrics}")
        return {
            "output": str(output.relative_to(run_dir)),
            "output_sha256": sha256_file(output),
            "metrics": str(metrics.relative_to(run_dir)),
            "metrics_sha256": sha256_file(metrics),
        }

    save()
    try:
        for dataset in config["datasets"]:
            events = (dataset_root / dataset["path"]).resolve(strict=True)
            if not events.is_relative_to(dataset_root) or sha256_file(events) != dataset["sha256"]:
                raise ValueError(f"dataset hash mismatch: {dataset['id']}")
            preparation = json.loads(events.with_name("metadata.json").read_text())
            if preparation.get("binary_sha256") != dataset["sha256"]:
                raise ValueError(f"dataset metadata mismatch: {dataset['id']}")
            for count in dataset["event_counts"]:
                for bins in config["bins"]:
                    case_dir = run_dir / dataset["id"] / str(count) / str(bins)
                    case_dir.mkdir(parents=True)
                    case = {
                        "dataset": dataset["id"], "dataset_sha256": dataset["sha256"],
                        "event_count": count, "bins": bins,
                        "cpu": run(cpu, [str(events), str(bins)], case_dir, "cpu",
                                   "cpu", "reference", count, bins),
                        "gpu": {},
                    }
                    manifest["cases"].append(case)
                    save()
                    if cuda is not None:
                        for strategy in config["strategies"]:
                            case["gpu"][strategy] = run(
                                cuda, [str(events), str(bins), "--strategy", strategy],
                                case_dir, strategy, "cuda", strategy, count, bins)
                            save()
                    print(f"Captured {dataset['id']}: {count} events, {bins} bins", flush=True)
        manifest["status"] = "complete"
        save()
        print(f"Saved arrays and run metadata: {run_dir}")
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = str(error)
        save()
        raise


if __name__ == "__main__":
    main()
