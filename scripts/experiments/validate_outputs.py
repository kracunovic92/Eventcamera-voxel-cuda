#!/usr/bin/env python3
"""Validate one saved voxel grid or a captured experiment against reference models."""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from output_comparison import analyze_run, artifact
from reference_models import (
    EXPECTED_RPG_REVISION, EXPECTED_RPG_SHA256, calculate_external_reference,
    compare, float64_reference_grid, load_events, load_rpg_function, sha256,
    verify_rpg_source,
)
from run_matrix import require_config, sha256_file


def sample_evenly(events, count):
    """Match src/data/sample_event_batch.cpp, retaining the source time interval."""
    total = len(events)
    if count < 1 or count > total:
        raise ValueError(f"requested {count} events from a {total}-event recording")
    if count == total:
        return events
    if count == 1:
        return events[:1]
    indices = np.fromiter((index * (total - 1) // (count - 1)
                           for index in range(count)), dtype=np.intp, count=count)
    return events[indices]


def references_for(events, metadata, bins, names, rpg_source):
    references = {}
    if "rpg" in names:
        function, adaptations = load_rpg_function(
            rpg_source, metadata["end_us"] - metadata["start_us"])
        references["rpg"] = (
            calculate_external_reference(events, metadata, bins, function),
            {"revision": EXPECTED_RPG_REVISION, "source_sha256": EXPECTED_RPG_SHA256,
             "adaptations": adaptations},
        )
    if "precision" in names:
        references["precision"] = (
            float64_reference_grid(events, metadata, bins),
            {"method": "float64 triangular-weight accumulation"},
        )
    return references


def checked_output(path, metadata, bins):
    actual = np.load(path, mmap_mode="r", allow_pickle=False)
    if (actual.dtype != np.float32 or actual.shape !=
            (bins, metadata["height"], metadata["width"])):
        raise ValueError(f"expected float32 [bins,height,width] output: {path}")
    return actual


def run_single(args, rpg_source):
    events_path = args.events.resolve(strict=True)
    output_path = args.voxels.resolve(strict=True)
    events, metadata = load_events(events_path)
    actual = np.load(output_path, mmap_mode="r", allow_pickle=False)
    if actual.ndim != 3 or actual.shape[0] < 2:
        raise ValueError(f"expected a 3D grid with at least two bins: {output_path}")
    bins = actual.shape[0]
    actual = checked_output(output_path, metadata, bins)
    results = {}
    for name, (reference, provenance) in references_for(
            events, metadata, bins, args.references, rpg_source).items():
        results[name] = {"result": compare(actual, reference, args.atol, args.rtol),
                         "provenance": provenance}
    passed = all(entry["result"]["passed"] for entry in results.values())
    executable = args.executable.resolve(strict=True) if args.executable else None
    report = {
        "schema_version": 2,
        "passed": passed,
        "backend": args.backend,
        "strategy": args.strategy,
        "recording": {**metadata, "path": str(events_path), "sha256": sha256(events_path)},
        "output": {"path": str(output_path), "sha256": sha256(output_path),
                   "shape": list(actual.shape)},
        "executable": ({"path": str(executable), "sha256": sha256(executable)}
                       if executable else None),
        "references": results,
        "numpy_version": np.__version__,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    for name, entry in results.items():
        metrics = entry["result"]
        print(f"{'PASS' if metrics['passed'] else 'FAIL'} {args.backend}/{args.strategy} "
              f"versus {name}: {metrics['failed_voxels']} / {metrics['voxel_count']} voxels")
    print(f"Validation report: {args.report}")
    return 0 if passed else 1


def run_capture(args, rpg_source):
    root = args.run_dir.resolve(strict=True)
    dataset_root = args.dataset_root.resolve(strict=True)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("schema_version") != 1 or manifest.get("status") != "complete":
        raise ValueError("captured run is incomplete")
    config_path = root / "config.json"
    config = require_config(config_path)
    if sha256_file(config_path) != manifest["config_sha256"]:
        raise ValueError("captured config differs from its manifest")
    datasets = {entry["id"]: entry for entry in config["datasets"]}
    expected_cases = {(entry["id"], count, bins)
                      for entry in config["datasets"]
                      for count in entry["event_counts"] for bins in config["bins"]}
    expected_strategies = (set(config["strategies"])
                           if manifest.get("cuda_executable_sha256") else set())
    rows = []
    seen_cases = set()
    active_dataset = None
    for case in manifest["cases"]:
        count, bins = case["event_count"], case["bins"]
        case_key = (case["dataset"], count, bins)
        if case_key not in expected_cases or case_key in seen_cases:
            raise ValueError(f"unexpected or duplicate captured case: {case_key}")
        seen_cases.add(case_key)
        definition = datasets[case["dataset"]]
        if (case["dataset_sha256"] != definition["sha256"] or
                set(case["gpu"]) != expected_strategies):
            raise ValueError(f"captured case differs from config: {case['dataset']}")
        path = (dataset_root / definition["path"]).resolve(strict=True)
        if not path.is_relative_to(dataset_root) or sha256_file(path) != definition["sha256"]:
            raise ValueError(f"dataset hash mismatch: {definition['id']}")
        if active_dataset != path:
            events, metadata = load_events(path)
            active_dataset = path
        selected = sample_evenly(events, count)
        references = references_for(selected, metadata, bins, args.references, rpg_source)
        outputs = [("cpu", "reference", case["cpu"])]
        outputs.extend(("cuda", strategy, record) for strategy, record in case["gpu"].items())
        for backend, strategy, record in outputs:
            output, run_metrics = artifact(root, record)
            if (run_metrics["backend"] != backend or run_metrics["strategy"] != strategy or
                    run_metrics["selected_events"] != count or run_metrics["bins"] != bins or
                    run_metrics["width"] != metadata["width"] or
                    run_metrics["height"] != metadata["height"] or
                    run_metrics["start_us"] != metadata["start_us"] or
                    run_metrics["end_us"] != metadata["end_us"]):
                raise ValueError(f"run record differs from requested case: {output}")
            actual = checked_output(output, metadata, bins)
            for name, (reference, provenance) in references.items():
                result = compare(actual, reference, args.atol, args.rtol)
                worst = result["worst_voxel"]
                rows.append({
                    "dataset": case["dataset"], "dataset_sha256": definition["sha256"],
                    "event_count": count, "bins": bins, "backend": backend,
                    "strategy": strategy, "reference": name,
                    "output_sha256": record["output_sha256"],
                    "rpg_revision": provenance.get("revision", ""),
                    "rpg_source_sha256": provenance.get("source_sha256", ""),
                    "numpy_version": np.__version__,
                    "voxelize_host_ms": run_metrics["voxelize_ms"],
                    "load_ms": run_metrics["load_ms"],
                    "export_ms": run_metrics["export_ms"],
                    "output_bytes": run_metrics["output_bytes"],
                    "peak_process_rss_kib": run_metrics["peak_process_rss_kib"],
                    "gpu_total_bytes": run_metrics["gpu_total_bytes"],
                    "gpu_free_before_bytes": run_metrics["gpu_free_before_bytes"],
                    "gpu_free_after_bytes": run_metrics["gpu_free_after_bytes"],
                    "passed": result["passed"], "voxel_count": result["voxel_count"],
                    "failed_voxels": result["failed_voxels"],
                    "max_absolute_error": result["max_absolute_error"],
                    "mean_absolute_error": result["mean_absolute_error"],
                    "rmse": result["rmse"],
                    "worst_bin": worst["bin"], "worst_y": worst["y"], "worst_x": worst["x"],
                    "actual_value": worst["actual_value"],
                    "reference_value": worst["reference_value"],
                    "atol": args.atol, "rtol": args.rtol,
                })
                print(f"{'PASS' if result['passed'] else 'FAIL'} {case['dataset']} "
                      f"{count} events {bins} bins {backend}/{strategy} versus {name}",
                      flush=True)
    if seen_cases != expected_cases:
        raise ValueError(f"captured run is missing {len(expected_cases - seen_cases)} cases")
    destination = args.output or root / "validation.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    failures = sum(not row["passed"] for row in rows)
    print(f"Validated {len(rows)} outputs against references: "
          f"{failures} failed. CSV: {destination}")
    return 1 if failures else 0


def compare_backends(args):
    rows = analyze_run(args.run_dir, args.atol, args.rtol)
    destination = args.output or args.run_dir / "comparisons.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    failures = sum(not row["passed"] for row in rows)
    print(f"Compared {len(rows)} CPU/CUDA pairs: {failures} failed. CSV: {destination}")
    return 1 if failures else 0


def main():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--references", nargs="+", choices=("rpg", "precision"),
                        default=["rpg", "precision"])
    common.add_argument("--rpg-repo", type=Path, default=Path("external/rpg_e2depth"))
    common.add_argument("--atol", type=float, default=1e-3)
    common.add_argument("--rtol", type=float, default=1e-4)
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    single = commands.add_parser("single", parents=[common], help="Check one saved grid")
    single.add_argument("--events", type=Path, required=True)
    single.add_argument("--voxels", type=Path, required=True)
    single.add_argument("--report", type=Path, required=True)
    single.add_argument("--executable", type=Path)
    single.add_argument("--backend", choices=("cpu", "cuda"), default="cpu")
    single.add_argument("--strategy", default="reference")
    capture = commands.add_parser("capture", parents=[common],
                                  help="Check every saved grid in a captured run")
    capture.add_argument("run_dir", type=Path)
    capture.add_argument("--dataset-root", type=Path, required=True)
    capture.add_argument("--output", type=Path)
    backends = commands.add_parser("compare-backends",
                                   help="Optional CPU versus CUDA diagnostic from saved grids")
    backends.add_argument("run_dir", type=Path)
    backends.add_argument("--output", type=Path)
    backends.add_argument("--atol", type=float, default=1e-4)
    backends.add_argument("--rtol", type=float, default=1e-5)
    args = parser.parse_args()
    if args.command == "compare-backends":
        if any(not math.isfinite(value) or value < 0 for value in (args.atol, args.rtol)):
            parser.error("tolerances must be finite and nonnegative")
        return compare_backends(args)
    if len(args.references) != len(set(args.references)):
        parser.error("references must not contain duplicates")
    if any(not math.isfinite(value) or value < 0 for value in (args.atol, args.rtol)):
        parser.error("tolerances must be finite and nonnegative")
    if args.command == "single" and args.backend == "cuda" and args.strategy == "reference":
        parser.error("CUDA validation requires --strategy")
    source = (verify_rpg_source(args.rpg_repo.resolve(strict=True))
              if "rpg" in args.references else None)
    return run_single(args, source) if args.command == "single" else run_capture(args, source)


if __name__ == "__main__":
    raise SystemExit(main())
