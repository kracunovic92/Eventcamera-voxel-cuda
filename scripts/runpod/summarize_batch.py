#!/usr/bin/env python3
"""Build one-row-per-run and one-row-per-stage tables from a RunPod batch."""

import csv
import json
import statistics
import sys
from collections import Counter
from pathlib import Path


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} results/runpod/batch-NAME", file=sys.stderr)
        return 2
    batch = Path(sys.argv[1]).resolve()
    if not batch.is_dir() or not (batch / "runs.tsv").is_file():
        print(f"Not a benchmark batch: {batch}", file=sys.stderr)
        return 2

    run_rows = []
    stage_rows = []
    run_dirs = sorted(path for path in batch.glob("run-*") if path.is_dir())
    for run_dir in run_dirs:
        metadata = json.loads((run_dir / "metadata.json").read_text())
        summary = json.loads((run_dir / "summary.json").read_text())
        backend = metadata["backend"]
        stages = summary["stages"]
        prefix = f"{backend}/"
        processing = stages[prefix + "processing-total"]
        processing_total_ms = processing["mean_ms"] * processing["count"]
        with (run_dir / "windows.csv").open(newline="") as stream:
            input_preparation_total_ms = sum(
                float(row["input_preparation_ms"]) for row in csv.DictReader(stream)
            )
        recording = Path(metadata["recording"]).name
        window_us = metadata["window_us"]
        common = {
            "recording": recording,
            "window_us": window_us,
            "window_ms": window_us // 1000 if window_us % 1000 == 0 else window_us / 1000,
            "bins": metadata["bins"],
            "backend": backend,
        }

        def mean(stage: str):
            entry = stages.get(prefix + stage)
            return entry["mean_ms"] if entry is not None else ""

        def total(stage: str):
            entry = stages.get(prefix + stage)
            return entry["mean_ms"] * entry["count"] if entry is not None else ""

        run_rows.append(
            {
                **common,
                "windows": summary["windows"],
                "events": summary["total_events"],
                "run_wall_ms": summary["run_wall_ms"],
                "events_per_second": summary["events_per_second"],
                "input_preparation_median_ms": summary["input_preparation_median_ms"],
                "input_preparation_total_ms": input_preparation_total_ms,
                "processing_total_sum_ms": processing_total_ms,
                "other_run_wall_ms": summary["run_wall_ms"] - input_preparation_total_ms - processing_total_ms,
                "processing_mean_ms": processing["mean_ms"],
                "processing_median_ms": processing["median_ms"],
                "processing_p95_ms": processing["p95_ms"],
                "prepare_mean_ms": mean("prepare"),
                "h2d_mean_ms": mean("h2d-events"),
                "strategy_mean_ms": mean("strategy"),
                "d2h_mean_ms": mean("d2h-grid"),
                "pipeline_mean_ms": mean("pipeline"),
                "transfer_mean_ms": (
                    mean("h2d-events") + mean("d2h-grid") if backend != "cpu" else ""
                ),
                "prepare_total_ms": total("prepare"),
                "h2d_total_ms": total("h2d-events"),
                "strategy_total_ms": total("strategy"),
                "d2h_total_ms": total("d2h-grid"),
                "transfer_total_ms": (
                    total("h2d-events") + total("d2h-grid") if backend != "cpu" else ""
                ),
                "run_directory": run_dir.name,
            }
        )

        samples: dict[str, list[dict]] = {}
        with (run_dir / "samples.csv").open(newline="") as stream:
            for sample in csv.DictReader(stream):
                samples.setdefault(sample["stage"], []).append(sample)
        for full_stage, values in sorted(stages.items()):
            stage = full_stage.removeprefix(prefix)
            sampled = samples.get(stage, [])
            if len(sampled) != values["count"]:
                raise ValueError(f"Sample count mismatch in {run_dir}: {stage}")
            cuda_values = [float(row["cuda_event_ms"]) for row in sampled if row["cuda_event_ms"]]
            bytes_total = sum(int(row["bytes"]) for row in sampled)
            stage_rows.append(
                {
                    **common,
                    "stage": stage,
                    "count": values["count"],
                    "mean_wall_ms": values["mean_ms"],
                    "median_wall_ms": values["median_ms"],
                    "p95_wall_ms": values["p95_ms"],
                    "total_wall_ms": values["mean_ms"] * values["count"],
                    "mean_cuda_event_ms": statistics.mean(cuda_values) if cuda_values else "",
                    "total_bytes": bytes_total,
                    "fraction_of_processing_mean": (
                        values["mean_ms"] / processing["mean_ms"]
                        if processing["mean_ms"] and stage in {"prepare", "h2d-events", "strategy", "d2h-grid"}
                        else ""
                    ),
                    "run_directory": run_dir.name,
                }
            )

    with (batch / "runs.tsv").open(newline="") as stream:
        manifest = list(csv.DictReader(stream, delimiter="\t"))
    if len(run_rows) != len(manifest) or any(row["status"] != "ok" for row in manifest):
        raise ValueError("Batch manifest is incomplete or contains failed runs")
    expected = Counter((row["recording"], int(row["window_us"]), int(row["bins"]), row["backend"]) for row in manifest)
    actual = Counter((row["recording"], row["window_us"], row["bins"], row["backend"]) for row in run_rows)
    if expected != actual:
        raise ValueError("Run metadata does not match the batch manifest")
    run_rows.sort(key=lambda row: (row["recording"], row["window_ms"], row["bins"], row["backend"], row["run_directory"]))
    stage_rows.sort(key=lambda row: (row["recording"], row["window_ms"], row["bins"], row["backend"], row["stage"], row["run_directory"]))
    write_csv(batch / "comparison.csv", run_rows, list(run_rows[0]))
    write_csv(batch / "stage_summary.csv", stage_rows, list(stage_rows[0]))
    print(f"Wrote {len(run_rows)} runs and {len(stage_rows)} stage summaries in {batch}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
