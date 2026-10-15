#!/usr/bin/env python3
"""Check and summarize real-recording validation coverage."""

import argparse
import hashlib
import json
import re
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-dir", type=Path, default=Path("results/metadata"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--validation-dir", type=Path, default=Path("results/validation"))
    parser.add_argument("--window-id", required=True)
    parser.add_argument("--bins", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    raw_files = sorted(args.raw_dir.glob("*.hdf5"))
    if not raw_files:
        parser.error("No raw recordings found")
    summary = {"schema_version": 2, "window_id": args.window_id,
               "bins": args.bins, "datasets": []}
    for raw_path in raw_files:
        dataset = raw_path.stem
        inspection_path = args.metadata_dir / f"{dataset}.metadata.json"
        inspection = json.loads(inspection_path.read_text())
        if (inspection["status"] != "valid" or
            inspection["source"]["sha256"] != sha256(raw_path)):
            raise ValueError(f"Source inspection failed: {dataset}")
        prepared_path = args.processed_dir / dataset / args.window_id / "metadata.json"
        prepared = json.loads(prepared_path.read_text())
        binary_path = prepared_path.parent / "events.bin"
        if (prepared["source_sha256"] != inspection["source"]["sha256"] or
            prepared["binary_sha256"] != sha256(binary_path) or
            prepared["source_event_count"] != inspection["statistics"]["event_count"] or
            prepared["width"] != inspection["recording"]["width"] or
            prepared["height"] != inspection["recording"]["height"]):
            raise ValueError(f"Prepared source differs from inspection: {dataset}")
        sample = re.fullmatch(r"(first|middle|tail)_(\d+)", args.window_id)
        if sample:
            position, count_text = sample.groups()
            count = int(count_text)
            source_count = prepared["source_event_count"]
            expected_index = {"first": 0, "middle": (source_count - count) // 2,
                              "tail": source_count - count}[position]
            if (count > source_count or prepared["event_count"] != count or
                prepared["selection"].get("start_index", 0) != expected_index or
                prepared["selection"].get("max_events") != count):
                raise ValueError(f"Selection does not match {args.window_id}: {dataset}")
        cases = []
        for bin_count in args.bins:
            path = args.validation_dir / dataset / args.window_id / f"bins_{bin_count}.validation.json"
            report = json.loads(path.read_text())
            references = report.get("references", {})
            if (report.get("schema_version") != 2 or not report.get("passed") or
                report.get("backend") != "cpu" or report.get("strategy") != "reference" or
                set(references) != {"rpg", "precision"} or
                any(not item["result"]["passed"] for item in references.values()) or
                report["recording"]["sha256"] != prepared["binary_sha256"] or
                report["output"]["shape"][0] != bin_count or
                not report.get("executable")):
                raise ValueError(f"Validation mismatch: {path}")
            cases.append({
                "bins": bin_count,
                "report": str(path),
                "report_sha256": sha256(path),
                "rpg_max_absolute_error": references["rpg"]["result"]["max_absolute_error"],
                "rpg_failed_voxels": references["rpg"]["result"]["failed_voxels"],
                "precision_max_absolute_error": references["precision"]["result"]["max_absolute_error"],
                "precision_failed_voxels": references["precision"]["result"]["failed_voxels"],
                "cpu_output_sha256": report["output"]["sha256"],
                "cpu_executable_sha256": report["executable"]["sha256"],
            })
        summary["datasets"].append({
            "name": dataset,
            "source_sha256": prepared["source_sha256"],
            "binary_sha256": prepared["binary_sha256"],
            "event_count": prepared["event_count"],
            "start_us": prepared["start_us"],
            "end_us": prepared["end_us"],
            "cases": cases,
        })
    summary["passed_cases"] = sum(len(item["cases"]) for item in summary["datasets"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Verified {len(summary['datasets'])} recordings and {summary['passed_cases']} cases")
    print(f"Summary: {args.output}")


if __name__ == "__main__":
    main()
