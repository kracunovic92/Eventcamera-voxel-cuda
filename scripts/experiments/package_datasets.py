#!/usr/bin/env python3
"""Bundle the exact prepared real-data inputs named by CUDA benchmark configs."""

import argparse
import json
import tarfile
from pathlib import Path

from run_matrix import require_config, sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, action="append", required=True)
    parser.add_argument("--source-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = args.source_root.resolve(strict=True)
    selected = {}
    for config_path in args.config:
        for dataset in require_config(config_path)["datasets"]:
            previous = selected.setdefault(dataset["path"], dataset["sha256"])
            if previous != dataset["sha256"]:
                raise ValueError(f"conflicting hashes for {dataset['path']}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.output, "x:gz") as archive:
        for relative, expected_hash in sorted(selected.items()):
            events = (root / relative).resolve(strict=True)
            if not events.is_relative_to(root) or not events.is_file():
                raise ValueError(f"invalid dataset path: {relative}")
            metadata = events.with_name("metadata.json")
            if sha256_file(events) != expected_hash:
                raise ValueError(f"dataset hash mismatch: {relative}")
            if json.loads(metadata.read_text())["binary_sha256"] != expected_hash:
                raise ValueError(f"metadata hash mismatch: {relative}")
            archive.add(events, arcname=relative, recursive=False)
            archive.add(metadata, arcname=str(Path(relative).with_name("metadata.json")), recursive=False)
            print(f"Added {relative}")
    print(f"Archive: {args.output}")
    print(f"SHA-256: {sha256_file(args.output)}")


if __name__ == "__main__":
    main()
