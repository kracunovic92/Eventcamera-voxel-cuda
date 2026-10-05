#!/usr/bin/env python3
"""Compare every real benchmark case with the CPU oracle before timing."""

import argparse
import subprocess
from pathlib import Path

from run_matrix import require_config, sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--executable", type=Path, required=True)
    args = parser.parse_args()
    config = require_config(args.config)
    root = args.dataset_root.resolve(strict=True)
    executable = args.executable.resolve(strict=True)
    for dataset in config["datasets"]:
        path = (root / dataset["path"]).resolve(strict=True)
        if not path.is_relative_to(root) or sha256_file(path) != dataset["sha256"]:
            raise ValueError(f"dataset hash mismatch: {dataset['id']}")
        for count in dataset["event_counts"]:
            for bins in config["bins"]:
                subprocess.run([str(executable), str(path), str(count), str(bins)], check=True)


if __name__ == "__main__":
    main()
