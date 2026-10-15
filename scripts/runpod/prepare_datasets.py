#!/usr/bin/env python3
"""Download or unpack the real-data bundle, then verify every required input."""

import argparse
import json
import os
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from run_matrix import require_config, sha256_file  # noqa: E402


def required_files(config_paths):
    selected = {}
    for config_path in config_paths:
        for dataset in require_config(config_path)["datasets"]:
            previous = selected.setdefault(dataset["path"], dataset["sha256"])
            if previous != dataset["sha256"]:
                raise ValueError(f"conflicting hashes for {dataset['path']}")
    return selected


def verify(root, selected):
    root = root.resolve(strict=True)
    for relative, expected_hash in selected.items():
        events = (root / relative).resolve(strict=True)
        if not events.is_relative_to(root) or sha256_file(events) != expected_hash:
            raise ValueError(f"dataset hash mismatch: {relative}")
        metadata = events.with_name("metadata.json")
        if json.loads(metadata.read_text())["binary_sha256"] != expected_hash:
            raise ValueError(f"metadata hash mismatch: {relative}")
    print(f"Verified {len(selected)} real datasets in {root}")


def verify_present(root, selected):
    """Reject changed files before extending an existing smoke dataset root."""
    for relative, expected_hash in selected.items():
        events = root / relative
        metadata = events.with_name("metadata.json")
        if not events.exists() and not metadata.exists():
            continue
        if (not events.is_file() or not metadata.is_file() or
                not events.resolve().is_relative_to(root.resolve()) or
                not metadata.resolve().is_relative_to(root.resolve()) or
                sha256_file(events) != expected_hash or
                json.loads(metadata.read_text())["binary_sha256"] != expected_hash):
            raise ValueError(f"existing dataset differs from config: {relative}")


def unpack(archive_path, target, selected):
    expected = set(selected)
    expected |= {str(Path(relative).with_name("metadata.json")) for relative in selected}
    seen = set()
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive:
            if not member.isfile() or member.name not in expected or member.name in seen:
                raise ValueError(f"unexpected archive member: {member.name}")
            if member.size > 2_000_000_000:
                raise ValueError(f"archive member is too large: {member.name}")
            seen.add(member.name)
            output = target / member.name
            output.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                raise ValueError(f"cannot extract {member.name}")
            with source, output.open("wb") as destination:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    destination.write(block)
    if seen != expected:
        raise ValueError(f"archive is missing {sorted(expected - seen)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, action="append", required=True)
    parser.add_argument("--destination", type=Path, required=True)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--archive", type=Path)
    source.add_argument("--url")
    args = parser.parse_args()
    selected = required_files(args.config)
    if args.destination.exists():
        verify_present(args.destination, selected)
        try:
            verify(args.destination, selected)
            return
        except FileNotFoundError:
            pass
    if args.archive is None and args.url is None:
        raise ValueError("dataset root is missing; provide --archive or --url")
    if args.url is not None and not args.url.startswith("https://"):
        raise ValueError("dataset URL must use HTTPS")

    args.destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.destination.parent) as temporary:
        staging = Path(temporary) / "datasets"
        staging.mkdir()
        archive_path = args.archive
        if args.url is not None:
            archive_path = Path(temporary) / "download.tar.gz"
            with urllib.request.urlopen(args.url, timeout=60) as response, archive_path.open("wb") as output:
                for block in iter(lambda: response.read(1024 * 1024), b""):
                    output.write(block)
        unpack(archive_path, staging, selected)
        verify(staging, selected)
        if args.destination.exists():
            for relative in selected:
                events = args.destination / relative
                if events.exists():
                    continue
                if not events.parent.resolve().is_relative_to(args.destination.resolve()):
                    raise ValueError(f"dataset path leaves destination: {relative}")
                events.parent.mkdir(parents=True, exist_ok=True)
                os.replace(staging / relative, events)
                os.replace(staging / Path(relative).with_name("metadata.json"),
                           events.with_name("metadata.json"))
        else:
            os.replace(staging, args.destination)
    verify(args.destination, selected)
    print(f"Ready: {args.destination}")


if __name__ == "__main__":
    main()
