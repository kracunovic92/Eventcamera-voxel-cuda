#!/usr/bin/env python3
"""Create a hashed synthetic EVXEVT01 benchmark workload."""

import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np


EVENT_DTYPE = np.dtype({
    "names": ["x", "y", "p", "t"],
    "formats": ["<u2", "<u2", "<i2", "<i8"],
    "offsets": [0, 2, 4, 8],
    "itemsize": 16,
})


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--events", type=int, required=True)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--start-us", type=int, default=0)
    parser.add_argument("--duration-us", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--spatial", choices=["uniform", "hotspot", "single_pixel"],
                        default="uniform")
    parser.add_argument("--temporal", choices=["uniform", "burst"], default="uniform")
    parser.add_argument("--hotspot-fraction", type=float, default=0.1)
    parser.add_argument("--hotspot-probability", type=float, default=0.9)
    parser.add_argument("--positive-probability", type=float, default=0.5)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.events <= 0 or not 1 <= args.width <= 65536 or not 1 <= args.height <= 65536:
        parser.error("Events must be positive and dimensions must be in [1, 65536]")
    if (args.start_us < 0 or args.duration_us <= 0 or
        args.start_us + args.duration_us > (1 << 63) - 1):
        parser.error("Invalid signed 64-bit temporal interval")
    if args.seed < 0 or args.seed >= 1 << 64:
        parser.error("Seed must fit in uint64")
    if (not 0 < args.hotspot_fraction <= 1 or
        not 0 <= args.hotspot_probability <= 1 or
        not 0 <= args.positive_probability <= 1):
        parser.error("Probabilities and hotspot fraction are out of range")

    binary = args.output / "events.bin"
    metadata = args.output / "metadata.json"
    if (binary.exists() or metadata.exists()) and not args.force:
        parser.error("Output exists; choose another case name or pass --force")
    args.output.mkdir(parents=True, exist_ok=True)
    rng = np.random.Generator(np.random.PCG64(args.seed))
    count = args.events
    x = rng.integers(0, args.width, count, dtype=np.uint16)
    y = rng.integers(0, args.height, count, dtype=np.uint16)

    if args.spatial == "single_pixel":
        x[:] = args.width // 2
        y[:] = args.height // 2
    elif args.spatial == "hotspot":
        hot_width = max(1, int(args.width * args.hotspot_fraction))
        hot_height = max(1, int(args.height * args.hotspot_fraction))
        hot_x = (args.width - hot_width) // 2
        hot_y = (args.height - hot_height) // 2
        selected = rng.random(count) < args.hotspot_probability
        selected_count = int(selected.sum())
        x[selected] = rng.integers(hot_x, hot_x + hot_width,
                                   selected_count, dtype=np.uint16)
        y[selected] = rng.integers(hot_y, hot_y + hot_height,
                                   selected_count, dtype=np.uint16)

    timestamps = rng.integers(
        args.start_us, args.start_us + args.duration_us,
        count, dtype=np.int64)
    if args.temporal == "burst":
        selected = rng.random(count) < 0.9
        timestamps[selected] = rng.integers(
            args.start_us, args.start_us + max(1, args.duration_us // 10),
            int(selected.sum()), dtype=np.int64)
    polarity = (rng.random(count) < args.positive_probability).astype(np.int16)
    order = np.argsort(timestamps, kind="stable")
    events = np.zeros(count, dtype=EVENT_DTYPE)
    events["x"] = x[order]
    events["y"] = y[order]
    events["p"] = polarity[order]
    events["t"] = timestamps[order]

    with binary.open("wb") as stream:
        stream.write(struct.pack(
            "<8sIIQqq", b"EVXEVT01", args.width, args.height, count,
            args.start_us, args.start_us + args.duration_us))
        stream.write(events.tobytes())
    configuration = {key: value for key, value in vars(args).items()
                     if key not in {"output", "force"}}
    metadata.write_text(json.dumps({
        "format": "EVXEVT01",
        "generator": "numpy.PCG64",
        "numpy_version": np.__version__,
        "configuration": configuration,
        "binary_sha256": sha256(binary),
    }, indent=2, sort_keys=True) + "\n")
    print(f"Synthetic recording: {binary}")
    print(f"SHA-256: {sha256(binary)}")


if __name__ == "__main__":
    main()
