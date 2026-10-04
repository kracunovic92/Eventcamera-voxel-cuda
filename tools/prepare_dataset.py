
#!/usr/bin/env python3
"""Prepare a verified Prophesee HDF5 recording for C++.

Run from the repository root after configuring the ECF plugin.
"""

import argparse
import hashlib
import json
import os
import struct
from pathlib import Path

import h5py
import numpy as np


MAGIC = b"EVXEVT01"
HEADER = struct.Struct("<8sIIQqq")
CHUNK_SIZE = 100_000

# Explicit field offsets prevent compiler or NumPy alignment
# choices from changing our documented binary format.
EVENT_DTYPE = np.dtype({
    "names": ["x", "y", "p", "t"],
    "formats": ["<u2", "<u2", "<i2", "<i8"],
    "offsets": [0, 2, 4, 8],
    "itemsize": 16,
})


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as source:
        for block in iter(
            lambda: source.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def load_verified_report(
    recording: Path,
    report_path: Path,
) -> dict:
    """Require a valid report matching the current source."""

    report = json.loads(
        report_path.read_text(encoding="utf-8")
    )

    if report.get("schema_version") != 1:
        raise ValueError("Unsupported inspection schema")

    if report.get("status") != "valid":
        raise ValueError("Recording has not passed inspection")

    source = report["source"]
    current = recording.stat()

    if Path(source["path"]).resolve() != recording.resolve():
        raise ValueError("Report belongs to another recording")

    if (
        source["size_bytes"] != current.st_size
        or source["mtime_ns"] != current.st_mtime_ns
    ):
        raise ValueError(
            "Recording changed. Run the inspector again."
        )

    if source["sha256"] != sha256_file(recording):
        raise ValueError(
            "Recording fingerprint does not match inspection"
        )

    return report


def prepare(
    recording: Path,
    report: dict,
    output_dir: Path,
    chunk_size: int,
    window_start_us: int | None = None,
    window_end_us: int | None = None,
    max_events: int | None = None,
    start_index: int = 0,
) -> None:
    """Export a full recording or a selected ordered event window."""

    metadata = report["recording"]
    stats = report["statistics"]

    width = int(metadata["width"])
    height = int(metadata["height"])
    source_count = int(stats["event_count"])

    start_us = stats["min_timestamp_us"]
    maximum_us = stats["max_timestamp_us"]

    if source_count <= 0 or start_us is None or maximum_us is None:
        raise ValueError("Recording contains no valid events")

    # Our C++ timestamps use signed 64-bit integers.
    int64_max = (1 << 63) - 1

    if not 0 <= start_us <= maximum_us < int64_max:
        raise ValueError("Unsupported recording timestamp range")

    if window_start_us is not None:
        start_us = window_start_us
    end_us = window_end_us if window_end_us is not None else maximum_us + 1
    if start_us < 0 or end_us <= start_us or end_us > int64_max:
        raise ValueError("Invalid selected temporal window")
    if max_events is not None and max_events <= 0:
        raise ValueError("--max-events must be positive")
    if not 0 <= start_index < source_count:
        raise ValueError("--start-index is outside the recording")

    output_dir.mkdir(parents=True, exist_ok=True)

    destination = output_dir / "events.bin"
    temporary = output_dir / "events.bin.tmp"

    try:
        written = 0
        previous_timestamp = None
        last_selected_timestamp = None
        first_selected_timestamp = None

        with (
            h5py.File(recording, "r") as source,
            temporary.open("wb") as output,
        ):
            events = source["CD/events"]

            if len(events) != source_count:
                raise ValueError("Event count differs from report")

            output.write(b"\0" * HEADER.size)

            for offset in range(start_index, source_count, chunk_size):
                chunk = events[offset:offset + chunk_size]

                # Recheck essential properties during conversion.
                if (
                    np.any(chunk["x"] >= width)
                    or np.any(chunk["y"] >= height)
                    or np.any(
                        (chunk["p"] != 0) & (chunk["p"] != 1)
                    )
                    or np.any(chunk["t"] < stats["min_timestamp_us"])
                    or np.any(chunk["t"] > maximum_us)
                    or np.any(chunk["t"][1:] < chunk["t"][:-1])
                ):
                    raise ValueError("Invalid event data")

                if (
                    previous_timestamp is not None
                    and len(chunk)
                    and chunk["t"][0] < previous_timestamp
                ):
                    raise ValueError("Timestamps out of order")

                if len(chunk):
                    previous_timestamp = int(chunk["t"][-1])

                selected = chunk[
                    (chunk["t"] >= start_us) & (chunk["t"] < end_us)
                ]
                if max_events is not None:
                    selected = selected[:max_events - written]
                if len(selected) == 0:
                    if max_events is not None and written == max_events:
                        break
                    continue
                if first_selected_timestamp is None:
                    first_selected_timestamp = int(selected["t"][0])

                # Initialize all padding bytes to zero.
                prepared = np.zeros(len(selected), dtype=EVENT_DTYPE)

                for field in ("x", "y", "p", "t"):
                    prepared[field] = selected[field]

                output.write(prepared.tobytes())
                written += len(selected)
                last_selected_timestamp = int(selected["t"][-1])
                if max_events is not None and written == max_events:
                    break

            if written == 0:
                raise ValueError("Selected window contains no events")

            if max_events is not None and window_end_us is None:
                end_us = last_selected_timestamp + 1
            if start_index and window_start_us is None:
                start_us = first_selected_timestamp

            output.seek(0)
            output.write(HEADER.pack(
                MAGIC, width, height, written, start_us, end_us
            ))

        expected_bytes = HEADER.size + written * EVENT_DTYPE.itemsize

        if temporary.stat().st_size != expected_bytes:
            raise ValueError("Unexpected output file size")

        os.replace(temporary, destination)

        prepared_metadata = {
            "format": "EVXEVT01",
            "source_file": str(recording.resolve()),
            "source_sha256": report["source"]["sha256"],
            "width": width,
            "height": height,
            "event_count": written,
            "source_event_count": source_count,
            "start_us": start_us,
            "end_us": end_us,
            "selection": {
                "start_us": window_start_us,
                "end_us": window_end_us,
                "max_events": max_events,
                "start_index": start_index,
            },
            "event_record_bytes": EVENT_DTYPE.itemsize,
            "binary_sha256": sha256_file(destination),
        }

        metadata_temporary = output_dir / "metadata.json.tmp"
        metadata_temporary.write_text(
            json.dumps(
                prepared_metadata,
                indent=2,
                sort_keys=True,
            ) + "\n",
            encoding="utf-8",
        )
        os.replace(metadata_temporary, output_dir / "metadata.json")

        print(f"Prepared {written} events")
        print(f"Output: {destination}")

    finally:
        temporary.unlink(missing_ok=True)
        (output_dir / "metadata.json.tmp").unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare inspected HDF5 events for C++"
    )

    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=CHUNK_SIZE,
    )
    parser.add_argument("--start-us", type=int)
    parser.add_argument("--end-us", type=int)
    parser.add_argument("--max-events", type=int)
    parser.add_argument("--start-index", type=int, default=0)

    args = parser.parse_args()

    if args.chunk_size <= 0:
        parser.error("--chunk-size must be positive")
    if args.max_events is not None and args.max_events <= 0:
        parser.error("--max-events must be positive")
    if args.start_index < 0:
        parser.error("--start-index must be nonnegative")

    report = load_verified_report(
        args.input.resolve(),
        args.report,
    )

    prepare(
        args.input,
        report,
        args.output,
        args.chunk_size,
        args.start_us,
        args.end_us,
        args.max_events,
        args.start_index,
    )


if __name__ == "__main__":
    main()
