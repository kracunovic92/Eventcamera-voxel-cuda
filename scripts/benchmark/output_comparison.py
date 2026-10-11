"""Read saved voxel arrays and compare them without running voxelization."""

import ast
import json
import math
import struct
import sys
from pathlib import Path

from run_matrix import sha256_file


def artifact(root, record):
    output = (root / record["output"]).resolve(strict=True)
    metrics = (root / record["metrics"]).resolve(strict=True)
    if not output.is_relative_to(root) or not metrics.is_relative_to(root):
        raise ValueError("artifact path leaves the captured run")
    if sha256_file(output) != record["output_sha256"]:
        raise ValueError(f"voxel file hash mismatch: {output}")
    if sha256_file(metrics) != record["metrics_sha256"]:
        raise ValueError(f"metrics file hash mismatch: {metrics}")
    return output, json.loads(metrics.read_text())


def npy_header(path):
    with path.open("rb") as source:
        magic = source.read(8)
        if magic != b"\x93NUMPY\x01\x00":
            raise ValueError(f"expected NumPy v1 array: {path}")
        length = struct.unpack("<H", source.read(2))[0]
        header = ast.literal_eval(source.read(length).decode("latin1"))
        offset = source.tell()
    if (header.get("descr") != "<f4" or header.get("fortran_order") is not False or
            len(header.get("shape", ())) != 3 or
            any(type(value) is not int or value < 1 for value in header["shape"])):
        raise ValueError(f"expected C-contiguous float32 [bins,height,width]: {path}")
    shape = header["shape"]
    if path.stat().st_size != offset + math.prod(shape) * 4:
        raise ValueError(f"NumPy array size differs from header: {path}")
    return shape, offset


def compare_arrays(cpu_path, gpu_path, atol, rtol):
    if sys.byteorder != "little":
        raise RuntimeError("float32 comparison requires a little-endian host")
    cpu_shape, cpu_offset = npy_header(cpu_path)
    gpu_shape, gpu_offset = npy_header(gpu_path)
    if cpu_shape != gpu_shape:
        raise ValueError(f"voxel shapes differ: {cpu_shape} versus {gpu_shape}")
    total = math.prod(cpu_shape)
    failed = 0
    max_error = 0.0
    sum_abs = 0.0
    sum_squares = 0.0
    cpu_sum = 0.0
    gpu_sum = 0.0
    worst_index = 0
    nonfinite = 0
    processed = 0
    with cpu_path.open("rb") as cpu, gpu_path.open("rb") as gpu:
        cpu.seek(cpu_offset)
        gpu.seek(gpu_offset)
        while processed < total:
            count = min(262144, total - processed)
            cpu_values = memoryview(cpu.read(count * 4)).cast("f")
            gpu_values = memoryview(gpu.read(count * 4)).cast("f")
            if len(cpu_values) != count or len(gpu_values) != count:
                raise ValueError("truncated voxel file")
            for index, (reference, actual) in enumerate(zip(cpu_values, gpu_values)):
                if not math.isfinite(reference) or not math.isfinite(actual):
                    nonfinite += 1
                    failed += 1
                    continue
                error = abs(float(reference) - float(actual))
                cpu_sum += reference
                gpu_sum += actual
                sum_abs += error
                sum_squares += error * error
                if error > max_error:
                    max_error = error
                    worst_index = processed + index
                if error > atol + rtol * max(1.0, abs(reference)):
                    failed += 1
            processed += count
    bins, height, width = cpu_shape
    plane = height * width
    return {
        "passed": failed == 0, "shape": list(cpu_shape), "voxel_count": total,
        "failed_voxels": failed, "nonfinite_voxels": nonfinite,
        "max_absolute_error": max_error,
        "mean_absolute_error": sum_abs / total,
        "rmse": math.sqrt(sum_squares / total),
        "cpu_sum": cpu_sum, "gpu_sum": gpu_sum,
        "worst_bin": worst_index // plane,
        "worst_y": (worst_index % plane) // width,
        "worst_x": worst_index % width,
        "atol": atol, "rtol": rtol,
    }


def analyze_run(run_dir, atol=1e-4, rtol=1e-5):
    if not math.isfinite(atol) or not math.isfinite(rtol) or atol < 0 or rtol < 0:
        raise ValueError("tolerances must be finite and nonnegative")
    root = run_dir.resolve(strict=True)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("schema_version") != 1 or manifest.get("status") != "complete":
        raise ValueError("captured run is incomplete")
    rows = []
    for case in manifest["cases"]:
        cpu_path, cpu_metrics = artifact(root, case["cpu"])
        for strategy, gpu_record in case["gpu"].items():
            gpu_path, gpu_metrics = artifact(root, gpu_record)
            if (cpu_metrics["backend"] != "cpu" or cpu_metrics["strategy"] != "reference" or
                    gpu_metrics["backend"] != "cuda" or gpu_metrics["strategy"] != strategy or
                    any(metrics["selected_events"] != case["event_count"] or
                        metrics["bins"] != case["bins"] for metrics in
                        (cpu_metrics, gpu_metrics)) or
                    any(cpu_metrics[key] != gpu_metrics[key] for key in
                        ("width", "height", "start_us", "end_us"))):
                raise ValueError(f"run metadata differs for {case['dataset']}/{strategy}")
            errors = compare_arrays(cpu_path, gpu_path, atol, rtol)
            rows.append({
                "dataset": case["dataset"], "dataset_sha256": case["dataset_sha256"],
                "event_count": case["event_count"], "bins": case["bins"],
                "strategy": strategy,
                "cpu_output_sha256": case["cpu"]["output_sha256"],
                "gpu_output_sha256": gpu_record["output_sha256"],
                "cpu_voxelize_ms": cpu_metrics["voxelize_ms"],
                "gpu_voxelize_ms": gpu_metrics["voxelize_ms"],
                "cpu_load_ms": cpu_metrics["load_ms"],
                "gpu_load_ms": gpu_metrics["load_ms"],
                "cpu_export_ms": cpu_metrics["export_ms"],
                "gpu_export_ms": gpu_metrics["export_ms"],
                "cpu_peak_process_rss_kib": cpu_metrics["peak_process_rss_kib"],
                "gpu_peak_process_rss_kib": gpu_metrics["peak_process_rss_kib"],
                "gpu_total_bytes": gpu_metrics["gpu_total_bytes"],
                "gpu_free_before_bytes": gpu_metrics["gpu_free_before_bytes"],
                "gpu_free_after_bytes": gpu_metrics["gpu_free_after_bytes"],
                **errors,
            })
    if not rows:
        raise ValueError("captured run has no GPU comparisons")
    return rows
