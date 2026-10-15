"""Check archived voxel comparison, including integrity and mismatch reporting."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/experiments"))
from output_comparison import analyze_run  # noqa: E402
from run_matrix import sha256_file  # noqa: E402


class OutputComparisonTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        cpu = np.zeros((2, 2, 2), dtype="<f4")
        cpu[0, 0, 0] = 1.0
        np.save(self.root / "cpu.npy", cpu)
        np.save(self.root / "gpu.npy", cpu)
        metrics = {
            "selected_events": 2, "bins": 2, "width": 2, "height": 2,
            "start_us": 0, "end_us": 10, "voxelize_ms": 1.0,
            "load_ms": 0.5, "export_ms": 0.2, "peak_process_rss_kib": 100,
            "gpu_total_bytes": None, "gpu_free_before_bytes": None,
            "gpu_free_after_bytes": None,
        }
        (self.root / "cpu.json").write_text(json.dumps({
            **metrics, "backend": "cpu", "strategy": "reference"}))
        (self.root / "gpu.json").write_text(json.dumps({
            **metrics, "backend": "cuda", "strategy": "event-scatter"}))
        self.write_manifest()

    def record(self, stem):
        return {
            "output": f"{stem}.npy",
            "output_sha256": sha256_file(self.root / f"{stem}.npy"),
            "metrics": f"{stem}.json",
            "metrics_sha256": sha256_file(self.root / f"{stem}.json"),
        }

    def write_manifest(self):
        manifest = {"schema_version": 1, "status": "complete", "cases": [{
            "dataset": "tiny", "dataset_sha256": "abc", "event_count": 2,
            "bins": 2, "cpu": self.record("cpu"),
            "gpu": {"event-scatter": self.record("gpu")},
        }]}
        (self.root / "manifest.json").write_text(json.dumps(manifest))

    def test_identical_outputs_pass(self):
        row = analyze_run(self.root)[0]
        self.assertTrue(row["passed"])
        self.assertEqual(row["failed_voxels"], 0)
        self.assertEqual(row["voxel_count"], 8)

    def test_mismatch_is_reported_from_saved_data(self):
        gpu = np.load(self.root / "gpu.npy")
        gpu[1, 1, 0] = 2.0
        np.save(self.root / "gpu.npy", gpu)
        self.write_manifest()
        row = analyze_run(self.root)[0]
        self.assertFalse(row["passed"])
        self.assertEqual(row["failed_voxels"], 1)
        self.assertEqual((row["worst_bin"], row["worst_y"], row["worst_x"]),
                         (1, 1, 0))

    def test_modified_array_is_rejected(self):
        with (self.root / "gpu.npy").open("ab") as output:
            output.write(b"x")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            analyze_run(self.root)


if __name__ == "__main__":
    unittest.main()
