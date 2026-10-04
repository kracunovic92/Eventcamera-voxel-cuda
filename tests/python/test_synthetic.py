"""Synthetic workload determinism and CPU acceptance."""

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
GENERATOR = ROOT / "tools/generate_synthetic.py"
CPU = Path(os.environ.get("EVENT_VOXEL_RUN_CPU", ROOT / "build/local-debug/event_voxel_run_cpu"))
PRECISION = ROOT / "tools/verify_precision.py"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class SyntheticWorkloads(unittest.TestCase):
    def test_reproducible_distributions(self):
        for spatial, temporal in (("uniform", "uniform"),
                                  ("hotspot", "burst"),
                                  ("single_pixel", "uniform")):
            with self.subTest(spatial=spatial, temporal=temporal):
                with tempfile.TemporaryDirectory() as temporary:
                    first = Path(temporary) / "first"
                    second = Path(temporary) / "second"
                    for output in (first, second):
                        subprocess.run([
                            sys.executable, str(GENERATOR), "--output", str(output),
                            "--events", "200", "--width", "8", "--height", "6",
                            "--seed", "123", "--spatial", spatial,
                            "--temporal", temporal,
                        ], check=True, capture_output=True, text=True)
                    binary = first / "events.bin"
                    self.assertEqual(digest(binary), digest(second / "events.bin"))
                    metadata = json.loads((first / "metadata.json").read_text())
                    self.assertEqual(metadata["binary_sha256"], digest(binary))
                    output = Path(temporary) / "grid.npy"
                    subprocess.run([str(CPU), str(binary), "5", str(output)],
                                   check=True, capture_output=True, text=True)
                    voxels = np.load(output, allow_pickle=False)
                    self.assertEqual(voxels.shape, (5, 6, 8))
                    self.assertTrue(np.isfinite(voxels).all())
                    report = Path(temporary) / "precision.json"
                    subprocess.run([
                        sys.executable, str(PRECISION), "--events", str(binary),
                        "--voxels", str(output), "--report", str(report),
                        "--cpu-executable", str(CPU),
                    ], check=True, capture_output=True, text=True)
                    self.assertTrue(json.loads(report.read_text())["result"]["passed"])

                    if spatial == "single_pixel":
                        voxels[0, 0, 0] += 1.0
                        np.save(output, voxels)
                        result = subprocess.run([
                            sys.executable, str(PRECISION), "--events", str(binary),
                            "--voxels", str(output), "--report", str(report),
                        ], capture_output=True, text=True)
                        self.assertEqual(result.returncode, 1)
                        self.assertGreater(json.loads(report.read_text())["result"]["failed_voxels"], 0)


if __name__ == "__main__":
    unittest.main()
