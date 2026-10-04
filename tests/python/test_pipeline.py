"""Cross-language tests using a tiny, uncompressed HDF5 recording."""

import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"
CPU = Path(os.environ.get("EVENT_VOXEL_RUN_CPU", ROOT / "build/local-debug/event_voxel_run_cpu"))


class PipelineRoundTrip(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.recording = self.root / "tiny.hdf5"
        dtype = np.dtype([
            ("x", "<u2"), ("y", "<u2"), ("p", "<i2"), ("t", "<i8")
        ])
        events = np.array([
            (1, 0, 1, 0),
            (1, 0, 1, 25),
            (1, 0, 0, 50),
            (2, 1, 1, 99),
        ], dtype=dtype)
        with h5py.File(self.recording, "w") as file:
            file.attrs["geometry"] = "3x2"
            file.create_dataset("CD/events", data=events)
        self.reports = self.root / "reports"
        self.run_tool("inspect_dataset.py", "--input", self.recording,
                      "--output", self.reports, "--chunk-size", 2)
        self.report = self.reports / "tiny.metadata.json"

    def run_tool(self, name, *arguments, expect_success=True):
        command = [sys.executable, str(TOOLS / name), *map(str, arguments)]
        result = subprocess.run(command, capture_output=True, text=True)
        if expect_success and result.returncode:
            self.fail(f"{' '.join(command)}\n{result.stdout}\n{result.stderr}")
        return result

    def prepare(self, directory, *options):
        self.run_tool("prepare_dataset.py", "--input", self.recording,
                      "--report", self.report, "--output", directory,
                      "--chunk-size", 2, *options)
        return directory / "events.bin"

    def run_cpu(self, binary, output):
        result = subprocess.run([str(CPU), str(binary), "3", str(output)],
                                capture_output=True, text=True)
        if result.returncode:
            self.fail(result.stdout + result.stderr)

    def test_full_recording_to_npy(self):
        prepared = self.root / "prepared"
        binary = self.prepare(prepared)
        metadata = json.loads((prepared / "metadata.json").read_text())
        self.assertEqual(metadata["event_count"], 4)
        self.assertEqual((metadata["start_us"], metadata["end_us"]), (0, 100))
        self.assertEqual(binary.stat().st_size, 40 + 4 * 16)

        output = self.root / "voxels.npy"
        self.run_cpu(binary, output)
        voxels = np.load(output, allow_pickle=False)
        self.assertEqual(voxels.shape, (3, 2, 3))
        self.assertEqual(voxels.dtype, np.dtype("float32"))
        expected = np.zeros((3, 2, 3), dtype=np.float32)
        expected[:, 0, 1] = [1.5, -0.5, 0.0]
        expected[:, 1, 2] = [0.0, 0.02, 0.98]
        np.testing.assert_allclose(voxels, expected, atol=1e-6, rtol=0)

    def test_bounded_event_selection(self):
        prepared = self.root / "subset"
        binary = self.prepare(prepared, "--start-us", 25, "--max-events", 2)
        with binary.open("rb") as stream:
            magic, width, height, count, start, end = struct.unpack(
                "<8sIIQqq", stream.read(40))
        self.assertEqual((magic, width, height), (b"EVXEVT01", 3, 2))
        self.assertEqual((count, start, end), (2, 25, 51))
        output = self.root / "subset.npy"
        self.run_cpu(binary, output)
        voxels = np.load(output, allow_pickle=False)
        self.assertAlmostEqual(float(voxels[:, 0, 1].sum()), 0.0, places=5)

    def test_explicit_end_is_exclusive(self):
        prepared = self.root / "interval"
        binary = self.prepare(prepared, "--start-us", 25, "--end-us", 50)
        metadata = json.loads((prepared / "metadata.json").read_text())
        self.assertEqual(metadata["event_count"], 1)
        self.assertEqual((metadata["start_us"], metadata["end_us"]), (25, 50))
        output = self.root / "interval.npy"
        self.run_cpu(binary, output)
        voxels = np.load(output, allow_pickle=False)
        self.assertEqual(float(voxels[0, 0, 1]), 1.0)
        self.assertEqual(float(voxels[1:, 0, 1].sum()), 0.0)

    def test_event_index_selection_uses_selected_time_origin(self):
        prepared = self.root / "tail"
        binary = self.prepare(prepared, "--start-index", 2, "--max-events", 2)
        metadata = json.loads((prepared / "metadata.json").read_text())
        self.assertEqual(metadata["selection"]["start_index"], 2)
        self.assertEqual(metadata["event_count"], 2)
        self.assertEqual((metadata["start_us"], metadata["end_us"]), (50, 100))
        output = self.root / "tail.npy"
        self.run_cpu(binary, output)
        voxels = np.load(output, allow_pickle=False)
        self.assertEqual(float(voxels[0, 0, 1]), -1.0)

    def test_out_of_range_event_index_is_rejected(self):
        output = self.root / "past_end"
        result = self.run_tool(
            "prepare_dataset.py", "--input", self.recording,
            "--report", self.report, "--output", output,
            "--start-index", 4, expect_success=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((output / "events.bin").exists())

    def test_empty_window_is_rejected(self):
        output = self.root / "empty"
        result = self.run_tool(
            "prepare_dataset.py", "--input", self.recording,
            "--report", self.report, "--output", output,
            "--start-us", 80, "--end-us", 90, expect_success=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((output / "events.bin").exists())


if __name__ == "__main__":
    unittest.main()
