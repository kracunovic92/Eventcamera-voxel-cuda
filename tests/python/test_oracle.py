"""Compare the CPU output with the independent triangular-weight definition."""

import os
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
CPU = Path(os.environ.get("EVENT_VOXEL_RUN_CPU", ROOT / "build/local-debug/event_voxel_run_cpu"))


def oracle(events, bins, width, height, start, end):
    output = np.zeros((bins, height, width), dtype=np.float64)
    for x, y, polarity, timestamp in events:
        position = (bins - 1) * (int(timestamp) - start) / (end - start)
        sign = 1 if polarity else -1
        for bin_index in range(bins):
            output[bin_index, y, x] += sign * max(0.0, 1.0 - abs(bin_index - position))
    return output


class IndependentOracle(unittest.TestCase):
    def test_distributions_and_bin_counts(self):
        rng = np.random.default_rng(4157)
        width, height, start, end = 8, 6, 1000, 2000
        count = 500
        for distribution in ("uniform", "hotspot", "single_pixel"):
            with self.subTest(distribution=distribution):
                if distribution == "uniform":
                    x = rng.integers(0, width, count)
                    y = rng.integers(0, height, count)
                elif distribution == "hotspot":
                    x = rng.integers(2, 4, count)
                    y = rng.integers(2, 4, count)
                else:
                    x = np.full(count, 2)
                    y = np.full(count, 3)
                polarity = rng.integers(0, 2, count)
                timestamps = np.sort(rng.integers(start, end, count))
                events = list(zip(x, y, polarity, timestamps))
                with tempfile.TemporaryDirectory() as directory:
                    binary = Path(directory) / "events.bin"
                    with binary.open("wb") as stream:
                        stream.write(struct.pack(
                            "<8sIIQqq", b"EVXEVT01", width, height,
                            count, start, end))
                        for event_x, event_y, event_p, event_t in events:
                            stream.write(struct.pack(
                                "<HHh2xq", int(event_x), int(event_y),
                                int(event_p), int(event_t)))
                    for bins in (2, 5, 10):
                        output = Path(directory) / f"bins_{bins}.npy"
                        result = subprocess.run(
                            [str(CPU), str(binary), str(bins), str(output)],
                            capture_output=True, text=True)
                        self.assertEqual(result.returncode, 0,
                                         result.stdout + result.stderr)
                        actual = np.load(output, allow_pickle=False)
                        expected = oracle(events, bins, width, height, start, end)
                        np.testing.assert_allclose(actual, expected, atol=2e-4, rtol=0)


if __name__ == "__main__":
    unittest.main()
