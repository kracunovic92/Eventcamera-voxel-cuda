"""Check that external-reference sampling matches the C++ selection rule."""

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/experiments"))
from validate_outputs import sample_evenly  # noqa: E402


class ExternalSamplingTest(unittest.TestCase):
    def test_endpoints_and_integer_spaced_indices(self):
        events = np.arange(10)
        np.testing.assert_array_equal(sample_evenly(events, 4), [0, 3, 6, 9])
        np.testing.assert_array_equal(sample_evenly(events, 1), [0])
        np.testing.assert_array_equal(sample_evenly(events, 10), events)

    def test_invalid_counts_are_rejected(self):
        with self.assertRaises(ValueError):
            sample_evenly(np.arange(3), 0)
        with self.assertRaises(ValueError):
            sample_evenly(np.arange(3), 4)


if __name__ == "__main__":
    unittest.main()
