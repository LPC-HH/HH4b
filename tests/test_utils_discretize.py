"""Tests for discretize_var in utils.py."""

from __future__ import annotations

import numpy as np

from HH4b.utils import discretize_var


class TestDiscretizeVar:
    def test_default_bins(self):
        """Values map to correct bin indices with default bins."""
        # default bins: [0, 0.8, 0.9, 0.94, 0.97, 0.99, 1]
        vals = np.array([0.5, 0.85, 0.92, 0.95, 0.98, 0.995])
        result = discretize_var(vals)
        # 0.5 -> bin 1, 0.85 -> bin 2, 0.92 -> bin 3,
        # 0.95 -> bin 4, 0.98 -> bin 5, 0.995 -> bin 6
        np.testing.assert_array_equal(result, [1, 2, 3, 4, 5, 6])

    def test_custom_bins(self):
        vals = np.array([0.1, 0.6, 0.9])
        result = discretize_var(vals, bins=[0, 0.5, 1.0])
        # 0.1 -> bin 1 (0-0.5), 0.6 -> bin 2 (0.5-1.0), 0.9 -> bin 2
        np.testing.assert_array_equal(result, [1, 2, 2])

    def test_clipping_below(self):
        """Values below all bins are clipped to bin 1."""
        vals = np.array([-1.0, 0.0])
        result = discretize_var(vals, bins=[0.5, 1.0])
        np.testing.assert_array_equal(result, [1, 1])

    def test_clipping_above(self):
        """Values above all bins are clipped to the last bin."""
        vals = np.array([2.0, 1.5])
        result = discretize_var(vals, bins=[0, 0.5, 1.0])
        np.testing.assert_array_equal(result, [2, 2])

    def test_returns_integer_indices(self):
        result = discretize_var(np.array([0.5]))
        assert result.dtype in (np.int32, np.int64, np.intp)
