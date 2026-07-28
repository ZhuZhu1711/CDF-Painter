"""Tests for ECDF plot thinning and unique-value compression."""

from __future__ import annotations

import time
import unittest

import numpy as np

from cdf_app.ecdf import compute_ecdf, downsample_step_xy, proportions_at


class EcdfPlotDataTests(unittest.TestCase):
    def test_compute_ecdf_compresses_duplicates_for_plot(self):
        values = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 3.0])
        ecdf = compute_ecdf(values, name="t")
        self.assertIsNotNone(ecdf)
        assert ecdf is not None
        self.assertEqual(ecdf.n, 6)
        np.testing.assert_array_equal(ecdf.x, [1.0, 2.0, 3.0])
        np.testing.assert_allclose(ecdf.y, [3 / 6, 5 / 6, 1.0])
        self.assertEqual(len(ecdf.raw), 6)

    def test_cdf_at_uses_full_raw_sample(self):
        rng = np.random.default_rng(1)
        values = rng.normal(size=5000)
        ecdf = compute_ecdf(values)
        self.assertIsNotNone(ecdf)
        assert ecdf is not None
        x = float(np.median(values))
        expected = float(np.searchsorted(np.sort(values), x, side="right") / len(values))
        self.assertAlmostEqual(ecdf.cdf_at(x), expected)

    def test_downsample_step_xy_keeps_endpoints(self):
        x = np.arange(10000, dtype=float)
        y = (np.arange(1, 10001) / 10000).astype(float)
        xd, yd = downsample_step_xy(x, y, max_points=400)
        self.assertLessEqual(len(xd), 400 + 2)
        self.assertEqual(xd[0], 0.0)
        self.assertEqual(xd[-1], 9999.0)
        self.assertEqual(yd[0], y[0])
        self.assertEqual(yd[-1], y[-1])
        self.assertTrue(np.all(np.diff(xd) >= 0))

    def test_downsample_noop_when_small(self):
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([0.3, 0.6, 1.0])
        xd, yd = downsample_step_xy(x, y, max_points=4000)
        np.testing.assert_array_equal(xd, x)
        np.testing.assert_array_equal(yd, y)

    def test_proportions_at_large_group_is_fast(self):
        rng = np.random.default_rng(0)
        ecdf = compute_ecdf(rng.normal(size=200_000), name="big")
        self.assertIsNotNone(ecdf)
        assert ecdf is not None
        t0 = time.perf_counter()
        for i in range(500):
            proportions_at([ecdf], float(i) * 0.01)
        elapsed = time.perf_counter() - t0
        # Binary search should finish well under a second even on slow CI.
        self.assertLess(elapsed, 1.0)


if __name__ == "__main__":
    unittest.main()
