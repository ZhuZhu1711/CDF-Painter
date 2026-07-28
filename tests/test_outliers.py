"""Unit tests for outlier rejection and lazy preview helpers."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from cdf_app.outliers import (
    OutlierRule,
    compute_outlier_mask,
    outlier_mask_iqr,
    outlier_mask_percentile,
    outlier_mask_zscore,
    remove_outliers,
)


class OutlierTests(unittest.TestCase):
    def test_iqr_flags_obvious_outliers(self):
        s = pd.Series([10, 11, 9, 10.5, 10.2, 100.0, -50.0])
        mask = outlier_mask_iqr(s)
        self.assertTrue(bool(mask.iloc[5]))
        self.assertTrue(bool(mask.iloc[6]))
        self.assertFalse(bool(mask.iloc[0]))

    def test_zscore_flags_extreme(self):
        rng = np.random.default_rng(0)
        vals = rng.normal(0, 1, 200).tolist()
        vals.append(50.0)
        s = pd.Series(vals)
        mask = outlier_mask_zscore(s, threshold=3.0)
        self.assertTrue(bool(mask.iloc[-1]))
        self.assertLess(int(mask.sum()), 10)

    def test_percentile_trims_tails(self):
        s = pd.Series(np.arange(100, dtype=float))
        mask = outlier_mask_percentile(s, low=0.01, high=0.99)
        # With 100 points, 1% / 99% cut roughly the extreme ends
        self.assertGreaterEqual(int(mask.sum()), 1)

    def test_remove_outliers_union_across_columns(self):
        df = pd.DataFrame(
            {
                "a": [1.0, 2.0, 3.0, 4.0, 100.0],
                "b": [1.0, 2.0, 3.0, 4.0, 5.0],
                "g": ["x", "x", "y", "y", "z"],
            }
        )
        result = remove_outliers(df, ["a"], OutlierRule.IQR)
        self.assertEqual(result.n_removed, 1)
        self.assertEqual(result.n_kept, 4)
        self.assertNotIn(100.0, set(result.cleaned["a"]))

    def test_compute_mask_empty_columns(self):
        df = pd.DataFrame({"a": [1.0, 2.0]})
        mask = compute_outlier_mask(df, [], OutlierRule.ZSCORE)
        self.assertFalse(bool(mask.any()))


if __name__ == "__main__":
    unittest.main()
