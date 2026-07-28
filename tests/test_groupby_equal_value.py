"""Tests for equal-value grouping column semantics."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from cdf_app.ecdf import (
    compute_multi_value_ecdfs,
    group_value_label,
    list_group_levels,
)


class GroupByEqualValueTests(unittest.TestCase):
    def setUp(self) -> None:
        self.df = pd.DataFrame(
            {
                "value": [1.0, 2.0, 10.0, 11.0, 12.0, np.nan],
                "line": ["A", "A", "B", "B", "B", "A"],
                "shift": ["Day", "Night", "Day", "Day", "Night", None],
            }
        )

    def test_list_group_levels_splits_by_equal_values(self):
        levels = list_group_levels(self.df, "line")
        labels = [label for _value, label, _count in levels]
        counts = {label: count for _value, label, count in levels}
        self.assertEqual(labels, ["A", "B"])
        self.assertEqual(counts["A"], 3)
        self.assertEqual(counts["B"], 3)

    def test_list_group_levels_includes_missing(self):
        levels = list_group_levels(self.df, "shift")
        labels = [label for _value, label, _count in levels]
        self.assertIn("缺失", labels)
        self.assertEqual(group_value_label(None), "缺失")

    def test_compute_groups_one_series_per_distinct_value(self):
        groups = compute_multi_value_ecdfs(self.df, ["value"], "line")
        names = [g.name for g in groups]
        self.assertEqual(names, ["A", "B"])
        # A has values 1, 2 (nan dropped in ecdf) → n=2
        by_name = {g.name: g for g in groups}
        self.assertEqual(by_name["A"].n, 2)
        self.assertEqual(by_name["B"].n, 3)

    def test_filter_subset_of_group_values(self):
        groups = compute_multi_value_ecdfs(
            self.df, ["value"], "line", group_values=["B"]
        )
        self.assertEqual([g.name for g in groups], ["B"])
        self.assertEqual(groups[0].n, 3)

    def test_empty_group_values_yields_nothing(self):
        groups = compute_multi_value_ecdfs(
            self.df, ["value"], "line", group_values=[]
        )
        self.assertEqual(groups, [])

    def test_rows_with_same_value_share_one_ecdf(self):
        """Canonical requirement: same grouping-column value → one group."""
        df = pd.DataFrame(
            {
                "x": [1.0, 2.0, 3.0, 4.0],
                "g": ["same", "same", "same", "other"],
            }
        )
        groups = compute_multi_value_ecdfs(df, ["x"], "g")
        by_name = {g.name: g for g in groups}
        self.assertEqual(by_name["same"].n, 3)
        self.assertEqual(by_name["other"].n, 1)
        np.testing.assert_allclose(sorted(by_name["same"].raw), [1.0, 2.0, 3.0])


if __name__ == "__main__":
    unittest.main()
