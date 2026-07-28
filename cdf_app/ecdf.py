"""Empirical CDF utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass
class GroupECDF:
    """ECDF result for one group."""

    name: str
    x: np.ndarray  # sorted unique-step x (sorted observations)
    y: np.ndarray  # cumulative proportion i/n
    n: int
    mean: float
    std: float
    raw: np.ndarray

    def cdf_at(self, value: float) -> float:
        """Return empirical CDF F(value) = P(X <= value)."""
        if self.n == 0:
            return float("nan")
        return float(np.searchsorted(self.raw, value, side="right") / self.n)


def compute_ecdf(values: np.ndarray, name: str = "全部") -> Optional[GroupECDF]:
    """Compute step-function ECDF for a 1-D numeric array."""
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return None

    sorted_vals = np.sort(arr)
    n = int(sorted_vals.size)
    y = np.arange(1, n + 1, dtype=float) / n

    return GroupECDF(
        name=str(name),
        x=sorted_vals,
        y=y,
        n=n,
        mean=float(np.mean(sorted_vals)),
        std=float(np.std(sorted_vals, ddof=1)) if n > 1 else 0.0,
        raw=sorted_vals,
    )


def group_value_label(value) -> str:
    """Display label for one distinct value in a grouping column."""
    if pd.isna(value):
        return "缺失"
    return str(value)


def list_group_levels(df: pd.DataFrame, group_col: str) -> List[Tuple[object, str, int]]:
    """List distinct values in ``group_col`` (equal values → one group).

    Returns a list of ``(raw_value, label, row_count)`` sorted by label
    (missing values last). This is the definition of「分组列」: rows that
    share the same value in this column belong to the same group.
    """
    if group_col not in df.columns:
        raise KeyError(f"Column not found: {group_col}")

    levels: List[Tuple[object, str, int]] = []
    for value, subset in df.groupby(group_col, dropna=False, sort=True):
        levels.append((value, group_value_label(value), int(len(subset))))

    # Stable, human-friendly order: non-missing labels sorted, then 缺失.
    levels.sort(key=lambda item: (item[1] == "缺失", item[1]))
    return levels


def compute_grouped_ecdfs(
    df: pd.DataFrame,
    value_col: str,
    group_col: Optional[str] = None,
    group_values: Optional[Sequence[object]] = None,
) -> List[GroupECDF]:
    """Compute ECDF for all data or each equal-value group (single value column)."""
    return compute_multi_value_ecdfs(df, [value_col], group_col, group_values=group_values)


def compute_multi_value_ecdfs(
    df: pd.DataFrame,
    value_cols: List[str],
    group_col: Optional[str] = None,
    group_values: Optional[Sequence[object]] = None,
) -> List[GroupECDF]:
    """Compute ECDFs for one or more value columns, optionally split by group.

    When ``group_col`` is set, rows are partitioned by **equal values** in
    that column (pandas ``groupby``): each distinct value is one group, and
    an ECDF is computed from the numeric values within that group only.

    ``group_values`` optionally restricts which distinct values are plotted
    (same equality rule). ``None`` means all distinct values.

    Series naming:
    - 1 value col, no group → "全部"
    - 1 value col, with group → group level name
    - N value cols, no group → column name
    - N value cols, with group → "{col} / {group}"
    """
    if not value_cols:
        return []

    missing = [c for c in value_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Column not found: {missing[0]}")

    multi_values = len(value_cols) > 1
    has_group = bool(group_col) and group_col in df.columns
    results: List[GroupECDF] = []

    allowed = None
    if has_group and group_values is not None:
        allowed = list(group_values)
        if not allowed:
            return []

    for value_col in value_cols:
        if not has_group:
            if multi_values:
                name = str(value_col)
            else:
                name = "全部"
            ecdf = compute_ecdf(df[value_col].to_numpy(), name=name)
            if ecdf is not None:
                results.append(ecdf)
            continue

        for group_name, subset in df.groupby(group_col, dropna=False, sort=True):
            if allowed is not None and not _value_in_group_selection(group_name, allowed):
                continue
            g_label = group_value_label(group_name)
            if multi_values:
                label = f"{value_col} / {g_label}"
            else:
                label = g_label
            ecdf = compute_ecdf(subset[value_col].to_numpy(), name=label)
            if ecdf is not None:
                results.append(ecdf)

    return results


def _value_in_group_selection(value, allowed: Sequence[object]) -> bool:
    """Membership test that treats NaN / NA as equal to themselves."""
    value_is_na = pd.isna(value)
    for item in allowed:
        item_is_na = pd.isna(item)
        if value_is_na and item_is_na:
            return True
        if not value_is_na and not item_is_na and item == value:
            return True
    return False


def data_x_range(groups: List[GroupECDF]) -> Tuple[float, float]:
    """Overall min/max of raw data across groups."""
    if not groups:
        return 0.0, 1.0
    lows = [float(g.raw.min()) for g in groups]
    highs = [float(g.raw.max()) for g in groups]
    lo, hi = min(lows), max(highs)
    if lo == hi:
        pad = abs(lo) * 0.05 if lo != 0 else 0.5
        return lo - pad, hi + pad
    return lo, hi


def stats_legend_text(group: GroupECDF) -> str:
    """Legend label with N / Mean / StDev (Minitab-style)."""
    return (
        f"{group.name}\n"
        f"  N = {group.n}\n"
        f"  MEAN = {group.mean:.4g}\n"
        f"  S.D. = {group.std:.4g}"
    )


def proportions_at(groups: List[GroupECDF], x: float) -> Dict[str, float]:
    """CDF proportion for each group at x."""
    return {g.name: g.cdf_at(x) for g in groups}
