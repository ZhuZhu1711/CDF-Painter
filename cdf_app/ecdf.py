"""Empirical CDF utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

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


def compute_grouped_ecdfs(
    df: pd.DataFrame,
    value_col: str,
    group_col: Optional[str] = None,
) -> List[GroupECDF]:
    """Compute ECDF for all data or each group."""
    if value_col not in df.columns:
        raise KeyError(f"Column not found: {value_col}")

    results: List[GroupECDF] = []
    if not group_col or group_col not in df.columns:
        ecdf = compute_ecdf(df[value_col].to_numpy(), name="全部")
        if ecdf is not None:
            results.append(ecdf)
        return results

    for group_name, subset in df.groupby(group_col, dropna=False, sort=True):
        label = "缺失" if pd.isna(group_name) else str(group_name)
        ecdf = compute_ecdf(subset[value_col].to_numpy(), name=label)
        if ecdf is not None:
            results.append(ecdf)
    return results


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
