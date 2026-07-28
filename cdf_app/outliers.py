"""Outlier / anomaly detection and removal for numeric columns."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


class OutlierRule(str, Enum):
    """User-selectable outlier rejection rules."""

    IQR = "iqr"
    ZSCORE = "zscore"
    PERCENTILE = "percentile"


RULE_LABELS = {
    OutlierRule.IQR: "IQR 法（超出 Q1−1.5·IQR ~ Q3+1.5·IQR）",
    OutlierRule.ZSCORE: "Z-Score 法（|z| > 3）",
    OutlierRule.PERCENTILE: "分位数法（剔除 <1% 或 >99%）",
}


@dataclass
class OutlierResult:
    """Result of applying an outlier rule to a DataFrame."""

    cleaned: pd.DataFrame
    mask_keep: pd.Series
    n_removed: int
    n_total: int
    rule: OutlierRule
    columns: List[str]

    @property
    def n_kept(self) -> int:
        return int(self.mask_keep.sum())


def _numeric_series(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce")


def outlier_mask_iqr(series: pd.Series, *, k: float = 1.5) -> pd.Series:
    """True where value is an IQR outlier (NaN is not treated as outlier)."""
    s = pd.to_numeric(series, errors="coerce")
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    if pd.isna(iqr) or iqr == 0:
        return pd.Series(False, index=series.index)
    lo = q1 - k * iqr
    hi = q3 + k * iqr
    return s.notna() & ((s < lo) | (s > hi))


def outlier_mask_zscore(series: pd.Series, *, threshold: float = 3.0) -> pd.Series:
    """True where |z| > threshold (NaN is not treated as outlier)."""
    s = pd.to_numeric(series, errors="coerce")
    mean = s.mean()
    std = s.std(ddof=1)
    if pd.isna(std) or std == 0 or pd.isna(mean):
        return pd.Series(False, index=series.index)
    z = (s - mean) / std
    return s.notna() & (z.abs() > threshold)


def outlier_mask_percentile(
    series: pd.Series,
    *,
    low: float = 0.01,
    high: float = 0.99,
) -> pd.Series:
    """True where value is outside [low, high] empirical percentiles."""
    s = pd.to_numeric(series, errors="coerce")
    if s.notna().sum() < 2:
        return pd.Series(False, index=series.index)
    lo = s.quantile(low)
    hi = s.quantile(high)
    if pd.isna(lo) or pd.isna(hi):
        return pd.Series(False, index=series.index)
    return s.notna() & ((s < lo) | (s > hi))


def compute_outlier_mask(
    df: pd.DataFrame,
    columns: Sequence[str],
    rule: OutlierRule,
) -> pd.Series:
    """Union of outlier flags across columns (a row is outlier if any col flags it)."""
    if df.empty or not columns:
        return pd.Series(False, index=df.index)

    mask = pd.Series(False, index=df.index)
    for col in columns:
        if col not in df.columns:
            continue
        series = _numeric_series(df, col)
        if rule == OutlierRule.IQR:
            col_mask = outlier_mask_iqr(series)
        elif rule == OutlierRule.ZSCORE:
            col_mask = outlier_mask_zscore(series)
        elif rule == OutlierRule.PERCENTILE:
            col_mask = outlier_mask_percentile(series)
        else:
            raise ValueError(f"Unknown outlier rule: {rule}")
        mask = mask | col_mask
    return mask


def remove_outliers(
    df: pd.DataFrame,
    columns: Sequence[str],
    rule: OutlierRule,
) -> OutlierResult:
    """Return a cleaned DataFrame with outlier rows removed."""
    mask_outlier = compute_outlier_mask(df, columns, rule)
    mask_keep = ~mask_outlier
    cleaned = df.loc[mask_keep].copy()
    return OutlierResult(
        cleaned=cleaned,
        mask_keep=mask_keep,
        n_removed=int(mask_outlier.sum()),
        n_total=len(df),
        rule=rule,
        columns=list(columns),
    )


def preview_outlier_counts(
    df: pd.DataFrame,
    columns: Sequence[str],
    rule: OutlierRule,
) -> Tuple[int, int]:
    """Return (n_removed, n_total) without copying the frame."""
    mask = compute_outlier_mask(df, columns, rule)
    return int(mask.sum()), len(df)


def parse_rule(value: Optional[str]) -> OutlierRule:
    if isinstance(value, OutlierRule):
        return value
    return OutlierRule(str(value))
