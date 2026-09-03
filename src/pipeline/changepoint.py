"""
Shared change-point scanning logic used by both the headline total-revenue
trend (baseline.py) and per-category detection (detect.py), so both layers
handle the same noise problem the same way.

MIN_SEGMENT_MONTHS is deliberately not tiny: a short "before" segment lets
a single abnormal month (e.g. a one-off bulk order) define the entire
baseline for that segment, manufacturing a false change-point right after
it. 5 months requires several independent months of evidence on each side.
"""

from __future__ import annotations

import pandas as pd

ROLLING_WINDOW = 3
MIN_SEGMENT_MONTHS = 5
DECLINE_THRESHOLD = 0.35
MIN_BASELINE_MONTHLY_REVENUE = 50.0
RECOVERY_FRACTION = 0.75

# Trailing window treated as "current" behaviour for recent-vs-baseline
# comparisons. 3 months proved too few orders (at ~2-3 orders/month) to
# average out noise — widened after verification showed a "healthy"
# archetype account swinging -59% purely from sampling variance.
RECENT_MONTHS = 6


def full_month_index(df: pd.DataFrame) -> pd.PeriodIndex:
    start, end = df["date"].min().to_period("M"), df["date"].max().to_period("M")
    return pd.period_range(start, end, freq="M")


WINSOR_CAP_MULTIPLE = 1.5


def winsorize(series: pd.Series, cap_multiple: float = WINSOR_CAP_MULTIPLE) -> pd.Series:
    """Cap each value at cap_multiple x the series' own median, so a single
    one-off spike (e.g. a bulk order) can't get smeared across several
    adjacent points by the rolling sum below and dominate a segment median
    that would otherwise correctly discount it as an outlier."""
    positive = series[series > 0]
    if positive.empty:
        return series
    cap = positive.median() * cap_multiple
    return series.clip(upper=cap)


def rolling(series: pd.Series, window: int = ROLLING_WINDOW) -> pd.Series:
    return winsorize(series).rolling(window=window, min_periods=1).sum()


def scan_change_point(
    smoothed: pd.Series,
    min_segment_months: int = MIN_SEGMENT_MONTHS,
    decline_threshold: float = DECLINE_THRESHOLD,
    min_baseline_revenue: float = MIN_BASELINE_MONTHLY_REVENUE,
) -> dict | None:
    """Find the split point maximizing the decline between a 'before' and
    'after' segment median. Returns None if no candidate split is viable
    (too little history) or nothing crosses the decline threshold."""
    n = len(smoothed)
    if n < 2 * min_segment_months:
        return None

    best = None
    for t in range(min_segment_months, n - min_segment_months + 1):
        before = smoothed.iloc[:t]
        after = smoothed.iloc[t:]
        before_med, after_med = before.median(), after.median()
        if before_med < min_baseline_revenue:
            continue
        decline = (before_med - after_med) / before_med
        if best is None or decline > best["decline"]:
            best = {
                "t": t,
                "decline": decline,
                "before_med": float(before_med),
                "after_med": float(after_med),
                "change_point_month": str(smoothed.index[t]),
            }

    if best is None or best["decline"] < decline_threshold:
        return None

    tail = smoothed.iloc[best["t"]:]
    # The trailing rolling sum blends pre-change months into the first
    # (ROLLING_WINDOW - 1) points right after the split, inflating them
    # regardless of what actually happened post-change — skip those points
    # when checking for recovery, or a clean permanent drop to zero can
    # still read as "recovered" purely from that boundary artifact.
    recovery_tail = tail.iloc[ROLLING_WINDOW - 1:] if len(tail) > ROLLING_WINDOW - 1 else tail
    recovered = bool(recovery_tail.max() >= best["before_med"] * RECOVERY_FRACTION)
    sustained_months = int((tail <= best["before_med"] * (1 - decline_threshold / 2)).sum())

    return {
        "change_point_month": best["change_point_month"],
        "before_monthly_median": round(best["before_med"], 2),
        "after_monthly_median": round(best["after_med"], 2),
        "pct_decline": round(best["decline"], 4),
        "recovered": recovered,
        "sustained_months_since_change_point": sustained_months,
    }


def normalize_medians_to_monthly_rate(cp: dict, window: int = ROLLING_WINDOW) -> dict:
    """scan_change_point's before/after medians are on whatever scale the
    input series was — when called on a rolling window-month SUM (the
    normal case, for noise smoothing), they're a `window`-month total, not
    a monthly rate, even though the field names say "monthly". Callers
    that pass a rolled series must call this before the numbers reach
    anything ₹-denominated downstream (Stage 5 impact) or every figure is
    inflated by `window`x. pct_decline is a ratio of two same-scale values
    and needs no adjustment."""
    normalized = dict(cp)
    normalized["before_monthly_median"] = round(cp["before_monthly_median"] / window, 2)
    normalized["after_monthly_median"] = round(cp["after_monthly_median"] / window, 2)
    return normalized
