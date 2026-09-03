"""
Stage 2: Customer Baseline Engine — establishes each account's historical
purchasing pattern. Deterministic only; no LLM involvement.

"Recent" vs "baseline" split: the trailing RECENT_MONTHS calendar months
vs everything before that (shared with detect.py via changepoint.py).
Baseline typical-values use medians (robust to a single abnormally large
historical order — see data.md archetype 4); recent-window aggregates use
a sum/months rate rather than a median, since a median over only a few
noisy months doesn't average out sampling variance the way a rate over
more orders does.
"""

from __future__ import annotations

import pandas as pd

from .changepoint import (
    RECENT_MONTHS,
    full_month_index,
    normalize_medians_to_monthly_rate,
    rolling,
    scan_change_point,
)

HIGH_VALUE_CUMULATIVE_SHARE = 0.70


def _monthly_series(df: pd.DataFrame, group_cols: list[str] | None = None) -> pd.DataFrame:
    """Pivot to a complete (no gaps) month index, summed revenue, optionally
    grouped by additional columns (e.g. category)."""
    d = df.copy()
    d["month"] = d["date"].dt.to_period("M")
    group_cols = group_cols or []
    monthly = d.groupby(["month", *group_cols])["revenue"].sum().reset_index()
    return monthly


def _split_recent_baseline(months: pd.PeriodIndex) -> tuple[pd.PeriodIndex, pd.PeriodIndex]:
    if len(months) <= RECENT_MONTHS:
        return months, pd.PeriodIndex([], freq="M")
    return months[:-RECENT_MONTHS], months[-RECENT_MONTHS:]


def category_mix(df: pd.DataFrame) -> dict:
    """Baseline vs recent revenue share per category, and empirically-derived
    high/low-value category labels (never taken from generator ground truth)."""
    months = full_month_index(df)
    baseline_months, recent_months = _split_recent_baseline(months)

    monthly_cat = _monthly_series(df, ["category"])
    monthly_cat = monthly_cat.set_index(["month", "category"])["revenue"]

    def share_for(month_window: pd.PeriodIndex) -> dict:
        if len(month_window) == 0:
            return {}
        sub = monthly_cat[monthly_cat.index.get_level_values("month").isin(month_window)]
        by_cat = sub.groupby("category").sum()
        total = by_cat.sum()
        if total == 0:
            return {c: 0.0 for c in by_cat.index}
        return (by_cat / total).round(4).to_dict()

    baseline_share = share_for(baseline_months)
    recent_share = share_for(recent_months)

    ranked = sorted(baseline_share.items(), key=lambda kv: kv[1], reverse=True)
    high_value, cum = [], 0.0
    for cat, share in ranked:
        high_value.append(cat)
        cum += share
        if cum >= HIGH_VALUE_CUMULATIVE_SHARE:
            break
    low_value = [c for c, _ in ranked if c not in high_value]

    return {
        "baseline_share": baseline_share,
        "recent_share": recent_share,
        "high_value_categories": high_value,
        "low_value_categories": low_value,
        "baseline_months": len(baseline_months),
        "recent_months": len(recent_months),
    }


def overall_revenue(df: pd.DataFrame) -> dict:
    months = full_month_index(df)
    baseline_months, recent_months = _split_recent_baseline(months)
    monthly_total = _monthly_series(df).set_index("month")["revenue"].reindex(months, fill_value=0.0)

    baseline_vals = monthly_total.loc[baseline_months] if len(baseline_months) else pd.Series(dtype=float)
    recent_vals = monthly_total.loc[recent_months] if len(recent_months) else pd.Series(dtype=float)

    baseline_median = float(baseline_vals.median()) if len(baseline_vals) else None
    recent_rate = float(recent_vals.sum() / len(recent_vals)) if len(recent_vals) else None
    pct_change = None
    if baseline_median and baseline_median > 0 and recent_rate is not None:
        pct_change = round((recent_rate - baseline_median) / baseline_median, 4)

    change_point = scan_change_point(rolling(monthly_total))
    if change_point:
        change_point = normalize_medians_to_monthly_rate(change_point)

    # Coefficient of variation on the raw monthly series — lets Stage 4 tell
    # a naturally volatile account (large swings are normal for it) apart
    # from a genuine shift, rather than reading pct_change in isolation.
    cv = None
    if monthly_total.mean() > 0:
        cv = round(float(monthly_total.std() / monthly_total.mean()), 4)

    return {
        "baseline_monthly_median": baseline_median,
        "recent_monthly_rate": recent_rate,
        "pct_change": pct_change,
        "monthly_revenue_coefficient_of_variation": cv,
        "change_point": change_point,
        "monthly_series": {str(m): float(v) for m, v in monthly_total.items()},
    }


def order_behavior(df: pd.DataFrame) -> dict:
    months = full_month_index(df)
    baseline_months, recent_months = _split_recent_baseline(months)

    orders = df.drop_duplicates("order_id").copy()
    orders["month"] = orders["date"].dt.to_period("M")
    basket_width = df.groupby("order_id")["product_id"].nunique()
    order_revenue = df.groupby("order_id")["revenue"].sum()

    def freq_for(window: pd.PeriodIndex) -> float | None:
        if len(window) == 0:
            return None
        n_orders = orders[orders["month"].isin(window)].shape[0]
        return round(n_orders / len(window), 2)

    def stat_for(window: pd.PeriodIndex, series: pd.Series) -> float | None:
        if len(window) == 0:
            return None
        oids = orders.loc[orders["month"].isin(window), "order_id"]
        vals = series.loc[series.index.intersection(oids)]
        return float(vals.median()) if len(vals) else None

    def pct(a, b):
        if a is None or b is None or a == 0:
            return None
        return round((b - a) / a, 4)

    freq_baseline, freq_recent = freq_for(baseline_months), freq_for(recent_months)
    aov_baseline, aov_recent = stat_for(baseline_months, order_revenue), stat_for(recent_months, order_revenue)
    width_baseline, width_recent = stat_for(baseline_months, basket_width), stat_for(recent_months, basket_width)

    return {
        "order_frequency_baseline_per_month": freq_baseline,
        "order_frequency_recent_per_month": freq_recent,
        "order_frequency_pct_change": pct(freq_baseline, freq_recent),
        "aov_baseline": aov_baseline,
        "aov_recent": aov_recent,
        "aov_pct_change": pct(aov_baseline, aov_recent),
        "basket_width_baseline": width_baseline,
        "basket_width_recent": width_recent,
        "basket_width_pct_change": pct(width_baseline, width_recent),
    }


def build_baseline(df: pd.DataFrame, account_id: str) -> dict:
    acc_df = df[df["account_id"] == account_id]
    if acc_df.empty:
        raise ValueError(f"No rows for account_id={account_id!r}")
    return {
        "account_id": account_id,
        "overall_revenue": overall_revenue(acc_df),
        "category_mix": category_mix(acc_df),
        "order_behavior": order_behavior(acc_df),
    }
