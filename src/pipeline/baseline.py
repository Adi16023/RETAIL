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

Revenue is not the only baseline. Two of the dataset's leak archetypes
(hidden mix collapse, discount creep) hold revenue flat and bleed value
through margin instead, so this stage profiles four dimensions in parallel
where the columns allow it — revenue, margin, discount, and value-tier mix.
Each profile degrades to None rather than guessing when its source column
is absent; see ingest.analysis_dimensions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .changepoint import (
    RECENT_MONTHS,
    full_month_index,
    normalize_medians_to_monthly_rate,
    rolling,
    scan_change_point,
)
from .ingest import TIER_ORDER

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


def _sales_lines(df: pd.DataFrame) -> pd.DataFrame:
    """Rows excluding return/credit lines.

    Returns still NET into revenue and margin totals — that is what a credit
    note means financially. They are excluded only from *behavioural*
    statistics (order frequency, basket width, AOV), where a standalone
    credit note is not an order and counting it as one would fake a
    frequency rise and a basket-width collapse in the same breath.
    """
    if "is_return" not in df.columns:
        return df
    return df[df["is_return"] != 1]


def margin_profile(df: pd.DataFrame) -> dict | None:
    """Margin rate and absolute margin, baseline vs recent.

    Margin percentage is a RATIO, so it is smoothed by rolling the numerator
    and denominator separately and dividing after — rolling the ratio itself
    would weight a quiet month the same as a heavy one.
    """
    if "margin" not in df.columns or not df["margin"].notna().any():
        return None

    months = full_month_index(df)
    baseline_months, recent_months = _split_recent_baseline(months)

    d = df.copy()
    d["month"] = d["date"].dt.to_period("M")
    monthly_margin = d.groupby("month")["margin"].sum().reindex(months, fill_value=0.0)
    monthly_revenue = d.groupby("month")["revenue"].sum().reindex(months, fill_value=0.0)

    def rate_for(window: pd.PeriodIndex) -> float | None:
        if len(window) == 0:
            return None
        revenue = float(monthly_revenue.loc[window].sum())
        if revenue == 0:
            return None
        return round(float(monthly_margin.loc[window].sum()) / revenue, 4)

    def monthly_rate(window: pd.PeriodIndex) -> float | None:
        if len(window) == 0:
            return None
        return round(float(monthly_margin.loc[window].sum()) / len(window), 2)

    baseline_pct, recent_pct = rate_for(baseline_months), rate_for(recent_months)
    change_pp = (
        round((recent_pct - baseline_pct) * 100, 2)
        if baseline_pct is not None and recent_pct is not None else None
    )

    smoothed_pct = (rolling(monthly_margin) / rolling(monthly_revenue)).replace(
        [np.inf, -np.inf], np.nan
    )

    return {
        "baseline_margin_pct": baseline_pct,
        "recent_margin_pct": recent_pct,
        "margin_pct_change_pp": change_pp,
        "baseline_monthly_margin": monthly_rate(baseline_months),
        "recent_monthly_margin": monthly_rate(recent_months),
        "monthly_margin_pct_series": {
            str(m): (None if pd.isna(v) else round(float(v), 4)) for m, v in smoothed_pct.items()
        },
    }


def discount_profile(df: pd.DataFrame) -> dict | None:
    """Revenue-weighted average discount, baseline vs recent.

    Weighted by gross list value (list_price x quantity) where available, so
    that a discount on a large high-value line counts for more than the same
    discount on one cheap line — an unweighted mean of discount_pct would let
    a handful of small orders mask creep on the lines that matter. Falls back
    to |revenue| weighting when list_price is absent.
    """
    if "discount_pct" not in df.columns or not df["discount_pct"].notna().any():
        return None

    months = full_month_index(df)
    baseline_months, recent_months = _split_recent_baseline(months)

    d = _sales_lines(df).copy()
    d = d[d["discount_pct"].notna()]
    if d.empty:
        return None
    d["month"] = d["date"].dt.to_period("M")

    if "list_price" in d.columns and d["list_price"].notna().any() and "quantity" in d.columns:
        d["weight"] = (d["list_price"] * d["quantity"]).abs()
    else:
        d["weight"] = d["revenue"].abs()
    d.loc[d["weight"].isna() | (d["weight"] <= 0), "weight"] = 1.0
    d["weighted"] = d["discount_pct"] * d["weight"]

    def avg_for(window: pd.PeriodIndex) -> float | None:
        sub = d[d["month"].isin(window)]
        total_weight = float(sub["weight"].sum())
        if len(window) == 0 or total_weight == 0:
            return None
        return round(float(sub["weighted"].sum()) / total_weight, 4)

    baseline_disc, recent_disc = avg_for(baseline_months), avg_for(recent_months)
    change_pp = (
        round((recent_disc - baseline_disc) * 100, 2)
        if baseline_disc is not None and recent_disc is not None else None
    )

    by_month = d.groupby("month").apply(
        lambda g: float(g["weighted"].sum()) / float(g["weight"].sum())
        if float(g["weight"].sum()) else float("nan"),
        include_groups=False,
    ).reindex(months)

    return {
        "baseline_avg_discount_pct": baseline_disc,
        "recent_avg_discount_pct": recent_disc,
        "discount_pct_change_pp": change_pp,
        "monthly_avg_discount_series": {
            str(m): (None if pd.isna(v) else round(float(v), 4)) for m, v in by_month.items()
        },
    }


def tier_mix(df: pd.DataFrame) -> dict | None:
    """Revenue share by product value tier, baseline vs recent.

    This is the decomposition that separates the dataset's two opposite mix
    stories: value sliding out of High into Low at flat revenue (leakage),
    versus sliding INTO High at flat revenue (premiumisation, not leakage).
    Direction of the high-tier share change is the whole signal, so it is
    reported signed and never as a magnitude.
    """
    if "tier" not in df.columns or not df["tier"].notna().any():
        return None

    months = full_month_index(df)
    baseline_months, recent_months = _split_recent_baseline(months)

    d = df[df["tier"].notna()].copy()
    d["month"] = d["date"].dt.to_period("M")

    def shares_for(window: pd.PeriodIndex) -> dict:
        sub = d[d["month"].isin(window)]
        total = float(sub["revenue"].sum())
        if len(window) == 0 or total == 0:
            return {}
        by_tier = sub.groupby("tier")["revenue"].sum()
        return {t: round(float(by_tier.get(t, 0.0)) / total, 4) for t in TIER_ORDER}

    baseline_shares, recent_shares = shares_for(baseline_months), shares_for(recent_months)
    high_change_pp = (
        round((recent_shares.get("High", 0.0) - baseline_shares.get("High", 0.0)) * 100, 2)
        if baseline_shares and recent_shares else None
    )

    monthly_high = d[d["tier"] == "High"].groupby("month")["revenue"].sum().reindex(months, fill_value=0.0)
    monthly_total = d.groupby("month")["revenue"].sum().reindex(months, fill_value=0.0)
    smoothed_high_share = (rolling(monthly_high) / rolling(monthly_total)).replace(
        [np.inf, -np.inf], np.nan
    )

    return {
        "baseline_share_by_tier": baseline_shares,
        "recent_share_by_tier": recent_shares,
        "high_tier_share_change_pp": high_change_pp,
        "monthly_high_tier_share_series": {
            str(m): (None if pd.isna(v) else round(float(v), 4))
            for m, v in smoothed_high_share.items()
        },
    }


def order_behavior(df: pd.DataFrame) -> dict:
    months = full_month_index(df)
    baseline_months, recent_months = _split_recent_baseline(months)

    df = _sales_lines(df)
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
        """Window MEAN, not median.

        Basket width is a small integer (typically 2-6 lines), so its median
        moves only in whole steps: a healthy account drifting from 4.4 to 3.6
        lines per order reads as a flat -25% or -50% cliff, and two genuinely
        different accounts collapse onto the same number. The mean over every
        order in the window tracks the underlying drift smoothly, which is
        what a gradual fragmentation signal actually needs.
        """
        if len(window) == 0:
            return None
        oids = orders.loc[orders["month"].isin(window), "order_id"]
        vals = series.loc[series.index.intersection(oids)]
        return round(float(vals.mean()), 2) if len(vals) else None

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
        "margin_profile": margin_profile(acc_df),
        "discount_profile": discount_profile(acc_df),
        "category_mix": category_mix(acc_df),
        "tier_mix": tier_mix(acc_df),
        "order_behavior": order_behavior(acc_df),
    }
