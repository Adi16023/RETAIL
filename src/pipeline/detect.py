"""
Stage 3: Change / Leakage Detector — finds meaningful deviations from the
Stage 2 baseline. Deterministic only; produces facts, not verdicts. Whether
a detected change is "temporary" or "structural" is a judgement call left
to Stage 4 (the LLM) — this stage only supplies the evidence: does a
change-point exist, is it sustained, does it have seasonal precedent.

Noise handling: per-month revenue can swing widely on a low-order-count
account (see plan.md "known issue"), so every series here is smoothed with
a trailing 3-month rolling sum before any threshold or change-point logic
runs — a single noisy month must not by itself register as a change-point.
"""

from __future__ import annotations

import pandas as pd

from .changepoint import (
    DECLINE_THRESHOLD,
    MIN_BASELINE_MONTHLY_REVENUE,
    RECENT_MONTHS,
    full_month_index,
    normalize_medians_to_monthly_rate,
    rolling,
    scan_change_point,
)


def _monthly_category_series(df: pd.DataFrame, category: str, months: pd.PeriodIndex) -> pd.Series:
    d = df[df["category"] == category].copy()
    d["month"] = d["date"].dt.to_period("M")
    monthly = d.groupby("month")["revenue"].sum().reindex(months, fill_value=0.0)
    return monthly


SEASONAL_INDEX_THRESHOLD = 0.65  # earlier target-month typical value must be below this fraction of contemporaneous peer months
MIN_EARLIER_MONTHS_FOR_SEASONALITY_CHECK = 3


def _seasonal_precedent(df: pd.DataFrame, category: str, decline_month: str, months: pd.PeriodIndex) -> str:
    """Check whether the decline's calendar month was *already* a low point
    for this category, relative to its contemporaneous peer months, before
    the decline in question happened. Returns 'confirmed', 'not_present',
    or 'insufficient_history'.

    Critically, only months strictly before decline_month are used. Without
    that restriction, a genuinely structural collapse would always look
    "seasonal" for whatever calendar month it happened to start in — the
    collapsed value itself would drag down whatever average included it.
    Comparing to contemporaneous peers (not a global average) also avoids
    mistaking a general account-wide trend for calendar-month seasonality.
    """
    target_month_num = pd.Period(decline_month, freq="M").month
    monthly = _monthly_category_series(df, category, months)

    earlier_months = [m for m in months if str(m) < decline_month]
    if len(earlier_months) < MIN_EARLIER_MONTHS_FOR_SEASONALITY_CHECK:
        return "insufficient_history"

    earlier_target_occurrences = [m for m in earlier_months if m.month == target_month_num]
    if len(earlier_target_occurrences) < 2:
        # A single prior occurrence is one data point, not a pattern — it
        # can't distinguish genuine recurring seasonality from an ordinary
        # noisy month (this is what produced false "confirmed" calls
        # before this fix). Requires ~2+ years of history to ever confirm,
        # which is honest: the project's own criteria say history under
        # 12-18 months often can't resolve a seasonal question at all.
        return "insufficient_history"

    earlier_peer_months = [m for m in earlier_months if m.month != target_month_num]
    peer_vals = monthly.loc[monthly.index.isin(earlier_peer_months)]
    if len(peer_vals) == 0:
        return "insufficient_history"
    peer_typical = peer_vals.median()
    if peer_typical <= 0:
        return "insufficient_history"

    target_typical = monthly.loc[monthly.index.isin(earlier_target_occurrences)].median()

    if target_typical <= peer_typical * SEASONAL_INDEX_THRESHOLD:
        return "confirmed"
    return "not_present"


def category_changes(df: pd.DataFrame, high_value_categories: list[str]) -> list[dict]:
    months = full_month_index(df)
    results = []
    for category in sorted(df["category"].unique()):
        raw = _monthly_category_series(df, category, months)
        smoothed = rolling(raw)
        cp = scan_change_point(smoothed)
        if cp:
            cp = normalize_medians_to_monthly_rate(cp)
        entry = {
            "category": category,
            "is_high_value": category in high_value_categories,
            "change_point_detected": cp is not None,
        }
        if cp:
            entry.update(cp)
            entry["seasonal_precedent"] = _seasonal_precedent(df, category, cp["change_point_month"], months)
        results.append(entry)

    results.sort(key=lambda r: r.get("pct_decline", 0) if r["change_point_detected"] else -1, reverse=True)
    return results


def product_changes(df: pd.DataFrame, top_n: int = 10) -> list[dict]:
    """Products whose revenue disappeared or materially declined between
    baseline and recent windows — locates leakage below category level."""
    months = full_month_index(df)
    if len(months) <= RECENT_MONTHS:
        return []
    baseline_months, recent_months = months[:-RECENT_MONTHS], months[-RECENT_MONTHS:]

    d = df.copy()
    d["month"] = d["date"].dt.to_period("M")
    by_product = d.groupby(["product_id", "category", "month"])["revenue"].sum().reset_index()

    baseline = by_product[by_product["month"].isin(baseline_months)].groupby(
        ["product_id", "category"])["revenue"].sum() / max(len(baseline_months), 1)
    recent = by_product[by_product["month"].isin(recent_months)].groupby(
        ["product_id", "category"])["revenue"].sum() / max(len(recent_months), 1)

    all_products = baseline.index.union(recent.index)
    out = []
    for key in all_products:
        product_id, category = key
        b = float(baseline.get(key, 0.0))
        r = float(recent.get(key, 0.0))
        if b < MIN_BASELINE_MONTHLY_REVENUE and r < MIN_BASELINE_MONTHLY_REVENUE:
            continue
        if b > 0 and r == 0:
            status = "disappeared"
        elif b > 0 and (b - r) / b >= DECLINE_THRESHOLD:
            status = "declined"
        elif b == 0 and r > 0:
            status = "new"
        else:
            status = "stable"
        if status == "stable":
            continue
        out.append({
            "product_id": product_id, "category": category,
            "baseline_monthly_avg": round(b, 2), "recent_monthly_avg": round(r, 2),
            "status": status,
        })

    out.sort(key=lambda r: r["baseline_monthly_avg"], reverse=True)
    return out[:top_n]
