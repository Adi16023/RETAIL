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

Revenue decline is only one of the things that can go wrong. This stage
also detects margin erosion at flat revenue, discount creep, an outright
category defection, prior-year seasonal echoes, gaps in the order history,
and one-off bulk months — because in the reference dataset the majority of
accounts that LOOK like leaks at revenue level are not leaks, and two of
the real leaks are invisible at revenue level entirely. Each detector still
produces facts, never a verdict.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .changepoint import (
    DECLINE_THRESHOLD,
    MIN_BASELINE_MONTHLY_REVENUE,
    RECENT_MONTHS,
    ROLLING_WINDOW,
    full_month_index,
    normalize_medians_to_monthly_rate,
    rolling,
    scan_change_point,
    winsorize,
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


# --- Prior-year echo -------------------------------------------------------
#
# _seasonal_precedent above needs 2+ prior occurrences of the same calendar
# month, which takes ~3 years of history to ever satisfy. With the 24-month
# reference dataset it therefore returns "insufficient_history" on exactly
# the account it was meant to resolve (a high-tier dip repeating in the same
# calendar window two years running). The prior-year echo below is the check
# that 24 months CAN support: not "is this month typically low", but "did
# this exact dip also happen twelve months ago". One prior year is one
# comparison, so a confirmed echo is weaker evidence than a multi-year
# seasonal index — reported as its own field so Stage 4 can weigh it as such.

SEASONAL_MONTHS_IN_YEAR = 12
DIP_FRACTION = 0.75            # a month below 75% of the series median counts as a dip
MIN_MONTHS_FOR_ECHO_CHECK = 18  # a recent dip plus a full prior-year window to compare it to


def prior_year_echo(series: pd.Series) -> dict:
    """Does the recent dip in this series repeat the same calendar window a
    year earlier? Returns a status plus the months it matched on, so the
    finding can be cited rather than asserted.

    Statuses: 'confirmed', 'not_present', 'no_recent_dip', 'insufficient_history'.
    """
    months = list(series.index)
    if len(months) < MIN_MONTHS_FOR_ECHO_CHECK:
        return {"status": "insufficient_history", "dip_months": [], "echo_months": []}

    reference = float(winsorize(series).median())
    if reference <= 0:
        return {"status": "insufficient_history", "dip_months": [], "echo_months": []}

    threshold = reference * DIP_FRACTION
    recent_window = months[-SEASONAL_MONTHS_IN_YEAR:]
    dip_months = [m for m in recent_window if float(series.loc[m]) < threshold]
    if not dip_months:
        return {"status": "no_recent_dip", "dip_months": [], "echo_months": []}

    echo_months = []
    for month in dip_months:
        prior = month - SEASONAL_MONTHS_IN_YEAR
        if prior in series.index and float(series.loc[prior]) < threshold:
            echo_months.append(prior)

    # A single echoing month out of many dipped months is coincidence, not a
    # pattern: require both an absolute floor and that most of the dip is
    # accounted for, so a structural collapse that happens to start in a
    # historically quiet month is not excused as seasonal.
    confirmed = len(echo_months) >= 2 and len(echo_months) >= len(dip_months) / 2
    return {
        "status": "confirmed" if confirmed else "not_present",
        "dip_months": [str(m) for m in dip_months],
        "echo_months": [str(m) for m in echo_months],
        "dip_threshold": round(threshold, 2),
    }


# --- Category-level detection ---------------------------------------------

# A category counts as defected (not merely declining) once it goes to zero
# and stays there. Below this it is still plausibly an ordering gap.
MIN_MONTHS_AT_ZERO_FOR_DEFECTION = 3


def _trailing_zero_months(series: pd.Series) -> int:
    count = 0
    for value in reversed(list(series)):
        if float(value) != 0.0:
            break
        count += 1
    return count


def category_changes(df: pd.DataFrame, high_value_categories: list[str]) -> list[dict]:
    months = full_month_index(df)
    results = []
    for category in sorted(df["category"].unique()):
        raw = _monthly_category_series(df, category, months)
        smoothed = rolling(raw)
        cp = scan_change_point(smoothed)
        if cp:
            cp = normalize_medians_to_monthly_rate(cp)

        trailing_zeros = _trailing_zero_months(raw)
        had_baseline = float(raw.iloc[:-RECENT_MONTHS].sum()) > 0 if len(raw) > RECENT_MONTHS else False

        entry = {
            "category": category,
            "is_high_value": category in high_value_categories,
            "change_point_detected": cp is not None,
            # A clean stop is a different business event from a decline: it
            # names a specific line lost, which is what makes attribution
            # actionable rather than a generic "revenue is down".
            "defected": bool(had_baseline and trailing_zeros >= MIN_MONTHS_AT_ZERO_FOR_DEFECTION),
            "consecutive_months_at_zero": trailing_zeros,
        }
        if cp:
            entry.update(cp)
            entry["seasonal_precedent"] = _seasonal_precedent(df, category, cp["change_point_month"], months)
            entry["prior_year_echo"] = prior_year_echo(raw)
        results.append(entry)

    results.sort(key=lambda r: r.get("pct_decline", 0) if r["change_point_detected"] else -1, reverse=True)
    return results


# --- Margin, discount, and mix detection ----------------------------------
#
# Thresholds are in PERCENTAGE POINTS of rate, not percent-of-percent, and
# were set against the reference dataset's own spread: across the twelve
# accounts with no margin story the baseline-to-recent margin rate moves by
# at most ~1pp, while the two genuine margin leaks move 11-12pp. 3pp sits an
# order of magnitude above the noise floor and well below either real
# signal, so it is not tuned to a specific account.

MARGIN_EROSION_PP = 3.0
DISCOUNT_CREEP_PP = 3.0
HIGH_TIER_SHIFT_PP = 8.0


def margin_erosion(margin_profile: dict | None) -> dict | None:
    """Is this account's margin RATE falling while it keeps buying?

    Reads the Stage 2 profile rather than recomputing, so the number the
    agent cites and the number the impact engine sizes from are the same one.
    Direction matters: an improving rate is reported as improvement, not as a
    smaller erosion, because trading up is not a leak.
    """
    if margin_profile is None:
        return None

    change_pp = margin_profile.get("margin_pct_change_pp")
    if change_pp is None:
        return {"status": "insufficient_history", "margin_pct_change_pp": None}

    if change_pp <= -MARGIN_EROSION_PP:
        status = "erosion_detected"
    elif change_pp >= MARGIN_EROSION_PP:
        status = "improvement_detected"
    else:
        status = "stable"

    series = margin_profile.get("monthly_margin_pct_series") or {}
    baseline_pct = margin_profile.get("baseline_margin_pct")
    months_below = 0
    if baseline_pct is not None:
        floor = baseline_pct - (MARGIN_EROSION_PP / 100)
        months_below = sum(
            1 for v in series.values() if v is not None and v < floor
        )

    return {
        "status": status,
        "baseline_margin_pct": baseline_pct,
        "recent_margin_pct": margin_profile.get("recent_margin_pct"),
        "margin_pct_change_pp": change_pp,
        "months_below_baseline_margin": months_below,
        "threshold_pp": MARGIN_EROSION_PP,
    }


def discount_creep(discount_profile: dict | None) -> dict | None:
    """Is the same business being sold at a steadily deeper discount?

    This is the one leak that a revenue-only pipeline cannot see at all:
    volume, mix and order pattern are unchanged, and the money leaves through
    the price.
    """
    if discount_profile is None:
        return None

    change_pp = discount_profile.get("discount_pct_change_pp")
    if change_pp is None:
        return {"status": "insufficient_history", "discount_pct_change_pp": None}

    if change_pp >= DISCOUNT_CREEP_PP:
        status = "creep_detected"
    elif change_pp <= -DISCOUNT_CREEP_PP:
        status = "discipline_improved"
    else:
        status = "stable"

    return {
        "status": status,
        "baseline_avg_discount_pct": discount_profile.get("baseline_avg_discount_pct"),
        "recent_avg_discount_pct": discount_profile.get("recent_avg_discount_pct"),
        "discount_pct_change_pp": change_pp,
        "threshold_pp": DISCOUNT_CREEP_PP,
    }


def tier_shift(tier_mix: dict | None) -> dict | None:
    """Which way is value moving between price tiers?

    Reported with an explicit direction because the two directions are
    opposite verdicts on identical-looking flat revenue: High -> Low is the
    hidden downgrade, Low -> High is premiumisation and must not be flagged.
    """
    if tier_mix is None:
        return None

    change_pp = tier_mix.get("high_tier_share_change_pp")
    if change_pp is None:
        return {"status": "insufficient_history", "high_tier_share_change_pp": None}

    if change_pp <= -HIGH_TIER_SHIFT_PP:
        status = "downgrade_detected"
    elif change_pp >= HIGH_TIER_SHIFT_PP:
        status = "premiumisation_detected"
    else:
        status = "stable"

    return {
        "status": status,
        "baseline_share_by_tier": tier_mix.get("baseline_share_by_tier"),
        "recent_share_by_tier": tier_mix.get("recent_share_by_tier"),
        "high_tier_share_change_pp": change_pp,
        "threshold_pp": HIGH_TIER_SHIFT_PP,
    }


# --- Revenue decline, graded -----------------------------------------------
#
# scan_change_point only finds STEP changes: it splits the series in two and
# compares medians, so a gentle slide that never steps loses ~25% between the
# two halves and never crosses the 35% threshold. That is the correct
# behaviour for a step detector and the wrong answer for an account that has
# quietly shed a fifth of its revenue, so gradual decline gets its own graded
# detector here.
#
# Both conditions must hold for "material": half-over-half size AND a
# consistent negative slope. Either alone is too easy to trip — a single
# month with no orders moves the half-over-half comparison by ~8% without any
# change in behaviour, and does not move the slope.

MATERIAL_DECLINE_HALVES = -0.15
MATERIAL_DECLINE_SLOPE = -0.015
MILD_DRIFT_HALVES = -0.05
MILD_DRIFT_SLOPE = -0.004
GROWTH_HALVES = 0.05


def revenue_decline(trend: dict) -> dict:
    """Grade the total-revenue trajectory: material decline, mild drift,
    stable, or growth. Graded rather than boolean because the reference
    dataset deliberately includes a ~12% drift that must NOT be flagged
    sitting just below declines of ~20% that must be."""
    halves = trend.get("first_half_vs_second_half_pct_change")
    slope = trend.get("slope_pct_per_month")

    if halves is None or slope is None:
        status = "insufficient_history"
    elif halves <= MATERIAL_DECLINE_HALVES and slope <= MATERIAL_DECLINE_SLOPE:
        status = "material_decline"
    elif halves <= MILD_DRIFT_HALVES or slope <= MILD_DRIFT_SLOPE:
        status = "mild_drift"
    elif halves >= GROWTH_HALVES:
        status = "growth"
    else:
        status = "stable"

    return {
        "status": status,
        "first_half_vs_second_half_pct_change": halves,
        "slope_pct_per_month": slope,
        "material_thresholds": {
            "first_half_vs_second_half_pct_change": MATERIAL_DECLINE_HALVES,
            "slope_pct_per_month": MATERIAL_DECLINE_SLOPE,
            "note": "both must be met for 'material_decline'",
        },
    }


# --- Dip episodes ----------------------------------------------------------

RECOVERY_LEVEL = 0.9  # back to 90% of the series median counts as recovered


def dip_episodes(series: pd.Series) -> list[dict]:
    """Every run of consecutive below-normal months in the history, each
    labelled recovered or ongoing.

    A dip that ended a year ago and a dip that is still running look
    identical in any single baseline-vs-recent comparison, but they are
    opposite verdicts: one is a resolved incident, the other is the leak.
    Listing episodes with their end state lets Stage 4 tell them apart, and
    lets it say "recovered in <month>" instead of merely "no change-point".
    """
    if len(series) < ROLLING_WINDOW:
        return []
    reference = float(winsorize(series).median())
    if reference <= 0:
        return []

    below = series < reference * DIP_FRACTION
    episodes = []
    start = None
    for month, is_low in below.items():
        if is_low and start is None:
            start = month
        elif not is_low and start is not None:
            episodes.append((start, month))
            start = None
    if start is not None:
        episodes.append((start, None))

    out = []
    for start, first_normal_month in episodes:
        end = start if first_normal_month is None else first_normal_month - 1
        window = series.loc[start:end]
        if first_normal_month is None:
            recovered, recovered_by = False, None
        else:
            after = series.loc[first_normal_month:]
            recovered = bool((after >= reference * RECOVERY_LEVEL).any())
            recovered_by = str(first_normal_month) if recovered else None
        out.append({
            "start_month": str(start),
            "end_month": str(end),
            "months": int(len(window)),
            "trough_revenue": round(float(window.min()), 2),
            "typical_monthly_revenue": round(reference, 2),
            "recovered": recovered,
            "recovered_from_month": recovered_by,
            "is_ongoing": first_normal_month is None,
        })
    return out


# --- Order pattern ---------------------------------------------------------

FRAGMENTATION_FREQUENCY_PP = 0.30   # orders per month up by 30%+
FRAGMENTATION_WIDTH_PP = -0.20      # lines per order down by 20%+


def order_pattern_shift(order_behavior: dict) -> dict:
    """More orders, each smaller — the shape of an account that has started
    buying part of its basket somewhere else.

    This is its own leakage dimension, not a modifier: revenue can stay
    almost flat while the account quietly multi-sources, and no category
    change-point fires because every category still shows up, just less of it
    per order.
    """
    frequency = order_behavior.get("order_frequency_pct_change")
    width = order_behavior.get("basket_width_pct_change")

    if frequency is None or width is None:
        status = "insufficient_history"
    elif frequency >= FRAGMENTATION_FREQUENCY_PP and width <= FRAGMENTATION_WIDTH_PP:
        status = "fragmentation_detected"
    elif width <= FRAGMENTATION_WIDTH_PP:
        status = "baskets_shrinking"
    else:
        status = "stable"

    return {
        "status": status,
        "order_frequency_pct_change": frequency,
        "basket_width_pct_change": width,
        "aov_pct_change": order_behavior.get("aov_pct_change"),
        "thresholds": {
            "order_frequency_pct_change": FRAGMENTATION_FREQUENCY_PP,
            "basket_width_pct_change": FRAGMENTATION_WIDTH_PP,
        },
    }


# --- Data-quality events ---------------------------------------------------
#
# These are not leakage signals. They exist so that Stage 4 can tell an
# artefact of the data from a change in the customer's behaviour — a month
# with no orders and a month where a big one-off landed both distort a
# trailing window, and both have been mistaken for trend breaks.

BULK_OUTLIER_MULTIPLE = 2.0


def data_gaps(df: pd.DataFrame) -> dict:
    """Months inside the account's own history with no orders at all."""
    months = full_month_index(df)
    d = df.copy()
    d["month"] = d["date"].dt.to_period("M")
    active = set(d["month"].unique())
    missing = [str(m) for m in months if m not in active]
    return {
        "months_in_history": len(months),
        "months_with_no_orders": missing,
        "note": (
            "A month with no orders is a gap in ordering, not a zero-revenue month "
            "of trading. It drags any trailing average down without any change in "
            "customer behaviour."
        ) if missing else None,
    }


def outlier_months(df: pd.DataFrame) -> list[dict]:
    """Months whose revenue is a large multiple of the account's typical
    month — a stock-up or bulk order. Flagged so that neither the spike
    itself nor the ordinary months after it are read as a trend break."""
    months = full_month_index(df)
    d = df.copy()
    d["month"] = d["date"].dt.to_period("M")
    monthly = d.groupby("month")["revenue"].sum().reindex(months, fill_value=0.0)
    median = float(monthly.median())
    if median <= 0:
        return []
    return [
        {
            "month": str(m),
            "revenue": round(float(v), 2),
            "multiple_of_median_month": round(float(v) / median, 2),
        }
        for m, v in monthly.items()
        if float(v) >= median * BULK_OUTLIER_MULTIPLE
    ]


def returns_summary(df: pd.DataFrame) -> dict | None:
    """Return/credit lines, netted. Present so a credit note is never read as
    a revenue decline — the value is already netted into every other figure."""
    if "is_return" not in df.columns:
        return None
    returns = df[df["is_return"] == 1]
    if returns.empty:
        return {"return_lines": 0, "net_return_value": 0.0, "share_of_gross_revenue": 0.0}

    gross = float(df.loc[df["is_return"] != 1, "revenue"].sum())
    net_value = float(returns["revenue"].sum())
    return {
        "return_lines": int(len(returns)),
        "net_return_value": round(net_value, 2),
        "share_of_gross_revenue": round(abs(net_value) / gross, 4) if gross else None,
        "months": sorted({str(m) for m in returns["date"].dt.to_period("M")}),
        "note": "Already netted into all revenue and margin figures; not a decline.",
    }


def revenue_trend(monthly_series: dict) -> dict:
    """Direction and shape of the total-revenue series.

    Reported separately from change-point detection because the change-point
    scanner only ever looks for DECLINES: without this, a growing account and
    a flat one are indistinguishable in the evidence pack (both simply have
    no change-point), and growth cannot be recognised as growth.
    """
    if not monthly_series:
        return {"direction": "unknown", "slope_pct_per_month": None}

    # Winsorized first: a single 2.8x stock-up month landing just after the
    # midpoint tilts a half-over-half comparison by ~13% and reads as growth
    # on an account that is simply flat. Capping it keeps the trend a
    # statement about the trend.
    values = winsorize(pd.Series([float(v) for v in monthly_series.values()])).to_numpy()
    # Same bar as every other comparative detector: without more months than
    # the recent window, there is no baseline to compare against, and half
    # of a five-month history is not a trend. Saying "unknown" here is what
    # keeps a newly-onboarded account out of the material-decline bucket.
    if len(values) <= RECENT_MONTHS or values.mean() <= 0:
        return {"direction": "unknown", "slope_pct_per_month": None}

    x = np.arange(len(values))
    slope, intercept = np.polyfit(x, values, 1)
    # Slope as a share of the fitted level at the midpoint, so it is
    # comparable across accounts of very different size.
    midpoint_level = intercept + slope * (len(values) / 2)
    slope_pct = round(float(slope / midpoint_level), 4) if midpoint_level > 0 else None

    first_half = values[: len(values) // 2].mean()
    second_half = values[len(values) // 2:].mean()
    endpoint_change = round(float((second_half - first_half) / first_half), 4) if first_half > 0 else None

    # Mirrors MILD_DRIFT_SLOPE below, so `direction` and `revenue_decline`
    # cannot disagree about a gently growing account — an asymmetric pair
    # let a +0.47%/month account read "flat" and "growth" at the same time.
    if slope_pct is None:
        direction = "unknown"
    elif slope_pct >= -MILD_DRIFT_SLOPE:
        direction = "growing"
    elif slope_pct <= MILD_DRIFT_SLOPE:
        direction = "declining"
    else:
        direction = "flat"

    return {
        "direction": direction,
        "slope_pct_per_month": slope_pct,
        "first_half_vs_second_half_pct_change": endpoint_change,
    }


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
