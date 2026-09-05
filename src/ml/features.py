"""
The evidence pack, flattened into the model's feature vector.

One function, one row per account. The model sees EXACTLY what the LLM sees —
the same evidence pack, reduced to numbers — so a judge can trace every input
to a chart on screen, and there is no second feature-engineering pipeline to
drift out of step with the first.

Two rules:

- Magnitudes and p-values, never status strings. If the model were fed our
  own "erosion_detected" labels it would simply relearn our thresholds and add
  nothing. It gets margin_change_pp = -12.26 and decides for itself.
- Missing stays missing. A five-month account has no baseline window, so every
  comparative feature is NaN — and "the comparison could not be made" is
  precisely the DEFER signal. Tree models handle NaN natively; we never fill.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SIGNIFICANCE_DIMENSIONS = ("revenue", "margin", "discount", "tier_mix", "order_frequency", "basket_width")


def _num(value) -> float:
    if value is None or isinstance(value, bool) and value is None:
        return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def extract_features(pack: dict) -> dict:
    """Flatten one evidence pack. Every key here is a column of X."""
    overall = pack.get("overall_revenue") or {}
    change_point = overall.get("change_point") or {}
    trend = pack.get("revenue_trend") or {}
    seasonality = pack.get("seasonality") or {}
    episodes = pack.get("dip_episodes") or []
    margin = pack.get("margin") or {}
    discount = pack.get("discount") or {}
    tier = pack.get("tier_mix") or {}
    order = pack.get("order_pattern") or {}
    behaviour = pack.get("order_behavior") or {}
    quality = pack.get("data_quality") or {}
    sufficiency = pack.get("data_sufficiency") or {}
    dimensions = pack.get("analysis_dimensions") or {}
    categories = pack.get("category_changes") or []
    defected = [c for c in categories if c.get("defected")]
    baseline_share = (pack.get("category_mix") or {}).get("baseline_share") or {}
    significance = pack.get("significance") or {}

    features = {
        # history / sufficiency
        "history_months": _num(sufficiency.get("history_months")),
        "order_count": _num(sufficiency.get("order_count")),
        "category_count": _num(sufficiency.get("category_count")),
        "sufficiency_score": _num(sufficiency.get("score")),
        # what the file could support (a missing column is information)
        "has_margin": float(bool(dimensions.get("margin"))),
        "has_discount": float(bool(dimensions.get("discount"))),
        "has_tier": float(bool(dimensions.get("tier_mix"))),
        # revenue
        "rev_pct_change": _num(overall.get("pct_change")),
        "rev_cv": _num(overall.get("monthly_revenue_coefficient_of_variation")),
        "rev_slope_pct_mo": _num(trend.get("slope_pct_per_month")),
        "rev_h1_h2_pct": _num(trend.get("first_half_vs_second_half_pct_change")),
        "rev_cp_detected": float(bool(change_point)),
        "rev_cp_pct_decline": _num(change_point.get("pct_decline")),
        "rev_cp_recovered": _num(change_point.get("recovered")) if change_point else np.nan,
        "rev_cp_sustained_months": _num(change_point.get("sustained_months_since_change_point")),
        # seasonality / dips
        "season_confirmed": float(seasonality.get("status") == "confirmed"),
        "season_dip_months": float(len(seasonality.get("dip_months") or [])),
        "season_echo_months": float(len(seasonality.get("echo_months") or [])),
        "dip_episodes": float(len(episodes)),
        "dip_ongoing": float(any(e.get("is_ongoing") for e in episodes)),
        "dip_recovered": float(any(e.get("recovered") for e in episodes)),
        # margin / discount / tier — magnitudes only
        "margin_base_pct": _num(margin.get("baseline_margin_pct")),
        "margin_recent_pct": _num(margin.get("recent_margin_pct")),
        "margin_change_pp": _num(margin.get("margin_pct_change_pp")),
        "disc_base_pct": _num(discount.get("baseline_avg_discount_pct")),
        "disc_recent_pct": _num(discount.get("recent_avg_discount_pct")),
        "disc_change_pp": _num(discount.get("discount_pct_change_pp")),
        "high_tier_base": _num((tier.get("baseline_share_by_tier") or {}).get("High")),
        "high_tier_recent": _num((tier.get("recent_share_by_tier") or {}).get("High")),
        "high_tier_change_pp": _num(tier.get("high_tier_share_change_pp")),
        # categories
        "cats_with_changepoint": float(sum(bool(c.get("change_point_detected")) for c in categories)),
        "cats_defected": float(len(defected)),
        "defected_baseline_share": float(sum(baseline_share.get(c["category"], 0.0) for c in defected)),
        "max_cat_pct_decline": float(max([c.get("pct_decline") or 0.0 for c in categories
                                          if c.get("change_point_detected")] or [0.0])),
        # order pattern
        "order_freq_pct_change": _num(order.get("order_frequency_pct_change")),
        "basket_width_pct_change": _num(order.get("basket_width_pct_change")),
        "aov_pct_change": _num(behaviour.get("aov_pct_change")),
        # data quality
        "gap_months": float(len((quality.get("gaps") or {}).get("months_with_no_orders") or [])),
        "gap_share": np.nan,
        "outlier_months": float(len(quality.get("outlier_months") or [])),
        "max_outlier_multiple": float(max([o.get("multiple_of_median_month") or 0.0
                                           for o in (quality.get("outlier_months") or [])] or [0.0])),
        "return_lines": _num((quality.get("returns") or {}).get("return_lines")) if quality.get("returns") else 0.0,
        "manager_changed": float(bool(pack.get("account_manager_changed"))),
    }
    months = (quality.get("gaps") or {}).get("months_in_history")
    if months:
        features["gap_share"] = features["gap_months"] / float(months)

    # noise-relative significance — both windows, every dimension
    for name in SIGNIFICANCE_DIMENSIONS:
        result = significance.get(name) or {}
        features[f"p_{name}"] = _num(result.get("p_value"))
        features[f"p_{name}_halves"] = _num(result.get("p_value_halves"))
        features[f"p_{name}_trend"] = _num(result.get("p_value_trend"))
        features[f"stat_{name}"] = _num(result.get("statistic"))
        features[f"stat_{name}_trend"] = _num(result.get("statistic_trend"))
        # The pipeline's combined verdict on "is this shift real" (Bonferroni
        # over both windows). NaN when the test could not run.
        sig = result.get("significant")
        features[f"sig_{name}"] = np.nan if sig is None else float(bool(sig))

    return features


FEATURE_NAMES: list[str] = list(extract_features({}).keys())


def features_frame(packs: dict[str, dict]) -> pd.DataFrame:
    """account_id -> pack, to a DataFrame with a stable column order."""
    rows = {account_id: extract_features(pack) for account_id, pack in packs.items()}
    frame = pd.DataFrame.from_dict(rows, orient="index")[FEATURE_NAMES]
    frame.index.name = "account_id"
    return frame
