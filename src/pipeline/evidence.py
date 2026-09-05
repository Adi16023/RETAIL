"""
Evidence pack assembly — combines Stage 1 (sufficiency), Stage 2 (baseline)
and Stage 3 (change detection) into the single structured JSON object the
Stage 4 LLM reasons over. The LLM never sees raw transactions and never
computes a number; every figure here traces back to deterministic code.

The pack is organised by DIMENSION (revenue, margin, discount, category mix,
tier mix, order pattern) rather than as a flat bag of numbers, because the
central discrimination this dataset demands is *which* dimension moved: flat
revenue with collapsing margin is a leak, flat revenue with rising margin is
a healthy account trading up, and the two are identical if you only look at
revenue. `analysis_dimensions` states which of those dimensions the input
file could actually support, so "not measured" is never mistaken for "no
change found".
"""

from __future__ import annotations

import pandas as pd

from .baseline import build_baseline
from .detect import (
    category_changes,
    data_gaps,
    dip_episodes,
    discount_creep,
    margin_erosion,
    order_pattern_shift,
    outlier_months,
    prior_year_echo,
    product_changes,
    returns_summary,
    revenue_decline,
    revenue_trend,
    tier_shift,
)
from .ingest import account_sufficiency, analysis_dimensions
from .significance import significance_tests


def _attach_p_values(pack: dict, significance: dict) -> None:
    """Put each dimension's p-value next to the status it qualifies.

    The detector says how BIG the shift is; the permutation test says whether
    a shift that big is distinguishable from this account's own noise. Kept
    side by side so a reader — human or model — never sees one without the
    other. Blocks that are None (dimension unavailable) are left alone.
    """
    pairs = (
        ("revenue_decline", "revenue"),
        ("margin", "margin"),
        ("discount", "discount"),
        ("tier_mix", "tier_mix"),
    )
    for block_key, sig_key in pairs:
        block = pack.get(block_key)
        if isinstance(block, dict):
            result = significance.get(sig_key, {})
            block["p_value"] = result.get("p_value")
            block["significant"] = result.get("significant")

    order = pack.get("order_pattern")
    if isinstance(order, dict):
        order["p_value_frequency"] = significance.get("order_frequency", {}).get("p_value")
        order["p_value_basket_width"] = significance.get("basket_width", {}).get("p_value")
        order["significant"] = bool(
            significance.get("order_frequency", {}).get("significant")
            or significance.get("basket_width", {}).get("significant")
        )


def _monthly_revenue_series(monthly_series: dict) -> pd.Series:
    return pd.Series(
        {pd.Period(month, freq="M"): float(value) for month, value in monthly_series.items()}
    ).sort_index()


def build_evidence_pack(df: pd.DataFrame, account_id: str) -> dict:
    acc_df = df[df["account_id"] == account_id].reset_index(drop=True)
    if acc_df.empty:
        raise ValueError(f"No rows for account_id={account_id!r}")

    sufficiency = account_sufficiency(df).get(account_id)
    baseline = build_baseline(df, account_id)
    changes = category_changes(acc_df, baseline["category_mix"]["high_value_categories"])
    products = product_changes(acc_df)
    trend = revenue_trend(baseline["overall_revenue"]["monthly_series"])
    revenue_series = _monthly_revenue_series(baseline["overall_revenue"]["monthly_series"])

    identity = {"account_id": account_id}
    for column, key in (("account_name", "account_name"), ("region", "region"),
                        ("account_manager", "account_manager")):
        if column in acc_df.columns and acc_df[column].notna().any():
            values = sorted(acc_df[column].dropna().astype(str).unique())
            identity[key] = values[0] if len(values) == 1 else values
            if column == "account_manager" and len(values) > 1:
                # A mid-history rep change correlates with all sorts of
                # things and causes none of them by itself. Surfaced as a
                # fact, explicitly labelled, so it can be mentioned without
                # being promoted to a cause.
                identity["account_manager_changed"] = True

    significance = significance_tests(acc_df)

    pack = {
        **identity,
        "data_sufficiency": sufficiency,
        "analysis_dimensions": analysis_dimensions(acc_df),
        "history": {
            "start_date": str(acc_df["date"].min().date()),
            "end_date": str(acc_df["date"].max().date()),
            "months_of_history": sufficiency["history_months"] if sufficiency else None,
            "order_count": sufficiency["order_count"] if sufficiency else None,
            "category_count": sufficiency["category_count"] if sufficiency else None,
        },
        "overall_revenue": baseline["overall_revenue"],
        "revenue_trend": trend,
        "revenue_decline": revenue_decline(trend, baseline["overall_revenue"].get("pct_change")),
        # Seasonality and dip history are checked on TOTAL revenue as well as
        # per category: a genuinely seasonal account dips across its whole
        # book at once, which no single category's change-point would show.
        "seasonality": prior_year_echo(revenue_series),
        "dip_episodes": dip_episodes(revenue_series),
        "margin": margin_erosion(baseline["margin_profile"]),
        "margin_profile": baseline["margin_profile"],
        "discount": discount_creep(baseline["discount_profile"]),
        "tier_mix": tier_shift(baseline["tier_mix"]),
        "category_mix": baseline["category_mix"],
        "category_changes": changes,
        "product_changes": products,
        "order_pattern": order_pattern_shift(baseline["order_behavior"]),
        "order_behavior": baseline["order_behavior"],
        "data_quality": {
            "gaps": data_gaps(acc_df),
            "outlier_months": outlier_months(acc_df),
            "returns": returns_summary(acc_df),
        },
        # Noise-relative significance per dimension. `p_value` is the share of
        # random re-orderings of this account's own months that produce a
        # recent-vs-baseline shift at least this large. Below `level` the
        # shift is real; above it, the threshold detector's status describes
        # something the account's ordinary lumpiness could have produced.
        "significance": significance,
    }
    _attach_p_values(pack, significance)
    return pack
