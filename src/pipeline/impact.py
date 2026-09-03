"""
Stage 5: Leakage / Impact Engine — quantifies in rupees, deterministically,
the categories Stage 4 attributed the leakage to. The LLM never computes a
number; every figure here is traceable back to the evidence pack's
category_changes facts (which are themselves traceable to Stage 2/3).
"""

from __future__ import annotations


def compute_impact(evidence_pack: dict, attributed_categories: list[str]) -> dict:
    category_changes_by_name = {c["category"]: c for c in evidence_pack["category_changes"]}
    baseline_share = evidence_pack["category_mix"]["baseline_share"]

    per_category = []
    for category in attributed_categories:
        change = category_changes_by_name.get(category)
        if change is None or not change.get("change_point_detected"):
            # Attributed but no quantifiable change-point fact to size it from —
            # surfaced, not silently dropped, so Stage 7 can flag it.
            per_category.append({
                "category": category,
                "quantifiable": False,
                "reason": "no change-point detected for this category in the evidence pack",
            })
            continue

        monthly_loss = round(change["before_monthly_median"] - change["after_monthly_median"], 2)
        sustained_months = change.get("sustained_months_since_change_point", 0)
        per_category.append({
            "category": category,
            "quantifiable": True,
            "monthly_revenue_at_risk": monthly_loss,
            "historical_value_lost_to_date": round(monthly_loss * sustained_months, 2),
            "annualized_run_rate_loss": round(monthly_loss * 12, 2),
            "persistence_months": sustained_months,
            "recovered": change.get("recovered", False),
            "category_baseline_revenue_share": baseline_share.get(category, 0.0),
        })

    quantifiable = [c for c in per_category if c["quantifiable"]]
    total_monthly_at_risk = round(sum(c["monthly_revenue_at_risk"] for c in quantifiable), 2)
    total_historical_lost = round(sum(c["historical_value_lost_to_date"] for c in quantifiable), 2)

    baseline_monthly_total = evidence_pack["overall_revenue"]["baseline_monthly_median"]
    pct_of_baseline_at_risk = (
        round(total_monthly_at_risk / baseline_monthly_total, 4)
        if baseline_monthly_total else None
    )

    return {
        "per_category": per_category,
        "total_monthly_revenue_at_risk": total_monthly_at_risk,
        "total_historical_value_lost": total_historical_lost,
        "pct_of_baseline_monthly_revenue_at_risk": pct_of_baseline_at_risk,
    }
