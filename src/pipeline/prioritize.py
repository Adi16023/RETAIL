"""
Stage 6: Prioritisation Engine — deterministic rules only, no LLM call.
Combines Stage 5's rupee impact with Stage 4's confidence into a priority
bucket and a churn-risk projection, per plan.md's consolidated architecture
("prioritisation ranking can be a formula, not a model call").
"""

from __future__ import annotations

MAGNITUDE_HIGH = 0.15   # >=15% of baseline monthly revenue at risk
MAGNITUDE_MEDIUM = 0.05  # >=5%

# (magnitude tier, confidence tier) -> priority. A large $ figure resting on
# low confidence is downgraded to Medium, not High — acting on shaky
# evidence is itself a risk (the 30%-weighted rubric line).
PRIORITY_MATRIX = {
    ("high", "high"): "High", ("high", "medium"): "High", ("high", "low"): "Medium",
    ("medium", "high"): "Medium", ("medium", "medium"): "Medium", ("medium", "low"): "Low",
    ("low", "high"): "Low", ("low", "medium"): "Low", ("low", "low"): "Low",
}


def _magnitude_tier(pct_at_risk: float | None) -> str:
    if pct_at_risk is None:
        return "low"
    if pct_at_risk >= MAGNITUDE_HIGH:
        return "high"
    if pct_at_risk >= MAGNITUDE_MEDIUM:
        return "medium"
    return "low"


def _has_sized_leak(impact: dict) -> bool:
    return bool(
        impact.get("per_category")
        or impact.get("account_level_revenue_impact")
        or impact.get("margin_impact")
    )


def prioritize(impact: dict, verdict: dict) -> dict:
    if verdict.get("defer") or verdict.get("verdict") == "insufficient_data":
        return {
            "priority": "Deferred",
            "reason": "Evidence insufficient for a confident verdict — see data_needed_if_deferring.",
            "churn_risk_projection": None,
        }

    if verdict.get("verdict") == "healthy" or not _has_sized_leak(impact):
        return {"priority": "None", "reason": "No leakage detected.", "churn_risk_projection": None}

    # Severity is the worse of the revenue and margin ratios (see impact.py):
    # a flat-revenue account bleeding a third of its gross margin must not
    # rank Low just because its topline is intact.
    severity = impact.get("overall_severity_pct_of_baseline")
    magnitude = _magnitude_tier(severity)
    confidence = verdict.get("confidence", "low")
    priority = PRIORITY_MATRIX[(magnitude, confidence)]

    revenue_loss = impact.get("total_monthly_revenue_at_risk") or 0.0
    margin_impact = impact.get("margin_impact") or {}
    margin_loss = margin_impact.get("monthly_margin_at_risk") or 0.0

    churn_risk_projection = None
    if revenue_loss > 0 or margin_loss > 0:
        churn_risk_projection = {
            "monthly_run_rate_loss": revenue_loss,
            "projected_12_month_loss_if_unaddressed": round(revenue_loss * 12, 2),
            "monthly_margin_loss": margin_loss,
            "projected_12_month_margin_loss_if_unaddressed": round(margin_loss * 12, 2),
            "basis": (
                "Assumes the current monthly loss rate persists unchanged for 12 months — "
                "not a forecast of further deterioration, just the cost of inaction at today's "
                "rate. Revenue and margin figures overlap and must not be added together."
            ),
        }

    reasons = []
    revenue_pct = impact.get("pct_of_baseline_monthly_revenue_at_risk")
    margin_pct = impact.get("pct_of_baseline_monthly_margin_at_risk")
    if revenue_pct:
        reasons.append(f"{round(revenue_pct * 100, 1)}% of baseline monthly revenue at risk")
    if margin_pct:
        reasons.append(f"{round(margin_pct * 100, 1)}% of baseline monthly gross margin at risk")
    reason = (
        f"{'; '.join(reasons)}, {confidence} confidence."
        if reasons else "Leak could not be expressed as a share of baseline."
    )

    return {
        "priority": priority,
        "magnitude_tier": magnitude,
        "confidence_tier": confidence,
        "severity_pct_of_baseline": severity,
        "reason": reason,
        "churn_risk_projection": churn_risk_projection,
    }
