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


def prioritize(impact: dict, verdict: dict) -> dict:
    if verdict.get("defer") or verdict.get("verdict") == "insufficient_data":
        return {
            "priority": "Deferred",
            "reason": "Evidence insufficient for a confident verdict — see data_needed_if_deferring.",
            "churn_risk_projection": None,
        }

    if verdict.get("verdict") == "healthy" or not impact["per_category"]:
        return {"priority": "None", "reason": "No leakage detected.", "churn_risk_projection": None}

    magnitude = _magnitude_tier(impact["pct_of_baseline_monthly_revenue_at_risk"])
    confidence = verdict.get("confidence", "low")
    priority = PRIORITY_MATRIX[(magnitude, confidence)]

    monthly_loss = impact["total_monthly_revenue_at_risk"]
    churn_risk_projection = {
        "monthly_run_rate_loss": monthly_loss,
        "projected_12_month_loss_if_unaddressed": round(monthly_loss * 12, 2),
        "basis": (
            "Assumes the current monthly loss rate persists unchanged for 12 months — "
            "not a forecast of further deterioration, just the cost of inaction at today's rate."
        ),
    } if monthly_loss > 0 else None

    return {
        "priority": priority,
        "magnitude_tier": magnitude,
        "confidence_tier": confidence,
        "reason": (
            f"{round(impact['pct_of_baseline_monthly_revenue_at_risk'] * 100, 1)}% of baseline monthly "
            f"revenue at risk, {confidence} confidence."
            if impact["pct_of_baseline_monthly_revenue_at_risk"] is not None else
            "Revenue at risk could not be expressed as a share of baseline."
        ),
        "churn_risk_projection": churn_risk_projection,
    }
