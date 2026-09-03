"""
Stage 5: Leakage / Impact Engine — quantifies in rupees, deterministically,
whatever Stage 4 attributed the leakage to. The LLM never computes a number;
every figure here is traceable back to evidence-pack facts (which are
themselves traceable to Stage 2/3).

Three kinds of leak have to be sized, because the dataset contains all
three and they are not interchangeable:

  category   a named line stopped or shrank  -> revenue lost per month
  margin     same revenue, worse margin rate -> margin lost per month
  account    broad decline, no single culprit -> revenue lost per month

Revenue at risk and margin at risk are reported side by side and NEVER
added together — margin is a slice of revenue, so summing them would double
count. Prioritisation compares each against its own baseline instead, and
takes the worse of the two: an account bleeding 40% of its gross margin at
flat revenue is not a small problem just because its revenue is intact.
"""

from __future__ import annotations


def _category_impact(evidence_pack: dict, attributed_categories: list[str]) -> list[dict]:
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
            "defected": change.get("defected", False),
            "category_baseline_revenue_share": baseline_share.get(category, 0.0),
        })
    return per_category


def _margin_impact(evidence_pack: dict) -> dict | None:
    """Margin lost per month to a falling margin RATE, holding revenue as it
    actually is.

    Sized as a counterfactual: what this month's revenue would have earned at
    the baseline margin rate, minus what it actually earned. That isolates
    the rate change from any volume change, so an account whose revenue also
    fell is not charged twice for the same rupees.
    """
    margin = evidence_pack.get("margin")
    profile = evidence_pack.get("margin_profile")
    if not margin or not profile or margin.get("status") != "erosion_detected":
        return None

    baseline_pct = profile.get("baseline_margin_pct")
    recent_margin = profile.get("recent_monthly_margin")
    overall = evidence_pack.get("overall_revenue", {})
    recent_revenue = overall.get("recent_monthly_rate")
    if baseline_pct is None or recent_margin is None or recent_revenue is None:
        return None

    counterfactual = recent_revenue * baseline_pct
    monthly_at_risk = round(counterfactual - recent_margin, 2)
    if monthly_at_risk <= 0:
        return None

    baseline_monthly_margin = profile.get("baseline_monthly_margin")
    return {
        "monthly_margin_at_risk": monthly_at_risk,
        "annualized_margin_at_risk": round(monthly_at_risk * 12, 2),
        "margin_pct_erosion_pp": margin.get("margin_pct_change_pp"),
        "baseline_margin_pct": baseline_pct,
        "recent_margin_pct": profile.get("recent_margin_pct"),
        "baseline_monthly_margin": baseline_monthly_margin,
        "recent_monthly_margin": recent_margin,
        "pct_of_baseline_monthly_margin_at_risk": (
            round(monthly_at_risk / baseline_monthly_margin, 4)
            if baseline_monthly_margin else None
        ),
        "basis": (
            "Current monthly revenue valued at the baseline margin rate, minus the margin "
            "actually earned. Isolates the rate change from any change in volume."
        ),
    }


def _account_level_impact(evidence_pack: dict, quantified_categories: int) -> dict | None:
    """Revenue lost per month on an account-wide decline that no single
    category accounts for.

    Only computed when the category attribution came up empty. If a named
    category already explains the loss, adding an account-level figure on top
    would count the same rupees twice — and the dataset's diffuse-decline
    archetype exists precisely to punish naming a scapegoat category, so the
    honest answer there is one whole-account number and no category list.
    """
    decline = evidence_pack.get("revenue_decline") or {}
    if quantified_categories > 0 or decline.get("status") != "material_decline":
        return None

    overall = evidence_pack.get("overall_revenue", {})
    baseline_rate = overall.get("baseline_monthly_median")
    recent_rate = overall.get("recent_monthly_rate")
    if baseline_rate is None or recent_rate is None:
        return None

    monthly_at_risk = round(baseline_rate - recent_rate, 2)
    if monthly_at_risk <= 0:
        return None

    return {
        "monthly_revenue_at_risk": monthly_at_risk,
        "annualized_run_rate_loss": round(monthly_at_risk * 12, 2),
        "baseline_monthly_revenue": baseline_rate,
        "recent_monthly_revenue": recent_rate,
        "basis": (
            "Whole-account decline: baseline monthly revenue minus the current monthly rate. "
            "Not attributed to any category because no single category explains it."
        ),
    }


def compute_impact(evidence_pack: dict, verdict) -> dict:
    """Size the leak. `verdict` is the Stage 4 verdict dict; a bare list of
    attributed category names is also accepted for callers that only have
    that much."""
    if isinstance(verdict, dict):
        attributed_categories = verdict.get("attributed_categories") or []
    else:
        attributed_categories = list(verdict or [])

    per_category = _category_impact(evidence_pack, attributed_categories)
    quantifiable = [c for c in per_category if c["quantifiable"]]

    total_monthly_at_risk = round(sum(c["monthly_revenue_at_risk"] for c in quantifiable), 2)
    total_historical_lost = round(sum(c["historical_value_lost_to_date"] for c in quantifiable), 2)

    account_level = _account_level_impact(evidence_pack, len(quantifiable))
    if account_level:
        total_monthly_at_risk = round(total_monthly_at_risk + account_level["monthly_revenue_at_risk"], 2)

    margin_impact = _margin_impact(evidence_pack)

    baseline_monthly_total = evidence_pack["overall_revenue"]["baseline_monthly_median"]
    pct_of_baseline_at_risk = (
        round(total_monthly_at_risk / baseline_monthly_total, 4)
        if baseline_monthly_total else None
    )
    pct_of_margin_at_risk = (
        margin_impact.get("pct_of_baseline_monthly_margin_at_risk") if margin_impact else None
    )

    # The headline severity is the WORSE of the two ratios, not their sum.
    # An account can be losing a fifth of its revenue, or a third of its
    # gross margin at untouched revenue; both are real, and whichever is
    # larger is what the account is actually losing.
    severity_candidates = [v for v in (pct_of_baseline_at_risk, pct_of_margin_at_risk) if v is not None]
    overall_severity = max(severity_candidates) if severity_candidates else None

    return {
        "per_category": per_category,
        "account_level_revenue_impact": account_level,
        "margin_impact": margin_impact,
        "total_monthly_revenue_at_risk": total_monthly_at_risk,
        "total_historical_value_lost": total_historical_lost,
        "pct_of_baseline_monthly_revenue_at_risk": pct_of_baseline_at_risk,
        "pct_of_baseline_monthly_margin_at_risk": pct_of_margin_at_risk,
        "overall_severity_pct_of_baseline": overall_severity,
    }
