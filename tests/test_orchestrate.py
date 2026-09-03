"""
End-to-end pipeline tests (Stages 1-7) using a scripted fake LLM client.

These verify the *wiring and the downstream deterministic logic* — evidence
pack -> agent loop -> rupee impact -> prioritisation -> report — using
verdicts scripted to represent what a well-behaved model should conclude for
each shape in the reference dataset. This is NOT a test of model judgement;
that needs a live key and is measured by scripts/validate_answer_key.py.

It is a real test of everything downstream of the model's answer: given a
verdict, does the pipeline turn it into the right rupees, the right priority
bucket, and a complete report — for every verdict shape the agent can
produce (structural, margin-only, diffuse, temporary, healthy, deferred),
without crashing and without double-counting.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pipeline.evidence import build_evidence_pack
from pipeline.impact import compute_impact
from pipeline.ingest import ingest
from pipeline.orchestrate import run_for_account
from pipeline.prioritize import prioritize

from datasets import MERIDIAN_CSV, thin_transactions
from fakes import ScriptedClient, message, tool_use_block


def a_verdict(**overrides) -> dict:
    verdict = {
        "verdict": "healthy", "temporary_or_structural": "not_applicable",
        "confidence": "high", "defer": False, "leak_dimensions": [],
        "attributed_categories": [], "cited_facts": ["placeholder"],
        "narrative": "placeholder", "recommended_actions": [],
        "data_needed_if_deferring": [],
    }
    verdict.update(overrides)
    return verdict


def submit(verdict: dict) -> ScriptedClient:
    return ScriptedClient([
        message([tool_use_block("submit_verdict", verdict, "toolu_1")], stop_reason="tool_use")
    ])


@pytest.fixture(scope="module")
def df():
    clean, _ = ingest(str(MERIDIAN_CSV))
    return clean


# --- A named category is responsible ---------------------------------------

def test_category_defection_end_to_end(df):
    """A single line lost: the impact engine sizes it from that category's
    own change-point, and nothing else."""
    verdict = a_verdict(
        verdict="leakage_detected", temporary_or_structural="structural",
        leak_dimensions=["category_mix"], attributed_categories=["Diagnostic Equipment"],
        cited_facts=["Diagnostic Equipment: defected, 10 months at zero"],
        narrative="A high-margin line stopped completely.",
        recommended_actions=["Win back the defected line"],
    )
    report = run_for_account(submit(verdict), str(MERIDIAN_CSV), "ACC-106")

    assert report["verdict"] == "leakage_detected"
    assert report["leak_dimensions"] == ["category_mix"]
    assert report["account_name"] == "Kingsway Distributors"

    impact = report["financial_impact"]
    entry = impact["per_category"][0]
    assert entry["quantifiable"] is True
    assert entry["monthly_revenue_at_risk"] > 0
    assert entry["defected"] is True
    assert impact["total_monthly_revenue_at_risk"] > 0
    assert report["prioritization"]["priority"] in ("High", "Medium")
    assert report["prioritization"]["churn_risk_projection"]["projected_12_month_loss_if_unaddressed"] == (
        round(impact["total_monthly_revenue_at_risk"] * 12, 2)
    )

    # Scale guard: one category cannot put more at risk than the whole
    # account earns. Catches a rolling-window sum leaking through as if it
    # were a monthly rate.
    assert impact["pct_of_baseline_monthly_revenue_at_risk"] <= 1.0


def test_account_level_impact_does_not_double_count_a_named_category(df):
    """If a category already explains the loss, the whole-account figure must
    stay out — otherwise the same rupees are counted twice."""
    pack = build_evidence_pack(df, "ACC-106")
    verdict = a_verdict(
        verdict="leakage_detected", temporary_or_structural="structural",
        leak_dimensions=["category_mix"], attributed_categories=["Diagnostic Equipment"],
    )
    impact = compute_impact(pack, verdict)

    assert impact["per_category"][0]["quantifiable"] is True
    assert impact["account_level_revenue_impact"] is None


# --- The leak is in margin, not revenue ------------------------------------

def test_margin_only_leak_is_sized_and_ranked_despite_flat_revenue(df):
    """The case that breaks a revenue-only pipeline. Revenue at risk is
    genuinely zero, and the account must still rank High."""
    pack = build_evidence_pack(df, "ACC-101")
    verdict = a_verdict(
        verdict="leakage_detected", temporary_or_structural="structural",
        leak_dimensions=["margin", "tier_mix"],
        cited_facts=["Margin 32.2% -> 19.9% with revenue flat"],
    )
    impact = compute_impact(pack, verdict)
    priority = prioritize(impact, verdict)

    assert impact["total_monthly_revenue_at_risk"] == 0.0
    assert impact["margin_impact"]["monthly_margin_at_risk"] > 0
    assert impact["margin_impact"]["pct_of_baseline_monthly_margin_at_risk"] > 0.15
    assert impact["overall_severity_pct_of_baseline"] > 0.15
    assert priority["priority"] == "High"
    assert priority["churn_risk_projection"]["monthly_margin_loss"] > 0


def test_revenue_and_margin_at_risk_are_never_summed(df):
    """Margin is a slice of revenue. Severity takes the worse of the two
    ratios; adding them would overstate every mixed case."""
    pack = build_evidence_pack(df, "ACC-107")
    verdict = a_verdict(
        verdict="leakage_detected", temporary_or_structural="structural",
        leak_dimensions=["discount", "margin"],
    )
    impact = compute_impact(pack, verdict)

    revenue_pct = impact["pct_of_baseline_monthly_revenue_at_risk"]
    margin_pct = impact["pct_of_baseline_monthly_margin_at_risk"]
    assert impact["overall_severity_pct_of_baseline"] == max(
        v for v in (revenue_pct, margin_pct) if v is not None
    )


# --- Nobody in particular is responsible -----------------------------------

@pytest.mark.parametrize("account_id", ["ACC-104", "ACC-112"])
def test_diffuse_decline_is_sized_without_inventing_a_category(df, account_id):
    """Naming a scapegoat category for a broad decline sends the account team
    after the wrong thing — but reporting zero would let a real loss rank
    None. The whole-account figure is the honest middle."""
    pack = build_evidence_pack(df, account_id)
    verdict = a_verdict(
        verdict="leakage_detected", temporary_or_structural="structural",
        leak_dimensions=["revenue"], attributed_categories=[],
        cited_facts=["Material broad-based decline across all categories"],
    )
    impact = compute_impact(pack, verdict)
    priority = prioritize(impact, verdict)

    assert impact["per_category"] == []
    assert impact["account_level_revenue_impact"]["monthly_revenue_at_risk"] > 0
    assert impact["total_monthly_revenue_at_risk"] > 0
    assert priority["priority"] in ("High", "Medium")


def test_attributed_category_without_a_change_point_degrades_gracefully(df):
    """If the model attributes a category the deterministic scanner never
    flagged, the impact engine must mark it unquantifiable rather than
    crash or fabricate a number."""
    pack = build_evidence_pack(df, "ACC-108")
    stable_category = next(
        c["category"] for c in pack["category_changes"] if not c["change_point_detected"]
    )
    verdict = a_verdict(
        verdict="leakage_detected", temporary_or_structural="structural",
        confidence="medium", leak_dimensions=["order_pattern"],
        attributed_categories=[stable_category],
        cited_facts=["Order frequency up, basket width down"],
    )
    report = run_for_account(submit(verdict), str(MERIDIAN_CSV), "ACC-108")

    entry = report["financial_impact"]["per_category"][0]
    assert entry["category"] == stable_category
    assert entry["quantifiable"] is False
    assert "reason" in entry
    assert report["prioritization"]["priority"] in ("Low", "Medium", "High")


# --- The non-leak outcomes -------------------------------------------------

def test_healthy_account_end_to_end():
    report = run_for_account(submit(a_verdict()), str(MERIDIAN_CSV), "ACC-102")

    assert report["verdict"] == "healthy"
    assert report["financial_impact"]["per_category"] == []
    assert report["financial_impact"]["margin_impact"] is None
    assert report["financial_impact"]["total_monthly_revenue_at_risk"] == 0.0
    assert report["prioritization"]["priority"] == "None"
    assert report["prioritization"]["churn_risk_projection"] is None


def test_seasonal_dip_reads_as_healthy_not_leakage():
    """A temporary classification must still rank as no action — the
    'temporary' label is not a soft flag."""
    verdict = a_verdict(
        temporary_or_structural="temporary", confidence="medium",
        cited_facts=["Dip echoes the same calendar window a year earlier"],
    )
    report = run_for_account(submit(verdict), str(MERIDIAN_CSV), "ACC-103")

    assert report["verdict"] == "healthy"
    assert report["temporary_or_structural"] == "temporary"
    assert report["prioritization"]["priority"] == "None"


def test_premiumisation_is_not_charged_for_its_mix_shift():
    """A healthy verdict on an account with a large mix shift must produce
    no impact at all — the margin engine only ever fires on erosion."""
    report = run_for_account(submit(a_verdict()), str(MERIDIAN_CSV), "ACC-111")

    assert report["financial_impact"]["margin_impact"] is None
    assert report["financial_impact"]["overall_severity_pct_of_baseline"] in (None, 0.0)
    assert report["prioritization"]["priority"] == "None"


def test_thin_history_defers_rather_than_ranking_low():
    """Deferring is a distinct outcome that routes to a human, not a
    low-priority leak."""
    verdict = a_verdict(
        verdict="insufficient_data", confidence="low", defer=True,
        cited_facts=["Only 4.4 months of history; sufficiency label insufficient"],
        data_needed_if_deferring=["At least 12 months of order history"],
    )
    report = run_for_account(submit(verdict), str(MERIDIAN_CSV), "ACC-109")

    assert report["defer"] is True
    assert report["prioritization"]["priority"] == "Deferred"
    assert report["prioritization"]["churn_risk_projection"] is None
    assert report["data_needed_if_deferring"]


def test_low_confidence_downgrades_a_large_figure(df):
    """Acting on shaky evidence is itself a risk: a High-magnitude leak held
    with low confidence must rank Medium, not High."""
    pack = build_evidence_pack(df, "ACC-101")
    high = a_verdict(verdict="leakage_detected", temporary_or_structural="structural",
                     confidence="high", leak_dimensions=["margin"])
    low = a_verdict(verdict="leakage_detected", temporary_or_structural="structural",
                    confidence="low", leak_dimensions=["margin"])
    impact = compute_impact(pack, high)

    assert prioritize(impact, high)["priority"] == "High"
    assert prioritize(impact, low)["priority"] == "Medium"


# --- Coverage across the book and across input shapes ----------------------

@pytest.mark.parametrize("account_id", [f"ACC-{n}" for n in range(101, 119)])
def test_pipeline_runs_end_to_end_for_every_account(account_id):
    """Every account must survive all seven stages. The Reality Test runs
    live on stage, so a crash on any one shape is a demo failure."""
    report = run_for_account(submit(a_verdict()), str(MERIDIAN_CSV), account_id)

    assert report["account_id"] == account_id
    assert report["account_name"]
    assert report["prioritization"]["priority"] in ("High", "Medium", "Low", "None", "Deferred")
    json.dumps(report)


def test_pipeline_runs_end_to_end_on_a_thin_file():
    """A file with no tier, discount or unit cost must produce a complete
    report — narrower, but complete, and honest about what it could not
    look at."""
    report = run_for_account(submit(a_verdict()), thin_transactions(), "ACC-101")

    assert report["account_id"] == "ACC-101"
    assert report["analysis_dimensions"]["tier_mix"] is False
    assert report["analysis_dimensions"]["discount"] is False
    assert report["analysis_dimensions"]["margin"] is True
    json.dumps(report)


def test_pipeline_runs_end_to_end_on_a_revenue_only_file():
    report = run_for_account(submit(a_verdict()), thin_transactions(with_margin=False), "ACC-101")

    assert report["financial_impact"]["margin_impact"] is None
    assert report["analysis_dimensions"]["margin"] is False
    json.dumps(report)
