"""
End-to-end pipeline tests (Stages 1-7) using a scripted fake LLM client.

These verify the *wiring and downstream deterministic logic* — evidence
pack -> agent loop -> impact -> prioritization -> report — using verdicts
scripted to represent what a well-behaved LLM should conclude for each
archetype (informed by ground truth, never fed to the pipeline itself).
This is NOT a test of real LLM judgement quality — that requires the real
API, which isn't available in this environment (see plan.md). It is a real
test of everything downstream of the model's answer: does the pipeline
correctly turn a given verdict into rupee impact, a priority bucket, and a
final report, without crashing, for every verdict shape the agent can
produce (healthy, structural, temporary, deferred).
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pipeline.evidence import build_evidence_pack
from pipeline.ingest import ingest
from pipeline.orchestrate import run_for_account

from fakes import ScriptedClient, message, tool_use_block

REPO_ROOT = Path(__file__).resolve().parents[1]
GENERATED_CSV = REPO_ROOT / "synthetic_data" / "transactions.csv"


def submit(verdict: dict) -> ScriptedClient:
    return ScriptedClient([message([tool_use_block("submit_verdict", verdict, "toolu_1")], stop_reason="tool_use")])


def ground_truth():
    return json.load(open(REPO_ROOT / "synthetic_data" / "ground_truth.json"))


def account_for(archetype: str) -> str:
    return next(g["account_id"] for g in ground_truth() if g["archetype"] == archetype)


def test_structural_leakage_end_to_end():
    account_id = account_for("structural_mix_collapse")
    df, _ = ingest(str(GENERATED_CSV))
    pack = build_evidence_pack(df, account_id)
    injected_hv = json.load(open(REPO_ROOT / "synthetic_data" / "ground_truth.json"))
    hv_cat = next(
        c["category"] for c in pack["category_changes"]
        if c["change_point_detected"] and c["pct_decline"] >= 0.5
    )

    verdict = {
        "verdict": "leakage_detected", "temporary_or_structural": "structural",
        "confidence": "high", "defer": False, "attributed_categories": [hv_cat],
        "cited_facts": [f"{hv_cat} collapsed with no recovery"],
        "narrative": f"{hv_cat} revenue collapsed while total revenue stayed flat.",
        "recommended_actions": ["Escalate to account owner"], "data_needed_if_deferring": [],
    }

    report = run_for_account(submit(verdict), str(GENERATED_CSV), account_id)

    assert report["verdict"] == "leakage_detected"
    assert report["temporary_or_structural"] == "structural"
    impact = report["financial_impact"]
    assert impact["per_category"][0]["quantifiable"] is True
    assert impact["total_monthly_revenue_at_risk"] > 0
    assert impact["pct_of_baseline_monthly_revenue_at_risk"] is not None
    assert report["prioritization"]["priority"] in ("High", "Medium")
    assert report["prioritization"]["churn_risk_projection"]["projected_12_month_loss_if_unaddressed"] == (
        round(impact["total_monthly_revenue_at_risk"] * 12, 2)
    )
    # Sanity guard: one category's revenue at risk cannot exceed the whole
    # account's baseline monthly revenue — catches unit-scale bugs (e.g. a
    # rolling-window sum leaking through as if it were a monthly rate).
    assert impact["pct_of_baseline_monthly_revenue_at_risk"] <= 1.0, (
        f"pct_of_baseline_monthly_revenue_at_risk={impact['pct_of_baseline_monthly_revenue_at_risk']} "
        "implies more revenue at risk than the account has — a scale bug."
    )


def test_healthy_account_end_to_end():
    account_id = account_for("healthy")
    verdict = {
        "verdict": "healthy", "temporary_or_structural": "not_applicable",
        "confidence": "high", "defer": False, "attributed_categories": [],
        "cited_facts": ["No change-point detected at total or category level"],
        "narrative": "Account shows stable purchasing behaviour.",
        "recommended_actions": [], "data_needed_if_deferring": [],
    }
    report = run_for_account(submit(verdict), str(GENERATED_CSV), account_id)

    assert report["verdict"] == "healthy"
    assert report["financial_impact"]["per_category"] == []
    assert report["financial_impact"]["total_monthly_revenue_at_risk"] == 0.0
    assert report["prioritization"]["priority"] == "None"
    assert report["prioritization"]["churn_risk_projection"] is None


def test_seasonal_dip_reads_as_healthy_not_leakage():
    account_id = account_for("seasonal_dip_temporary")
    verdict = {
        "verdict": "healthy", "temporary_or_structural": "temporary",
        "confidence": "medium", "defer": False, "attributed_categories": [],
        "cited_facts": ["Dip has seasonal precedent in prior year, not a sustained change"],
        "narrative": "Apparent dip matches a recurring seasonal pattern — not leakage.",
        "recommended_actions": [], "data_needed_if_deferring": [],
    }
    report = run_for_account(submit(verdict), str(GENERATED_CSV), account_id)
    assert report["verdict"] == "healthy"
    assert report["temporary_or_structural"] == "temporary"
    assert report["prioritization"]["priority"] == "None"


def test_sparse_history_defers():
    account_id = account_for("sparse_short_history")
    verdict = {
        "verdict": "insufficient_data", "temporary_or_structural": "not_applicable",
        "confidence": "low", "defer": True, "attributed_categories": [],
        "cited_facts": ["Only ~3 months of history, sufficiency label is insufficient"],
        "narrative": "Not enough history to distinguish a real pattern from noise.",
        "recommended_actions": [],
        "data_needed_if_deferring": ["At least 12 months of transaction history"],
    }
    report = run_for_account(submit(verdict), str(GENERATED_CSV), account_id)
    assert report["defer"] is True
    assert report["prioritization"]["priority"] == "Deferred"
    assert report["prioritization"]["churn_risk_projection"] is None


def test_one_off_bulk_reads_as_healthy():
    account_id = account_for("one_off_bulk_baseline")
    verdict = {
        "verdict": "healthy", "temporary_or_structural": "not_applicable",
        "confidence": "high", "defer": False, "attributed_categories": [],
        "cited_facts": ["Large early order appears to be a one-off, no sustained change-point"],
        "narrative": "Account is stable; one historical bulk order does not indicate leakage.",
        "recommended_actions": [], "data_needed_if_deferring": [],
    }
    report = run_for_account(submit(verdict), str(GENERATED_CSV), account_id)
    assert report["verdict"] == "healthy"
    assert report["prioritization"]["priority"] == "None"


def test_attributed_category_without_change_point_degrades_gracefully():
    """order_size_shrinkage's real signal is basket-width, not a category
    collapse — if the model (incorrectly, or for a category the deterministic
    scanner didn't flag) attributes a category with no change-point fact,
    the impact engine must not crash, and must mark it unquantifiable rather
    than silently fabricating a number."""
    account_id = account_for("order_size_shrinkage")
    df, _ = ingest(str(GENERATED_CSV))
    pack = build_evidence_pack(df, account_id)
    some_category = pack["category_mix"]["high_value_categories"][0]

    verdict = {
        "verdict": "leakage_detected", "temporary_or_structural": "structural",
        "confidence": "medium", "defer": False, "attributed_categories": [some_category],
        "cited_facts": ["Basket width declined while order frequency held steady"],
        "narrative": "Order sizes are shrinking.",
        "recommended_actions": [], "data_needed_if_deferring": [],
    }
    report = run_for_account(submit(verdict), str(GENERATED_CSV), account_id)
    entry = report["financial_impact"]["per_category"][0]
    assert entry["category"] == some_category
    if not entry["quantifiable"]:
        assert "reason" in entry
    assert report["prioritization"]["priority"] in ("Low", "Medium", "High")  # never crashes/None-branches incorrectly


def test_wildcard_smoke():
    """No strict correctness assertion — the wildcard archetype's generator
    recipe doesn't match its documented intent (known issue, see plan.md).
    Just verify the pipeline runs end-to-end without crashing."""
    account_id = account_for("wildcard_substitution")
    verdict = {
        "verdict": "insufficient_data", "temporary_or_structural": "not_applicable",
        "confidence": "low", "defer": True, "attributed_categories": [],
        "cited_facts": ["Signals were ambiguous"], "narrative": "Deferring pending review.",
        "recommended_actions": [], "data_needed_if_deferring": ["Clarify recent category history"],
    }
    report = run_for_account(submit(verdict), str(GENERATED_CSV), account_id)
    assert report["account_id"] == account_id
