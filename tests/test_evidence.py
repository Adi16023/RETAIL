"""
Phase 2 regression suite: for each scenario archetype, the evidence pack
must contain the facts needed to reach the right verdict — per plan.md's
Phase 2 checklist. Ground truth (ground_truth.json) is never fed into the
pipeline itself; it's only used here, in tests, to check the pipeline's
output against what was actually injected.

These are the facts Stage 4 (not yet built) will reason over, so this
suite is the real regression protection for the deterministic layer: if
these assertions hold, Stage 4 has what it needs; if they don't, no amount
of prompt engineering downstream can compensate for missing evidence.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pipeline.evidence import build_evidence_pack
from pipeline.ingest import ingest

REPO_ROOT = Path(__file__).resolve().parents[1]
GENERATED_CSV = REPO_ROOT / "synthetic_data" / "transactions.csv"
GROUND_TRUTH = REPO_ROOT / "synthetic_data" / "ground_truth.json"


@pytest.fixture(scope="module")
def df():
    clean, _ = ingest(str(GENERATED_CSV))
    return clean


@pytest.fixture(scope="module")
def ground_truth():
    return json.load(open(GROUND_TRUTH))


def accounts_for(ground_truth, archetype):
    return [g for g in ground_truth if g["archetype"] == archetype]


def test_structural_mix_collapse_detected(df, ground_truth):
    """The jury's known shape: total revenue roughly flat, injected
    high-value categories collapse and don't recover."""
    accounts = accounts_for(ground_truth, "structural_mix_collapse")
    assert len(accounts) >= 5

    flat_total_count = 0
    for g in accounts:
        pack = build_evidence_pack(df, g["account_id"])
        pct_change = pack["overall_revenue"]["pct_change"]
        if pct_change is not None and abs(pct_change) <= 0.30:
            flat_total_count += 1

        injected_hv = set(g["injected_high_value_categories"])
        collapsed = {
            c["category"] for c in pack["category_changes"]
            if c["change_point_detected"] and c["pct_decline"] >= 0.5
        }
        assert collapsed & injected_hv, (
            f"{g['account_id']}: expected at least one of {injected_hv} to show a "
            f">=50% decline, got category_changes={pack['category_changes']}"
        )

    # Allow one noisy outlier account, not a blanket pass requirement.
    assert flat_total_count >= 4, f"only {flat_total_count}/5 accounts stayed within +-30% total revenue"


def test_healthy_accounts_never_flag_a_change_point(df, ground_truth):
    """Core 'no false alarm' requirement — a healthy account must never
    get a structural change-point at the total-revenue level."""
    for g in accounts_for(ground_truth, "healthy"):
        pack = build_evidence_pack(df, g["account_id"])
        assert pack["overall_revenue"]["change_point"] is None, (
            f"{g['account_id']} (healthy) falsely triggered a total-revenue change-point"
        )


def test_seasonal_dip_does_not_trigger_total_change_point(df, ground_truth):
    """Temporary/seasonal variation must not be declared a structural
    total-revenue decline."""
    for g in accounts_for(ground_truth, "seasonal_dip_temporary"):
        pack = build_evidence_pack(df, g["account_id"])
        assert pack["overall_revenue"]["change_point"] is None, (
            f"{g['account_id']} (seasonal_dip_temporary) falsely triggered a total-revenue change-point"
        )


def test_one_off_bulk_order_does_not_manufacture_a_decline(df, ground_truth):
    """A single historical bulk order must not create a false *structural*
    verdict — this is what the median-based baseline is specifically for.

    The raw pct_change headline is deliberately not asserted tightly here:
    it's meant to reflect genuine recent volatility honestly (that's what
    monthly_revenue_coefficient_of_variation is for downstream), not to be
    suppressed into a narrow band. change_point is the fact that actually
    drives the verdict, and must not fire on a one-off historical event.
    """
    for g in accounts_for(ground_truth, "one_off_bulk_baseline"):
        pack = build_evidence_pack(df, g["account_id"])
        assert pack["overall_revenue"]["change_point"] is None, (
            f"{g['account_id']} (one_off_bulk_baseline) falsely triggered a total-revenue change-point"
        )


def test_sparse_short_history_flagged_insufficient(df, ground_truth):
    for g in accounts_for(ground_truth, "sparse_short_history"):
        pack = build_evidence_pack(df, g["account_id"])
        assert pack["data_sufficiency"]["label"] == "insufficient", (
            f"{g['account_id']}: sufficiency={pack['data_sufficiency']}"
        )


def test_order_size_shrinkage_shows_basket_width_signal(df, ground_truth):
    """Frequency should stay roughly intact while basket width shrinks —
    the defining, subtler signal for this archetype."""
    accounts = accounts_for(ground_truth, "order_size_shrinkage")
    shrinking = 0
    for g in accounts:
        pack = build_evidence_pack(df, g["account_id"])
        width_change = pack["order_behavior"]["basket_width_pct_change"]
        if width_change is not None and width_change < -0.15:
            shrinking += 1
    assert shrinking >= 3, f"only {shrinking}/{len(accounts)} accounts showed a basket-width decline"


def test_evidence_pack_is_json_serializable(df, ground_truth):
    """Everything downstream (LLM prompt, evidence pack JSON) depends on
    this — a numpy scalar leaking through would break serialization."""
    for g in ground_truth[:1] + [ground_truth[len(ground_truth) // 2]]:
        pack = build_evidence_pack(df, g["account_id"])
        json.dumps(pack)  # must not raise
