"""
Deterministic-layer regression suite: for each shape in the reference
dataset, the evidence pack must contain the facts needed to reach the right
verdict.

Stage 4 can only reason over what Stages 1-3 hand it. If the pack does not
distinguish a rising high-tier share from a falling one, no prompt can stop
the agent flagging an account that is simply trading up; if it does not
carry the prior-year echo, no prompt can make the agent cite it. So this is
the real regression protection for the deterministic layer — it fails
loudly and locally, where a prompt-level failure would be silent and
expensive.

Ground truth (the workbook's Answer Key) is used here to decide what to
assert; it is never fed to the pipeline. Whether the live model draws the
right conclusion from these facts is a separate question, measured by
scripts/validate_answer_key.py.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pipeline.evidence import build_evidence_pack
from pipeline.ingest import ingest

from datasets import MERIDIAN_CSV, thin_transactions

# The six accounts the answer key says to flag, and the eleven it clears.
# ACC-109 is neither: it is the deferral case.
FLAG_ACCOUNTS = ["ACC-101", "ACC-104", "ACC-106", "ACC-107", "ACC-108", "ACC-112"]
NO_FLAG_ACCOUNTS = [
    "ACC-102", "ACC-103", "ACC-105", "ACC-110", "ACC-111",
    "ACC-113", "ACC-114", "ACC-115", "ACC-116", "ACC-117", "ACC-118",
]

# A "material" status is one that should make an agent consider flagging.
MATERIAL_SIGNALS = {
    "revenue_decline": ("material_decline",),
    "margin": ("erosion_detected",),
    "discount": ("creep_detected",),
    "tier_mix": ("downgrade_detected",),
    "order_pattern": ("fragmentation_detected",),
}


@pytest.fixture(scope="module")
def df():
    clean, _ = ingest(str(MERIDIAN_CSV))
    return clean


@pytest.fixture(scope="module")
def packs(df):
    return {account_id: build_evidence_pack(df, account_id)
            for account_id in sorted(df["account_id"].unique())}


def material_signals(pack: dict) -> list[str]:
    found = [
        name for name, statuses in MATERIAL_SIGNALS.items()
        if (pack.get(name) or {}).get("status") in statuses
    ]
    if any(c["defected"] for c in pack["category_changes"]):
        found.append("category_defection")
    return found


# --- The two leaks that revenue cannot see ---------------------------------

def test_hidden_mix_collapse_shows_in_margin_and_tier_but_not_revenue(packs):
    """The disclosed jury scenario: flat topline, value rotating out of the
    high-value tier. Both halves are asserted, because the finding is only
    interesting if revenue really does look fine."""
    pack = packs["ACC-101"]

    assert pack["revenue_decline"]["status"] == "stable"
    assert pack["overall_revenue"]["change_point"] is None

    assert pack["margin"]["status"] == "erosion_detected"
    assert pack["margin"]["margin_pct_change_pp"] < -10
    assert pack["tier_mix"]["status"] == "downgrade_detected"
    assert pack["tier_mix"]["high_tier_share_change_pp"] < -30

    baseline = pack["tier_mix"]["baseline_share_by_tier"]
    recent = pack["tier_mix"]["recent_share_by_tier"]
    assert baseline["High"] > 0.6 and recent["High"] < 0.3
    assert recent["Low"] > baseline["Low"], "low-value volume should have backfilled"


def test_discount_creep_shows_with_volume_and_mix_untouched(packs):
    """The margin-vs-revenue test. Revenue-only analysis misses it entirely,
    so the discount and margin signals must both fire while everything else
    stays quiet."""
    pack = packs["ACC-107"]

    assert pack["revenue_decline"]["status"] == "stable"
    assert pack["tier_mix"]["status"] == "stable"
    assert pack["discount"]["status"] == "creep_detected"
    assert pack["discount"]["baseline_avg_discount_pct"] < 0.15
    assert pack["discount"]["recent_avg_discount_pct"] > 0.20
    assert pack["margin"]["status"] == "erosion_detected"


def test_the_two_flat_revenue_leaks_are_told_apart(packs):
    """Both hold revenue flat and lose ~11-12pp of margin. What separates
    them is where the margin went — and the pack has to make that separable,
    or the two get the same wrong recommendation."""
    collapse, creep = packs["ACC-101"], packs["ACC-107"]

    assert collapse["tier_mix"]["status"] == "downgrade_detected"
    assert collapse["discount"]["status"] == "stable"

    assert creep["discount"]["status"] == "creep_detected"
    assert creep["tier_mix"]["status"] == "stable"


# --- Direction, not magnitude ----------------------------------------------

def test_premiumisation_is_not_read_as_a_downgrade(packs):
    """The same flat revenue and the same large mix shift as the flagship
    leak, in the opposite direction. A detector that reported magnitude
    would make these two accounts identical."""
    pack = packs["ACC-111"]

    assert pack["tier_mix"]["status"] == "premiumisation_detected"
    assert pack["tier_mix"]["high_tier_share_change_pp"] > 0
    assert pack["margin"]["margin_pct_change_pp"] > 0
    assert pack["discount"]["status"] != "creep_detected"


@pytest.mark.parametrize("account_id", ["ACC-105", "ACC-111", "ACC-115"])
def test_growth_is_recognised_as_growth(packs, account_id):
    """Change-point detection only ever looks for declines, so without an
    explicit trend direction a growing account and a flat one both simply
    have no change-point."""
    pack = packs[account_id]
    assert pack["revenue_trend"]["direction"] == "growing"
    assert pack["revenue_decline"]["status"] == "growth"
    assert pack["overall_revenue"]["change_point"] is None


# --- Ruling out the innocent explanations ----------------------------------

def test_seasonal_dip_is_caught_by_the_prior_year_echo(packs):
    """A naive reading flags the recent dip as structural. The echo has to
    be confirmed AND name the prior-year months, since an uncitable finding
    is not usable in a verdict."""
    import pandas as pd

    pack = packs["ACC-103"]
    seasonality = pack["seasonality"]

    assert seasonality["status"] == "confirmed"
    assert len(seasonality["echo_months"]) >= 2
    for dip, echo in zip(seasonality["dip_months"], seasonality["echo_months"]):
        assert pd.Period(dip, freq="M") - 12 == pd.Period(echo, freq="M")

    assert all(episode["recovered"] for episode in pack["dip_episodes"])


def test_recovered_dip_is_marked_recovered_not_ongoing(packs):
    """A dip that ended a year ago and one still running look identical in
    any baseline-vs-recent comparison, and are opposite verdicts."""
    pack = packs["ACC-110"]

    episodes = pack["dip_episodes"]
    assert len(episodes) == 1
    assert episodes[0]["recovered"] is True
    assert episodes[0]["is_ongoing"] is False
    assert episodes[0]["recovered_from_month"] is not None
    assert pack["overall_revenue"]["change_point"] is None


def test_defection_is_not_excused_as_seasonal(packs):
    """The defected line stopped in a month it had traded happily the year
    before. A structural collapse that begins in a historically quiet month
    is what this check exists to prevent being written off."""
    pack = packs["ACC-106"]
    entry = next(c for c in pack["category_changes"] if c["category"] == "Diagnostic Equipment")

    assert entry["defected"] is True
    assert entry["consecutive_months_at_zero"] >= 9
    assert entry["prior_year_echo"]["status"] == "not_present"
    assert entry["prior_year_echo"]["echo_months"] == []
    # Only that one line stopped — that is what makes it partial defection
    # rather than an account-wide collapse.
    assert all(not c["defected"] for c in pack["category_changes"]
               if c["category"] != "Diagnostic Equipment")


def test_bulk_month_is_flagged_as_an_outlier_not_a_trend_break(packs):
    """Both directions matter: the spike must not read as growth, and the
    ordinary months after it must not read as a decline."""
    pack = packs["ACC-114"]

    outliers = pack["data_quality"]["outlier_months"]
    assert len(outliers) == 1
    assert outliers[0]["multiple_of_median_month"] >= 2.0

    assert pack["revenue_decline"]["status"] == "stable"
    assert pack["overall_revenue"]["change_point"] is None


def test_returns_are_netted_labelled_and_kept_out_of_order_behaviour(df, packs):
    """Credit notes arrive as standalone negative orders, which would
    otherwise fake a frequency rise and a basket-width collapse at once."""
    pack = packs["ACC-116"]
    returns = pack["data_quality"]["returns"]

    assert returns["return_lines"] == 3
    assert returns["net_return_value"] < 0
    assert returns["share_of_gross_revenue"] < 0.01

    assert pack["revenue_decline"]["status"] == "stable"
    assert pack["order_pattern"]["status"] == "stable"

    account = df[df["account_id"] == "ACC-116"]
    sales_orders = account[account["is_return"] != 1]["order_id"].nunique()
    assert account["order_id"].nunique() == sales_orders + 3
    counted = (pack["order_behavior"]["order_frequency_recent_per_month"]
               * pack["category_mix"]["recent_months"])
    assert counted < account["order_id"].nunique()


def test_skipped_month_is_reported_as_a_gap(packs):
    """The gap costs this account ~8% on a half-over-half comparison —
    exactly the range where an unexplained drift starts to look like a leak."""
    pack = packs["ACC-117"]
    gaps = pack["data_quality"]["gaps"]

    assert len(gaps["months_with_no_orders"]) == 1
    assert gaps["note"] is not None
    assert pack["revenue_decline"]["status"] != "material_decline"


def test_rep_change_is_surfaced_as_context_not_as_a_cause(packs):
    """A correlated red herring. It must be visible so the agent can mention
    it, but must never arrive inside the leak evidence."""
    pack = packs["ACC-108"]

    assert pack.get("account_manager_changed") is True
    assert isinstance(pack["account_manager"], list) and len(pack["account_manager"]) > 1
    for signal in ("revenue_decline", "margin", "discount", "tier_mix", "order_pattern"):
        assert "manager" not in json.dumps(pack[signal]).lower()


# --- The subtle and the borderline -----------------------------------------

def test_order_fragmentation_is_found_without_a_category_collapse(packs):
    """No category disappears and revenue is only mildly down, so this is
    findable only in the shape of the orders."""
    pack = packs["ACC-108"]

    assert pack["order_pattern"]["status"] == "fragmentation_detected"
    assert pack["order_pattern"]["order_frequency_pct_change"] > 0.30
    assert pack["order_pattern"]["basket_width_pct_change"] < -0.20
    assert not any(c["defected"] for c in pack["category_changes"])


def test_borderline_drift_stays_below_the_material_threshold(packs):
    """The hardest "leave it alone" call: mild_drift, not stable (that would
    be dishonest) and not material (that would be a false positive)."""
    pack = packs["ACC-118"]

    assert pack["revenue_decline"]["status"] == "mild_drift"
    assert pack["margin"]["status"] == "stable"
    assert pack["tier_mix"]["status"] == "stable"
    assert pack["order_pattern"]["status"] == "stable"


@pytest.mark.parametrize("account_id", ["ACC-102", "ACC-113"])
def test_stable_controls_produce_no_signal_at_all(packs, account_id):
    """Population realism: not every account is a story, and every detector
    has to agree that nothing is happening."""
    pack = packs[account_id]

    for dimension in ("revenue_decline", "margin", "discount", "tier_mix", "order_pattern"):
        assert pack[dimension]["status"] == "stable", dimension
    assert pack["overall_revenue"]["change_point"] is None
    assert pack["dip_episodes"] == []


def test_thin_history_reports_insufficiency_rather_than_numbers(packs):
    """The rubric's heaviest case. With no baseline window there is nothing
    to compare against, and every comparative detector must say so rather
    than return a figure that looks like a finding."""
    pack = packs["ACC-109"]

    assert pack["data_sufficiency"]["label"] == "insufficient"
    assert "history_too_short_for_baseline" in pack["data_sufficiency"]["flags"]
    for dimension in ("margin", "discount", "tier_mix", "order_pattern", "seasonality"):
        assert pack[dimension]["status"] == "insufficient_history", dimension


# --- Population-level discrimination ---------------------------------------

@pytest.mark.parametrize("account_id", FLAG_ACCOUNTS)
def test_every_flag_account_gives_stage_4_something_to_flag(packs, account_id):
    """The floor. A miss here is an evidence gap no prompt can close."""
    assert material_signals(packs[account_id]), f"{account_id} reaches Stage 4 with nothing"


@pytest.mark.parametrize("account_id", NO_FLAG_ACCOUNTS)
def test_no_cleared_account_trips_a_material_signal(packs, account_id):
    """The harder half. "Always flag" is wrong for most of this book, so a
    false positive here is a live demo failure."""
    signals = material_signals(packs[account_id])
    assert signals == [], f"{account_id} falsely tripped {signals}"


def test_the_book_splits_the_way_the_answer_key_says(packs):
    """The discrimination — not any single account — is what is being
    tested. Asserted over the whole population at once."""
    flagged = {a for a in packs if material_signals(packs[a])}
    deferred = {a for a in packs if packs[a]["data_sufficiency"]["label"] == "insufficient"}

    assert flagged == set(FLAG_ACCOUNTS)
    assert deferred == {"ACC-109"}
    assert not flagged & deferred
    assert len(flagged) < len(packs) / 2, "flagging most of the book is the known failure mode"


# --- Degraded input --------------------------------------------------------

def test_evidence_pack_narrows_honestly_on_a_thin_file():
    """On a file with no tier, discount or unit cost, the pack must drop
    those dimensions to None and say so in analysis_dimensions — never
    report them as stable."""
    clean, _ = ingest(thin_transactions())
    pack = build_evidence_pack(clean, "ACC-101")

    assert pack["analysis_dimensions"]["tier_mix"] is False
    assert pack["analysis_dimensions"]["discount"] is False
    assert pack["tier_mix"] is None
    assert pack["discount"] is None

    # Revenue and margin survive, so the flagship leak is still partly
    # visible — narrower analysis, not a broken one.
    assert pack["margin"]["status"] == "erosion_detected"
    assert pack["revenue_decline"]["status"] == "stable"
    json.dumps(pack)


def test_evidence_pack_survives_a_revenue_only_file():
    """The narrowest input the pipeline accepts. Four of the six dimensions
    go dark and the pack must still build."""
    clean, _ = ingest(thin_transactions(with_margin=False))
    pack = build_evidence_pack(clean, "ACC-101")

    assert pack["margin"] is None
    assert pack["analysis_dimensions"]["margin"] is False
    assert pack["revenue_decline"]["status"] == "stable"
    assert pack["category_changes"]
    json.dumps(pack)


def test_evidence_pack_is_json_serializable_for_every_account(packs):
    """The pack is serialized straight into the Stage 4 prompt — a numpy
    scalar leaking through breaks the live run, not a test."""
    for account_id, pack in packs.items():
        json.dumps(pack), account_id
