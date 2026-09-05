"""
Intervention pricing and the commercial recommendation (pipeline/decide.py).

The load-bearing test here is `test_full_restoration_reproduces_the_accounts_own_baseline_margin`.
The pricing model claims that changing price does not change cost, so the
whole price movement lands in margin. If that is true, then winding the
discount all the way back to where it started must reproduce the account's
own historical margin rate — a figure the calculator was never given. That
one assertion is what makes these numbers defensible rather than plausible,
and it would fail loudly if the arithmetic ever drifted.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pipeline.decide import (
    DecisionError,
    SUBMIT_DECISION_TOOL,
    build_decision_input,
    levers_available,
    price_options,
    recommend,
)
from pipeline.evidence import build_evidence_pack
from pipeline.impact import compute_impact
from pipeline.ingest import ingest
from pipeline.prioritize import prioritize
from pipeline.report import assemble_report

from datasets import (
    DEFECTED_ACCOUNT,
    DEFECTED_CATEGORY,
    FLAGSHIP_LEAK,
    HEALTHY_CONTROL,
    MERIDIAN_CSV,
    THIN_HISTORY,
)
from fakes import ScriptedClient, message, text_block, tool_use_block

# The dataset's discount-creep account: volume and mix steady, money leaving
# through the price. The only archetype this lever applies to.
DISCOUNT_CREEP = "ACC-107"


@pytest.fixture(scope="module")
def df():
    frame, _ = ingest(str(MERIDIAN_CSV))
    return frame


@pytest.fixture(scope="module")
def pack(df):
    return build_evidence_pack(df, DISCOUNT_CREEP)


@pytest.fixture(scope="module")
def options(pack):
    """The discount ladder specifically — ACC-107's only lever."""
    return [o for o in price_options(pack) if o["lever"] == "price_recovery"]


def a_decision(**overrides):
    result = {
        "action_type": "price_recovery",
        "recommended_option": "negotiate the discount back to 18.1%",
        "headline": "Reset the discount at renewal.",
        "rationale": "The money is leaving through the price, not through volume.",
        "why_not_more_aggressive": "A full reset asks for too much at once.",
        "why_not_less_aggressive": "A smaller move leaves most of the leak in place.",
        "talking_points": ["Discounting has deepened while order volume has not grown."],
        "check_before_acting": ["Confirm whether the current discount was contractually agreed."],
        "restores_account_to_healthy": False,
        "what_is_left_over": "A 2.3pp margin gap remains; the 75% option would close it.",
        "downside_if_wrong": "The buyer may push volume elsewhere.",
        "owner": "Account manager",
        "timing": "At the next renewal.",
        "how_we_will_know_it_worked": ["Average discount falls back within two billing cycles."],
        "review_in_months": 3,
    }
    result.update(overrides)
    return result


# --- the arithmetic --------------------------------------------------------

def test_full_restoration_reproduces_the_accounts_own_baseline_margin(pack, options):
    """The model's self-check: wind the discount back to where it started and
    the margin rate must land on the account's real historical baseline,
    which the calculator is never told."""
    full = next(o for o in options if o["restores_original_discount"])
    baseline_margin_pct = pack["margin"]["baseline_margin_pct"] * 100
    assert full["resulting_margin_rate_pct"] == pytest.approx(baseline_margin_pct, abs=0.5)


def test_full_restoration_targets_the_baseline_discount(pack, options):
    full = next(o for o in options if o["restores_original_discount"])
    assert full["discount_target_pct"] == pytest.approx(
        pack["discount"]["baseline_avg_discount_pct"] * 100, abs=0.1
    )


def test_recovering_more_of_the_creep_recovers_more_money(options):
    ordered = sorted(options, key=lambda o: o["share_recovered_pct"])
    recovered = [o["recovers_margin_per_month"] for o in ordered]
    assert recovered == sorted(recovered), "a deeper cut must recover strictly more"


def test_annual_figure_is_twelve_times_the_monthly_one(options):
    for option in options:
        assert option["recovers_margin_over_12_months"] == pytest.approx(
            option["recovers_margin_per_month"] * 12, rel=0.01
        )


def test_break_even_widens_as_the_recovery_deepens(options):
    """The counter-intuitive result the feature exists to surface: the more
    margin an intervention restores, the more volume the business could
    afford to lose before it stops being worth doing."""
    ordered = sorted(options, key=lambda o: o["share_recovered_pct"])
    tolerances = [o["volume_that_could_be_lost_before_this_stops_paying_pct"] for o in ordered]
    assert tolerances == sorted(tolerances)


def test_break_even_is_a_real_break_even(pack, options):
    """At the break-even volume, the intervention must leave the account
    exactly where it is today — no better, no worse."""
    revenue = pack["overall_revenue"]["recent_monthly_rate"]
    margin_now = revenue * pack["margin"]["recent_margin_pct"]
    for option in options:
        retained = 1 - option["volume_that_could_be_lost_before_this_stops_paying_pct"] / 100
        margin_after = (margin_now + option["recovers_margin_per_month"]) * retained
        assert margin_after == pytest.approx(margin_now, rel=0.02)


def test_reduction_is_precomputed_so_the_model_never_subtracts(pack, options):
    current = pack["discount"]["recent_avg_discount_pct"] * 100
    for option in options:
        assert option["reduction_from_today_pp"] == pytest.approx(
            current - option["discount_target_pct"], abs=0.15
        )


def test_every_option_improves_the_margin_rate(pack, options):
    current_rate = pack["margin"]["recent_margin_pct"] * 100
    assert options
    for option in options:
        assert option["resulting_margin_rate_pct"] > current_rate


# --- the other levers ------------------------------------------------------

def test_a_defected_category_is_priced_as_a_win_back(df):
    """The account whose Diagnostic Equipment stopped dead. Sized from what
    the line was worth before it stopped, not from a guess about winning it
    back."""
    options = [o for o in price_options(build_evidence_pack(df, DEFECTED_ACCOUNT))
               if o["lever"] == "win_back"]
    assert options
    full = max(options, key=lambda o: o["share_recovered_pct"])
    assert full["share_recovered_pct"] == 100
    assert DEFECTED_CATEGORY in full["categories"]
    assert full["recovers_revenue_per_month"] == full["lost_revenue_per_month"]
    assert full["months_already_gone"] >= 3
    assert full["value_lost_so_far"] == pytest.approx(
        full["lost_revenue_per_month"] * full["months_already_gone"], rel=0.01
    )


def test_win_back_concession_room_equals_the_margin_rate(df):
    """Discounting cuts price without cutting cost, so margin on recovered
    business runs out exactly when the concession reaches the margin rate."""
    pack = build_evidence_pack(df, DEFECTED_ACCOUNT)
    options = [o for o in price_options(pack) if o["lever"] == "win_back"]
    assert options[0]["max_price_concession_pct"] == pytest.approx(
        pack["margin"]["recent_margin_pct"] * 100, abs=0.1
    )


def test_win_back_at_half_the_concession_returns_half_the_margin(df):
    options = [o for o in price_options(build_evidence_pack(df, DEFECTED_ACCOUNT))
               if o["lever"] == "win_back"]
    for option in options:
        assert option["margin_if_won_back_at_half_that_concession"] == pytest.approx(
            option["recovers_margin_per_month"] / 2, rel=0.02
        )


def test_a_whole_account_decline_is_not_priced_twice(df):
    """A named line that stopped already explains the loss. Pricing a
    whole-account rebuild on top would sell the same rupees twice — the same
    double-count guard impact.py applies."""
    levers = levers_available(price_options(build_evidence_pack(df, DEFECTED_ACCOUNT)))
    assert "win_back" in levers
    assert "revenue_recovery" not in levers


def test_margin_erosion_from_mix_is_priced_as_mix_recovery(df):
    """ACC-101 holds revenue flat while value slides into cheaper lines."""
    options = price_options(build_evidence_pack(df, FLAGSHIP_LEAK))
    assert levers_available(options) == ["mix_recovery"]
    full = max(options, key=lambda o: o["share_recovered_pct"])
    assert full["resulting_margin_rate_pct"] == pytest.approx(full["margin_rate_before_pct"], abs=0.2)


def test_margin_recovery_is_not_priced_when_discount_already_is(df, pack):
    """Both would be closing the same margin gap."""
    assert levers_available(price_options(pack)) == ["price_recovery"]


def test_shrinking_baskets_are_never_priced(df):
    """Nothing has been lost yet, so any recovery figure would be invented —
    the obvious formula overstated ACC-108 by roughly ten times."""
    options = price_options(build_evidence_pack(df, "ACC-108"))
    assert options == []


def test_no_lever_claims_more_than_the_account_actually_lost(df):
    """A magnitude sanity check across the book: a recovery bigger than the
    decline it is recovering from is a double count. Excludes the discount
    lever, whose revenue figure is an uplift from repricing rather than
    recovery of anything lost."""
    for account_id in sorted(df["account_id"].unique()):
        pack = build_evidence_pack(df, account_id)
        options = [o for o in price_options(pack) if o["lever"] != "price_recovery"]
        if not options:
            continue
        revenue = pack["overall_revenue"]
        gap = (revenue["baseline_monthly_median"] or 0) - (revenue["recent_monthly_rate"] or 0)
        biggest = max(o["recovers_revenue_per_month"] for o in options)
        assert biggest <= max(gap, 0) * 1.05 + 1, account_id


# --- does the fix actually fix it? ----------------------------------------

def test_every_option_states_whether_it_restores_health(df):
    """An option that recovers real money can still leave the account
    reading as leaking. Every option must say which it is."""
    for account_id in sorted(df["account_id"].unique()):
        for option in price_options(build_evidence_pack(df, account_id)):
            assert "restores_account_to_healthy" in option, account_id


def test_partial_recovery_does_not_claim_to_restore_health(pack):
    """ACC-107 needs its margin rate back within 3pp of baseline. Closing
    half the discount gap does not get there, and must not say it does."""
    options = [o for o in price_options(pack) if o["lever"] == "price_recovery"]
    half = next(o for o in options if o["share_recovered_pct"] == 50)
    assert half["restores_account_to_healthy"] is False
    assert half["short_of_healthy_by_pp"] > 0


def test_full_recovery_restores_health(pack):
    options = [o for o in price_options(pack) if o["lever"] == "price_recovery"]
    full = next(o for o in options if o["share_recovered_pct"] == 100)
    assert full["restores_account_to_healthy"] is True
    assert full["short_of_healthy_by_pp"] == 0


def test_health_threshold_is_the_pipelines_own(pack):
    """Not a second opinion invented here — the same constant Stage 3 uses."""
    from pipeline.detect import MARGIN_EROSION_PP
    options = [o for o in price_options(pack) if o["lever"] == "price_recovery"]
    needed = pack["margin"]["baseline_margin_pct"] * 100 - MARGIN_EROSION_PP
    for option in options:
        assert option["restores_account_to_healthy"] == (
            option["resulting_margin_rate_pct"] >= needed - 0.05
        )


def test_winning_the_whole_line_back_restores_the_account(df):
    """Recovering all of a defected category closes the revenue gap it left."""
    options = [o for o in price_options(build_evidence_pack(df, DEFECTED_ACCOUNT))
               if o["lever"] == "win_back"]
    full = next(o for o in options if o["share_recovered_pct"] == 100)
    partial = next(o for o in options if o["share_recovered_pct"] == 50)
    assert full["restores_account_to_healthy"] is True
    assert partial["restores_account_to_healthy"] is False
    assert partial["still_short_per_month"] > 0


def test_the_model_must_report_whether_it_fixed_the_account():
    required = SUBMIT_DECISION_TOOL["input_schema"]["required"]
    assert "restores_account_to_healthy" in required
    assert "what_is_left_over" in required


# --- when the lever does not apply ----------------------------------------

def test_no_options_when_discount_did_not_creep(df):
    """Pricing a discount change for an account whose discount never moved
    would answer a question nobody asked."""
    assert price_options(build_evidence_pack(df, HEALTHY_CONTROL)) == []


def test_no_options_on_an_account_too_new_to_have_a_baseline(df):
    assert price_options(build_evidence_pack(df, THIN_HISTORY)) == []


def test_no_options_when_the_pack_is_empty():
    assert price_options({}) == []


def test_no_options_when_margin_is_absent(pack):
    """Without margin there is no cost to hold constant, so the whole model
    is unavailable — it must decline rather than guess."""
    crippled = dict(pack)
    crippled["margin"] = {"status": "insufficient_history", "recent_margin_pct": None}
    assert price_options(crippled) == []


# --- what the adviser is shown --------------------------------------------

@pytest.fixture(scope="module")
def report(df, pack):
    verdict = {
        "verdict": "leakage_detected", "temporary_or_structural": "structural",
        "leak_dimensions": ["discount", "margin"], "confidence": "high", "defer": False,
        "narrative": "Money is leaving through the price.",
        "attributed_categories": [], "cited_facts": ["Average discount rose 12.7% -> 23.5%"],
        "recommended_actions": ["Review the discount authority."], "data_needed_if_deferring": [],
    }
    impact = compute_impact(pack, verdict)
    return assemble_report(DISCOUNT_CREEP, pack, verdict, impact, prioritize(impact, verdict))


def test_decision_input_never_carries_the_raw_evidence_pack(report, options):
    """The adviser reads business statements, not detector vocabulary — and
    presentation-only keys never travel anywhere near a prompt."""
    serialized = json.dumps(build_decision_input(report, options))
    assert "_presentation" not in serialized
    assert '"_' not in serialized
    assert "category_changes" not in serialized
    assert "monthly_series" not in serialized


def test_decision_input_carries_the_options_and_the_money(report, options):
    payload = build_decision_input(report, options)
    assert payload["priced_options"] == options
    assert payload["money"]["monthly_margin_at_risk"]
    assert payload["verdict"]["verdict"] == "leakage_detected"


def test_severity_reaches_the_model_already_in_percent(report, options):
    """A ratio would make the model multiply by 100 to quote it — correct,
    trivial, and still the arithmetic it was told not to do. Every quantity
    must arrive in the form it will be spoken in."""
    payload = build_decision_input(report, options)
    severity = payload["money"]["severity_pct_of_baseline"]
    expected = (report["financial_impact"]["overall_severity_pct_of_baseline"]) * 100
    assert severity == pytest.approx(expected, abs=0.1)
    assert severity > 1, "a value below 1 would be a ratio, not a percentage"


def test_decision_input_states_that_break_even_is_not_a_forecast(report, options):
    assert "not forecasts" in build_decision_input(report, options)["note"]


# --- the model call --------------------------------------------------------

def test_returns_the_submitted_decision(report, options):
    expected = a_decision()
    client = ScriptedClient([
        message([tool_use_block("submit_decision", expected)], stop_reason="tool_use")
    ])
    payload = build_decision_input(report, options)
    assert recommend(client, payload, model="test-model") == expected


def test_is_one_call_with_no_drilldown_tools(report, options):
    client = ScriptedClient([
        message([tool_use_block("submit_decision", a_decision())], stop_reason="tool_use")
    ])
    recommend(client, build_decision_input(report, options), model="test-model")
    assert client.call_count == 1
    assert [t["name"] for t in client.calls[0]["tools"]] == ["submit_decision"]


def test_system_prompt_forbids_arithmetic_and_forecasting(report, options):
    client = ScriptedClient([
        message([tool_use_block("submit_decision", a_decision())], stop_reason="tool_use")
    ])
    recommend(client, build_decision_input(report, options), model="test-model")
    system = client.calls[0]["system"]
    assert "Never calculate" in system
    assert "NEVER PREDICT HOW THE CUSTOMER WILL REACT" in system
    assert "TOLERANCES, not forecasts" in system


def test_text_only_reply_raises(report, options):
    client = ScriptedClient([
        message([text_block("I think you should lower the discount.")], stop_reason="end_turn")
    ])
    with pytest.raises(DecisionError, match="without calling submit_decision"):
        recommend(client, build_decision_input(report, options), model="test-model")


def test_wrong_tool_raises(report, options):
    client = ScriptedClient([
        message([tool_use_block("submit_verdict", {"verdict": "healthy"})], stop_reason="tool_use")
    ])
    with pytest.raises(DecisionError):
        recommend(client, build_decision_input(report, options), model="test-model")


# --- the output contract ---------------------------------------------------

def test_no_action_and_gather_data_are_available_verdicts():
    """Without these the adviser must recommend an intervention on every
    account, including the healthy ones and the ones nobody would call."""
    actions = SUBMIT_DECISION_TOOL["input_schema"]["properties"]["action_type"]["enum"]
    assert {"no_action", "gather_data", "expansion"} <= set(actions)


def test_rejected_alternatives_are_required():
    """A recommendation without them is an assertion, not advice."""
    required = SUBMIT_DECISION_TOOL["input_schema"]["required"]
    assert "why_not_more_aggressive" in required
    assert "why_not_less_aggressive" in required
    assert "check_before_acting" in required
    assert "how_we_will_know_it_worked" in required
