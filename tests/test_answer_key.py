"""
Tests for scoring the agent against the workbook's Answer Key.

Two things are checked here, and it matters that they are separate:

1. The COMPARISON logic (src/validation/answer_key.py) — that the key's free
   text is normalised correctly, that a deferred verdict scores as DEFER, and
   that a failed run counts as a miss rather than vanishing from the
   denominator. This is pure logic and is fully tested.

2. The end-to-end SCORING PATH — that a report coming out of the real
   pipeline can be scored against a real key row. Run with a ScriptedClient
   standing in for Stage 4, exactly as elsewhere in this suite.

What is deliberately NOT tested here is whether the live model gets the
answers right. That needs an API key and a real run; use
`python scripts/validate_answer_key.py` or the Streamlit "Answer Key
validation" mode. A green suite here means the scorecard is trustworthy,
not that the score is good.
"""

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pipeline.orchestrate import run_for_account
from validation.answer_key import (
    AnswerKeyUnavailable,
    actual_outcome,
    confidence_matches,
    expected_outcome,
    load_answer_key,
    score_error,
    score_report,
    summarize,
)

from fakes import ScriptedClient, message, tool_use_block

TRANSACTIONS = REPO_ROOT / "data" / "meridian" / "transactions.csv"


@pytest.fixture(scope="module")
def key():
    return load_answer_key()


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


# --- The key itself --------------------------------------------------------

def test_answer_key_loads_and_covers_the_whole_book(key):
    assert len(key) == 18
    assert set(key) == {f"ACC-{n}" for n in range(101, 119)}
    for row in key.values():
        assert row["expected_verdict"]
        assert row["what_it_tests"], "each row states what the account is designed to test"


def test_missing_answer_key_raises_a_clear_error(tmp_path):
    """An empty scorecard would look like a perfect score, so a missing key
    must fail loudly and say how to regenerate it."""
    with pytest.raises(AnswerKeyUnavailable, match="prepare_dataset"):
        load_answer_key(tmp_path / "nope.json")


def test_every_expected_verdict_normalises_to_a_known_outcome(key):
    """The column is free text ("NO FLAG (temporary)", "DEFER -> human").
    If a row ever normalised to UNKNOWN it could never be matched, and the
    account would be permanently scored as a miss for the wrong reason."""
    outcomes = {account_id: expected_outcome(row) for account_id, row in key.items()}
    assert "UNKNOWN" not in outcomes.values()
    assert set(outcomes.values()) == {"FLAG", "NO FLAG", "DEFER"}


def test_the_book_is_mostly_not_leaks(key):
    """The workbook's own claim about this dataset: only a minority of
    accounts are genuine leaks, so "always flag" must score badly. If this
    ever stops holding, the validation view is measuring something else."""
    outcomes = [expected_outcome(row) for row in key.values()]
    assert outcomes.count("FLAG") == 6
    assert outcomes.count("NO FLAG") == 11
    assert outcomes.count("DEFER") == 1


# --- Normalising the two sides --------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("FLAG", "FLAG"),
    ("FLAG (soft / early)", "FLAG"),
    ("NO FLAG", "NO FLAG"),
    ("NO FLAG (temporary)", "NO FLAG"),
    ("NO FLAG (resolved)", "NO FLAG"),
    ("NO FLAG / opportunity", "NO FLAG"),
    ("NO FLAG or LOW watch", "NO FLAG"),
    ("DEFER -> human", "DEFER"),
    ("something else entirely", "UNKNOWN"),
])
def test_expected_verdict_text_normalisation(text, expected):
    """"NO FLAG" contains "FLAG", so the order of these checks is the whole
    correctness of the scorecard — getting it wrong would silently invert
    every negative case."""
    assert expected_outcome({"expected_verdict": text}) == expected


@pytest.mark.parametrize("report,expected", [
    ({"verdict": "leakage_detected", "defer": False}, "FLAG"),
    ({"verdict": "healthy", "defer": False}, "NO FLAG"),
    ({"verdict": "insufficient_data", "defer": True}, "DEFER"),
    ({"verdict": "insufficient_data", "defer": False}, "DEFER"),
    # defer wins over the verdict enum: an agent that names a leak but
    # declines to stand behind it has deferred, whatever the enum says.
    ({"verdict": "leakage_detected", "defer": True}, "DEFER"),
    ({"verdict": "healthy", "defer": True}, "DEFER"),
    ({"verdict": None, "defer": False}, "UNKNOWN"),
])
def test_actual_outcome_normalisation(report, expected):
    assert actual_outcome(report) == expected


@pytest.mark.parametrize("expected_conf,actual_conf,matches", [
    ("High", "high", True),
    ("High", "medium", False),
    ("Med-High", "medium", True),
    ("Med-High", "high", True),
    ("Med-High", "low", False),
    ("Low-Med", "low", True),
    ("LOW", "low", True),
    ("Medium", "medium", True),
])
def test_confidence_ranges_are_honoured(expected_conf, actual_conf, matches):
    """The key gives confidence as ranges ("Med-High"). Treating those as a
    single value would fail runs that got the call right."""
    row = {"expected_confidence": expected_conf}
    assert confidence_matches(row, {"confidence": actual_conf}) is matches


def test_unrecognised_expected_confidence_is_unscored_not_failed():
    assert confidence_matches({"expected_confidence": "?"}, {"confidence": "high"}) is None


# --- Scoring one account ---------------------------------------------------

def test_score_report_matches_when_the_outcome_agrees(key):
    row = key["ACC-101"]  # expected: FLAG, High
    report = {
        "verdict": "leakage_detected", "defer": False, "confidence": "high",
        "temporary_or_structural": "structural", "leak_dimensions": ["margin", "tier_mix"],
        "attributed_categories": [], "narrative": "n", "cited_evidence": ["f"],
        "prioritization": {"priority": "High"},
    }
    result = score_report(report, row)

    assert result["outcome_match"] is True
    assert result["confidence_match"] is True
    assert result["expected_outcome"] == "FLAG"
    assert result["actual_outcome"] == "FLAG"
    # The key's own description of the account travels with the result, so a
    # mismatch can be read without cross-referencing the workbook.
    assert result["what_it_tests"] == row["what_it_tests"]


def test_score_report_records_a_miss_when_a_healthy_account_is_flagged(key):
    """The false positive is the failure mode this dataset is built around,
    so it must score as a miss rather than as partial credit."""
    row = key["ACC-111"]  # premiumisation — expected NO FLAG
    report = {
        "verdict": "leakage_detected", "defer": False, "confidence": "high",
        "temporary_or_structural": "structural", "leak_dimensions": ["tier_mix"],
        "attributed_categories": [], "narrative": "n", "cited_evidence": [],
        "prioritization": {"priority": "High"},
    }
    result = score_report(report, row)

    assert result["outcome_match"] is False
    assert result["expected_outcome"] == "NO FLAG"
    assert result["actual_outcome"] == "FLAG"


def test_forcing_a_verdict_on_the_thin_history_account_is_a_miss(key):
    """ACC-109 is the rubric's heaviest case: deferring is the right answer
    and any confident call is wrong, even a plausible-sounding one."""
    row = key["ACC-109"]
    forced = {
        "verdict": "leakage_detected", "defer": False, "confidence": "high",
        "temporary_or_structural": "structural", "leak_dimensions": ["revenue"],
        "attributed_categories": [], "narrative": "n", "cited_evidence": [],
        "prioritization": {"priority": "High"},
    }
    deferred = {
        "verdict": "insufficient_data", "defer": True, "confidence": "low",
        "temporary_or_structural": "not_applicable", "leak_dimensions": [],
        "attributed_categories": [], "narrative": "n", "cited_evidence": [],
        "prioritization": {"priority": "Deferred"},
    }

    assert score_report(forced, row)["outcome_match"] is False
    assert score_report(deferred, row)["outcome_match"] is True
    assert score_report(deferred, row)["confidence_match"] is True


def test_a_failed_run_counts_as_a_miss_not_a_gap(key):
    """A crashed account must not drop out of the denominator — that would
    make a broken run look more accurate than a working one."""
    result = score_error("ACC-104", key["ACC-104"], "RateLimitError")

    assert result["outcome_match"] is False
    assert result["actual_outcome"] == "ERROR"
    assert result["error"] == "RateLimitError"
    assert result["expected_outcome"] == "FLAG"


# --- Aggregating ------------------------------------------------------------

def test_summary_breaks_accuracy_down_by_expected_outcome(key):
    """A single accuracy number cannot tell a discriminating agent from one
    that flags everything. The by-outcome split is what exposes it."""
    always_flag = [
        score_report(
            {"verdict": "leakage_detected", "defer": False, "confidence": "high",
             "temporary_or_structural": "structural", "leak_dimensions": [],
             "attributed_categories": [], "narrative": "n", "cited_evidence": [],
             "prioritization": {"priority": "High"}},
            row,
        )
        for row in key.values()
    ]
    summary = summarize(always_flag)

    assert summary["total"] == 18
    assert summary["by_expected_outcome"]["FLAG"]["accuracy"] == 1.0
    assert summary["by_expected_outcome"]["NO FLAG"]["accuracy"] == 0.0
    assert summary["by_expected_outcome"]["DEFER"]["accuracy"] == 0.0
    assert summary["accuracy"] == round(6 / 18, 4)
    assert len(summary["mismatches"]) == 12


def test_summary_counts_errors_and_still_reports_them_as_misses(key):
    results = [
        score_error("ACC-101", key["ACC-101"], "boom"),
        score_report(
            {"verdict": "healthy", "defer": False, "confidence": "high",
             "temporary_or_structural": "not_applicable", "leak_dimensions": [],
             "attributed_categories": [], "narrative": "n", "cited_evidence": [],
             "prioritization": {"priority": "None"}},
            key["ACC-102"],
        ),
    ]
    summary = summarize(results)

    assert summary["total"] == 2
    assert summary["matched"] == 1
    assert summary["errors"] == 1
    assert summary["accuracy"] == 0.5


# --- End to end through the real pipeline ----------------------------------

@pytest.mark.parametrize("account_id,verdict,should_match", [
    ("ACC-101", a_verdict(verdict="leakage_detected", temporary_or_structural="structural",
                          leak_dimensions=["margin", "tier_mix"]), True),
    ("ACC-102", a_verdict(), True),
    ("ACC-109", a_verdict(verdict="insufficient_data", confidence="low", defer=True,
                          data_needed_if_deferring=["12 months of history"]), True),
    ("ACC-111", a_verdict(verdict="leakage_detected", temporary_or_structural="structural"), False),
])
def test_a_real_pipeline_report_can_be_scored_against_the_key(key, account_id, verdict, should_match):
    """Closes the loop: a report produced by all seven stages, scored against
    a real key row. Guards the field names the scorer reads — if the report
    shape ever drifts, this fails rather than silently scoring None."""
    report = run_for_account(submit(verdict), str(TRANSACTIONS), account_id)
    result = score_report(report, key[account_id])

    assert result["outcome_match"] is should_match
    assert result["account_id"] == account_id
    assert result["account_name"] == key[account_id]["account_name"]
    assert result["actual_outcome"] != "UNKNOWN"
    assert result["priority"] is not None
    json.dumps(result)  # the scorecard is downloadable JSON


def test_a_perfect_run_scores_100_percent(key):
    """The scorecard must be able to reach 100%. Built by feeding each account
    the verdict its key row calls for, which also proves every expected
    outcome is reachable through the real pipeline."""
    outcome_to_verdict = {
        "FLAG": a_verdict(verdict="leakage_detected", temporary_or_structural="structural",
                          leak_dimensions=["revenue"]),
        "NO FLAG": a_verdict(),
        "DEFER": a_verdict(verdict="insufficient_data", confidence="low", defer=True,
                           data_needed_if_deferring=["12 months of history"]),
    }

    results = []
    for account_id, row in key.items():
        verdict = outcome_to_verdict[expected_outcome(row)]
        report = run_for_account(submit(verdict), str(TRANSACTIONS), account_id)
        results.append(score_report(report, row))

    summary = summarize(results)
    assert summary["accuracy"] == 1.0
    assert summary["mismatches"] == []
    assert summary["errors"] == 0
