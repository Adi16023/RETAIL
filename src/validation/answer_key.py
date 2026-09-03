"""
Scoring the agent's live output against the workbook's Answer Key.

`Quessathon_Revenue_Leakage_Dataset.xlsx` ships an "Answer Key" tab giving,
per account, the expected verdict ("FLAG" / "NO FLAG" / "DEFER"), the
expected confidence, the leak type, and what each account is designed to
test. This module compares a finished pipeline report against that row and
says whether it matched.

It lives OUTSIDE src/pipeline/ on purpose. The workbook is explicit that the
answer key must never be fed to the agent, so nothing the pipeline imports
can reach this code — it is only ever called after a report exists, by the
Streamlit validation view or a script.

The expected-verdict column is free text ("NO FLAG (temporary)", "FLAG (soft
/ early)", "DEFER -> human"), so it is normalised to one of three outcomes
before comparing. Everything after the outcome word is nuance for a human
reader, not something to match on.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ANSWER_KEY_PATH = REPO_ROOT / "data" / "meridian" / "answer_key.json"

# Pipeline verdict -> the answer key's three outcomes.
VERDICT_TO_OUTCOME = {
    "leakage_detected": "FLAG",
    "healthy": "NO FLAG",
    "insufficient_data": "DEFER",
}

# The key's expected_confidence column uses ranges ("Med-High", "Low-Med")
# and inconsistent case ("LOW"). Each maps to the set of pipeline confidence
# values that should count as agreeing with it.
CONFIDENCE_ALIASES = {
    "high": {"high"},
    "med-high": {"medium", "high"},
    "medium": {"medium"},
    "low-med": {"low", "medium"},
    "low": {"low"},
}


class AnswerKeyUnavailable(Exception):
    """Raised when the answer key file is missing — surfaced as a clear
    message rather than an empty scorecard that looks like a perfect score."""


def load_answer_key(path: Path | str = ANSWER_KEY_PATH) -> dict[str, dict]:
    path = Path(path)
    if not path.exists():
        raise AnswerKeyUnavailable(
            f"Answer key not found at {path}. Run `python scripts/prepare_dataset.py` "
            "to extract it from the workbook."
        )
    return {row["account_id"]: row for row in json.loads(path.read_text(encoding="utf-8"))}


def expected_outcome(row: dict) -> str:
    """Normalise the key's free-text expected_verdict to FLAG / NO FLAG / DEFER.

    Order matters: "NO FLAG" contains "FLAG", so it must be tested first.
    """
    text = str(row.get("expected_verdict", "")).strip().upper()
    if text.startswith("NO FLAG"):
        return "NO FLAG"
    if text.startswith("DEFER"):
        return "DEFER"
    if text.startswith("FLAG"):
        return "FLAG"
    return "UNKNOWN"


def actual_outcome(report: dict) -> str:
    """What the pipeline actually concluded, in the key's vocabulary.

    `defer` wins over the verdict enum: an agent that says "leakage_detected"
    but sets defer=true has declined to make the call, and deferring is the
    outcome that routes to a human.
    """
    if report.get("defer"):
        return "DEFER"
    return VERDICT_TO_OUTCOME.get(report.get("verdict"), "UNKNOWN")


def confidence_matches(row: dict, report: dict) -> bool | None:
    expected = str(row.get("expected_confidence", "")).strip().lower()
    accepted = CONFIDENCE_ALIASES.get(expected)
    if accepted is None:
        return None
    return str(report.get("confidence", "")).strip().lower() in accepted


def score_report(report: dict, row: dict) -> dict:
    """Compare one finished report against its answer-key row.

    `outcome_match` is the headline and the only pass/fail. Confidence and
    leak type are reported alongside as secondary agreement, not folded into
    the verdict: the key gives confidence as a range and its leak_type column
    is a narrative label, so treating either as a hard assertion would fail
    runs that got the actual call right.
    """
    expected = expected_outcome(row)
    actual = actual_outcome(report)

    return {
        "account_id": row.get("account_id"),
        "account_name": row.get("account_name"),
        "archetype": row.get("archetype"),
        "what_it_tests": row.get("what_it_tests"),
        "what_is_really_happening": row.get("what_is_really_happening"),
        "expected_outcome": expected,
        "expected_verdict_text": row.get("expected_verdict"),
        "expected_confidence": row.get("expected_confidence"),
        "expected_leak_type": row.get("leak_type"),
        "actual_outcome": actual,
        "actual_verdict": report.get("verdict"),
        "actual_confidence": report.get("confidence"),
        "actual_temporary_or_structural": report.get("temporary_or_structural"),
        "leak_dimensions": report.get("leak_dimensions", []),
        "attributed_categories": report.get("attributed_categories", []),
        "narrative": report.get("narrative"),
        "cited_evidence": report.get("cited_evidence", []),
        "priority": (report.get("prioritization") or {}).get("priority"),
        "outcome_match": expected == actual,
        "confidence_match": confidence_matches(row, report),
    }


def score_error(account_id: str, row: dict, error: str) -> dict:
    """A run that never produced a report. Recorded as a non-match with the
    error attached rather than skipped, so a failed run can never quietly
    improve the accuracy figure."""
    return {
        "account_id": account_id,
        "account_name": row.get("account_name"),
        "archetype": row.get("archetype"),
        "what_it_tests": row.get("what_it_tests"),
        "what_is_really_happening": row.get("what_is_really_happening"),
        "expected_outcome": expected_outcome(row),
        "expected_verdict_text": row.get("expected_verdict"),
        "expected_confidence": row.get("expected_confidence"),
        "expected_leak_type": row.get("leak_type"),
        "actual_outcome": "ERROR",
        "actual_verdict": None,
        "actual_confidence": None,
        "actual_temporary_or_structural": None,
        "leak_dimensions": [],
        "attributed_categories": [],
        "narrative": None,
        "cited_evidence": [],
        "priority": None,
        "outcome_match": False,
        "confidence_match": None,
        "error": error,
    }


def summarize(results: list[dict]) -> dict:
    """Aggregate a scorecard.

    Accuracy is broken out by expected outcome as well as overall, because a
    single percentage hides the failure that matters most here: an agent that
    flags everything scores well on the FLAG accounts and badly on the rest,
    and the two are indistinguishable in one number.
    """
    total = len(results)
    matched = sum(1 for r in results if r["outcome_match"])
    errors = [r for r in results if r["actual_outcome"] == "ERROR"]

    by_expected: dict[str, dict] = {}
    for result in results:
        bucket = by_expected.setdefault(
            result["expected_outcome"], {"total": 0, "matched": 0}
        )
        bucket["total"] += 1
        bucket["matched"] += int(result["outcome_match"])
    for bucket in by_expected.values():
        bucket["accuracy"] = round(bucket["matched"] / bucket["total"], 4) if bucket["total"] else None

    scored_confidence = [r for r in results if r["confidence_match"] is not None]

    return {
        "total": total,
        "matched": matched,
        "accuracy": round(matched / total, 4) if total else None,
        "by_expected_outcome": by_expected,
        "confidence_agreement": (
            round(sum(1 for r in scored_confidence if r["confidence_match"]) / len(scored_confidence), 4)
            if scored_confidence else None
        ),
        "errors": len(errors),
        "mismatches": [
            f"{r['account_id']}: expected {r['expected_outcome']}, got {r['actual_outcome']}"
            for r in results if not r["outcome_match"]
        ],
    }
