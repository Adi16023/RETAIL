"""
Cross-account comparison (pipeline/compare.py).

The load-bearing tests here are the ones about what reaches the prompt.
The comparison's entire claim is that it re-reads analysis the earlier
stages already produced — so if a threshold, a presentation-only series or
a figure nobody computed can get into the digest, the feature is no longer
what it says it is, and no amount of prompt wording fixes that.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pipeline.compare import (
    MAX_ACCOUNTS,
    MIN_ACCOUNTS,
    SUBMIT_COMPARISON_TOOL,
    ComparisonError,
    build_comparison_digest,
    compare_accounts,
)
from pipeline.ingest import ingest

from datasets import FLAGSHIP_LEAK, HEALTHY_CONTROL, MERIDIAN_CSV, THIN_HISTORY
from fakes import ScriptedClient, message, text_block, tool_use_block

SELECTION = [FLAGSHIP_LEAK, HEALTHY_CONTROL, THIN_HISTORY]


@pytest.fixture(scope="module")
def df():
    frame, _ = ingest(str(MERIDIAN_CSV))
    return frame


@pytest.fixture(scope="module")
def digest(df):
    return build_comparison_digest(df, SELECTION)


def a_comparison(**overrides):
    result = {
        "headline": "Two accounts are stable and one cannot be judged yet.",
        "groups": [
            {
                "story": "margin eroding under deeper discounting",
                "account_ids": [FLAGSHIP_LEAK],
                "what_they_share": "Selling the same goods at a steadily deeper discount.",
                "reads_as": "concern",
            },
            {
                "story": "too new to judge",
                "account_ids": [THIN_HISTORY],
                "what_they_share": "Not enough trading history to establish what normal is.",
                "reads_as": "cannot_judge_yet",
            },
        ],
        "standouts": [
            {"account_id": FLAGSHIP_LEAK, "why_it_stands_out": "Revenue flat while margin falls."}
        ],
        "per_account": [
            {"account_id": account_id, "business_read": "A one-line business read."}
            for account_id in SELECTION
        ],
        "where_to_look_first": [
            {
                "account_id": FLAGSHIP_LEAK,
                "reason": "Losing margin without losing revenue.",
                "suggested_action": "Review the discount authority on this account.",
            }
        ],
    }
    result.update(overrides)
    return result


# --- what the model is allowed to see --------------------------------------

def test_digest_carries_every_selected_account(digest):
    assert [a["account_id"] for a in digest["accounts"]] == sorted(SELECTION)


def test_digest_never_leaks_presentation_keys(digest):
    """The `_presentation` block is report-only, exactly as for Stage 4."""
    serialized = json.dumps(digest)
    assert "_presentation" not in serialized
    assert '"_' not in serialized


def test_digest_never_leaks_thresholds(digest):
    """`method` is the threshold a call was judged against — reviewer
    material, not business POV. Its absence is what keeps the comparison a
    business reading rather than a second statistical opinion."""
    for account in digest["accounts"]:
        for event in account["what_changed"]:
            assert "method" not in event
    assert "threshold" not in json.dumps(digest).lower()


def test_digest_events_are_business_statements(digest):
    """Every event carries the fields a business reader needs and no others."""
    for account in digest["accounts"]:
        for event in account["what_changed"]:
            assert set(event) == {"when", "headline", "detail", "reads_as"}


def test_digest_keeps_only_what_moved(digest):
    """Checks that came back normal are dropped: across many accounts they
    are the same reassuring lines repeated, and they crowd out the findings."""
    leak = next(a for a in digest["accounts"] if a["account_id"] == FLAGSHIP_LEAK)
    assert leak["what_changed"], "an account with known findings must carry events"
    assert all(e["reads_as"] != "checked" for e in leak["what_changed"])


def test_digest_marks_the_unjudgeable_account(digest):
    """A thin-history account must be identifiable as such, so the model can
    put it aside rather than rank it as healthy."""
    thin = next(a for a in digest["accounts"] if a["account_id"] == THIN_HISTORY)
    healthy = next(a for a in digest["accounts"] if a["account_id"] == HEALTHY_CONTROL)
    assert thin["history_is_sufficient_to_judge"] is False
    assert healthy["history_is_sufficient_to_judge"] is True


def test_digest_carries_business_context(digest):
    for account in digest["accounts"]:
        assert account["account_name"]
        assert "region" in account
        assert "account_manager" in account
        assert account["months_of_history"] is not None


# --- selection guardrails --------------------------------------------------

def test_one_account_is_not_a_comparison(df):
    with pytest.raises(ValueError, match="at least"):
        build_comparison_digest(df, [FLAGSHIP_LEAK])


def test_duplicate_selection_is_deduplicated_before_the_minimum(df):
    with pytest.raises(ValueError, match="at least"):
        build_comparison_digest(df, [FLAGSHIP_LEAK, FLAGSHIP_LEAK])


def test_too_many_accounts_is_refused_not_truncated(df):
    """Silently dropping accounts would answer a different question than the
    one the user asked."""
    too_many = [f"ACC-{i}" for i in range(MAX_ACCOUNTS + 1)]
    with pytest.raises(ValueError, match="at most"):
        build_comparison_digest(df, too_many)


def test_minimum_is_below_maximum():
    assert MIN_ACCOUNTS < MAX_ACCOUNTS


# --- the model call --------------------------------------------------------

def test_returns_the_submitted_comparison(digest):
    expected = a_comparison()
    client = ScriptedClient([
        message([tool_use_block("submit_comparison", expected, "toolu_a")], stop_reason="tool_use")
    ])
    assert compare_accounts(client, digest, model="test-model") == expected


def test_is_exactly_one_call_with_no_drilldown_tools(digest):
    """One call over many accounts, and no route back into the raw data:
    drilling into a single account belongs to Stage 4, and a second route
    could reach a figure that account's own verdict never saw."""
    client = ScriptedClient([
        message([tool_use_block("submit_comparison", a_comparison())], stop_reason="tool_use")
    ])
    compare_accounts(client, digest, model="test-model")

    assert client.call_count == 1
    tool_names = [t["name"] for t in client.calls[0]["tools"]]
    assert tool_names == ["submit_comparison"]


def test_prompt_payload_is_the_digest_and_nothing_else(digest):
    client = ScriptedClient([
        message([tool_use_block("submit_comparison", a_comparison())], stop_reason="tool_use")
    ])
    compare_accounts(client, digest, model="test-model")

    sent = client.calls[0]["messages"]
    assert len(sent) == 1
    assert json.loads(sent[0]["content"]) == digest


def test_system_prompt_forbids_inventing_numbers(digest):
    client = ScriptedClient([
        message([tool_use_block("submit_comparison", a_comparison())], stop_reason="tool_use")
    ])
    compare_accounts(client, digest, model="test-model")
    assert "never calculate" in client.calls[0]["system"].lower()


def test_text_only_reply_raises(digest):
    """A prose answer is unrenderable — fail loudly rather than show nothing."""
    client = ScriptedClient([
        message([text_block("Here is my comparison in prose.")], stop_reason="end_turn")
    ])
    with pytest.raises(ComparisonError, match="without calling submit_comparison"):
        compare_accounts(client, digest, model="test-model")


def test_wrong_tool_raises(digest):
    client = ScriptedClient([
        message([tool_use_block("submit_verdict", {"verdict": "healthy"})], stop_reason="tool_use")
    ])
    with pytest.raises(ComparisonError):
        compare_accounts(client, digest, model="test-model")


# --- the output contract ---------------------------------------------------

def test_tool_schema_requires_every_rendered_field():
    """The renderer reads these keys unconditionally."""
    schema = SUBMIT_COMPARISON_TOOL["input_schema"]
    assert set(schema["required"]) == {
        "headline", "groups", "standouts", "per_account", "where_to_look_first",
    }
    assert schema["additionalProperties"] is False


def test_group_reads_as_covers_the_unjudgeable_case():
    """Without `cannot_judge_yet` a thin-history account has to be filed as
    healthy or as a concern, and both are wrong."""
    groups = SUBMIT_COMPARISON_TOOL["input_schema"]["properties"]["groups"]
    assert set(groups["items"]["properties"]["reads_as"]["enum"]) == {
        "concern", "reassuring", "cannot_judge_yet", "context",
    }
