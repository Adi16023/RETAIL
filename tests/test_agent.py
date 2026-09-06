import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pipeline.agent import AgentError, SUBMIT_VERDICT_TOOL, investigate
from pipeline.evidence import build_evidence_pack
from pipeline.ingest import ingest

from datasets import DEFECTED_ACCOUNT, DEFECTED_CATEGORY, MERIDIAN_CSV
from fakes import ScriptedClient, message, text_block, tool_use_block


def a_verdict(**overrides):
    v = {
        "verdict": "leakage_detected",
        "temporary_or_structural": "structural",
        "confidence": "high",
        "defer": False,
        "leak_dimensions": ["category_mix"],
        "attributed_categories": [DEFECTED_CATEGORY],
        "cited_facts": [f"{DEFECTED_CATEGORY}: defected, 10 months at zero, recovered=false"],
        "narrative": f"{DEFECTED_CATEGORY} stopped completely while other lines continued.",
        "recommended_actions": [f"Reach out to the account owner about {DEFECTED_CATEGORY}."],
        "data_needed_if_deferring": [],
        "model_opinion_response": "",
    }
    v.update(overrides)
    return v


def test_tool_markup_leaked_into_a_string_field_is_stripped():
    """Seen on 2 of 18 live Sonnet verdicts: the model closed the narrative
    in its own tool syntax and wrote the next parameter after it, all inside
    the string. The text before the tag is the narrative and is kept
    exactly; everything from the tag on is not."""
    leaked = a_verdict(
        narrative="This account looks healthy overall: margin only slipped from 28.68% to 28.35%."
                  '</narrative>\n<parameter name="recommended_actions">[]',
        cited_facts=["Revenue is ₹207,204 a month versus a ₹216,359 baseline.</cited_facts>"],
    )
    client = ScriptedClient([
        message([tool_use_block("submit_verdict", leaked, "toolu_a")], stop_reason="tool_use"),
    ])
    df, _ = ingest(str(MERIDIAN_CSV))
    result = investigate(client, df, DEFECTED_ACCOUNT, build_evidence_pack(df, DEFECTED_ACCOUNT))

    assert result["narrative"] == (
        "This account looks healthy overall: margin only slipped from 28.68% to 28.35%."
    )
    assert result["cited_facts"] == ["Revenue is ₹207,204 a month versus a ₹216,359 baseline."]
    # a clean verdict passes through untouched, including its < and > free prose
    assert {k: v for k, v in result.items() if k not in ("narrative", "cited_facts")} == {
        k: v for k, v in leaked.items() if k not in ("narrative", "cited_facts")
    }


def test_happy_path_one_drilldown_then_submit():
    verdict = a_verdict()
    client = ScriptedClient([
        message(
            [text_block("Let me check the seasonal pattern."),
             tool_use_block("get_category_seasonal_breakdown", {"category": DEFECTED_CATEGORY}, "toolu_a")],
            stop_reason="tool_use",
        ),
        message(
            [tool_use_block("submit_verdict", verdict, "toolu_b")],
            stop_reason="tool_use",
        ),
    ])

    df, _ = ingest(str(MERIDIAN_CSV))
    pack = build_evidence_pack(df, DEFECTED_ACCOUNT)
    result = investigate(client, df, DEFECTED_ACCOUNT, pack)

    assert result == verdict
    assert client.call_count == 2
    # second call's messages must include the tool_result for the first call's tool_use
    second_call_messages = client.calls[1]["messages"]
    tool_result_msg = second_call_messages[-1]
    assert tool_result_msg["role"] == "user"
    assert tool_result_msg["content"][0]["tool_use_id"] == "toolu_a"
    payload = json.loads(tool_result_msg["content"][0]["content"])
    assert "4" in payload or "8" in payload or "12" in payload  # some calendar month key present


def test_submit_verdict_tool_schema_matches_helper_verdicts():
    required = set(SUBMIT_VERDICT_TOOL["input_schema"]["required"])
    assert required == set(a_verdict().keys())


def test_unknown_tool_name_returns_error_result_not_a_crash():
    client = ScriptedClient([
        message([tool_use_block("not_a_real_tool", {}, "toolu_x")], stop_reason="tool_use"),
        message([tool_use_block("submit_verdict", a_verdict(), "toolu_y")], stop_reason="tool_use"),
    ])
    df, _ = ingest(str(MERIDIAN_CSV))
    pack = build_evidence_pack(df, DEFECTED_ACCOUNT)
    result = investigate(client, df, DEFECTED_ACCOUNT, pack)
    assert result == a_verdict()

    tool_result_msg = client.calls[1]["messages"][-1]
    assert tool_result_msg["content"][0]["is_error"] is True


def test_raises_when_model_stops_without_submitting():
    client = ScriptedClient([
        message([text_block("I'm not sure, here's my analysis in prose.")], stop_reason="end_turn"),
    ])
    df, _ = ingest(str(MERIDIAN_CSV))
    pack = build_evidence_pack(df, DEFECTED_ACCOUNT)
    with pytest.raises(AgentError, match="without calling submit_verdict"):
        investigate(client, df, DEFECTED_ACCOUNT, pack)


def test_reply_cut_off_by_output_limit_is_nudged_to_submit():
    """A reply that hits max_tokens with no tool call is not a failed
    verdict: the truncated prose goes back to the model with a request for
    the submit call. The output ceiling cannot simply be raised — Groq
    bills the requested ceiling against its per-minute cap."""
    from pipeline.agent import TRUNCATION_NUDGE

    client = ScriptedClient([
        message([text_block("Let me walk through every dimension in detail...")],
                stop_reason="max_tokens"),
        message([tool_use_block("submit_verdict", a_verdict(), "toolu_s")], stop_reason="tool_use"),
    ])
    df, _ = ingest(str(MERIDIAN_CSV))
    pack = build_evidence_pack(df, DEFECTED_ACCOUNT)
    assert investigate(client, df, DEFECTED_ACCOUNT, pack) == a_verdict()
    assert client.call_count == 2

    second_call_messages = client.calls[1]["messages"]
    assert second_call_messages[-2]["role"] == "assistant"
    assert second_call_messages[-1] == {"role": "user", "content": TRUNCATION_NUDGE}


def test_submit_call_cut_off_by_output_limit_is_not_accepted():
    """A submit_verdict whose input was truncated mid-way is a wrong
    answer, not a verdict. It is discarded (it cannot be replayed without
    a tool_result) and the model is asked to call again."""
    from pipeline.agent import TRUNCATION_NUDGE

    partial = {"verdict": "leakage_detected", "confidence": "high"}
    client = ScriptedClient([
        message([tool_use_block("submit_verdict", partial, "toolu_cut")], stop_reason="max_tokens"),
        message([tool_use_block("submit_verdict", a_verdict(), "toolu_ok")], stop_reason="tool_use"),
    ])
    df, _ = ingest(str(MERIDIAN_CSV))
    pack = build_evidence_pack(df, DEFECTED_ACCOUNT)
    assert investigate(client, df, DEFECTED_ACCOUNT, pack) == a_verdict()

    second_call_messages = client.calls[1]["messages"]
    assert second_call_messages[-1] == {"role": "user", "content": TRUNCATION_NUDGE}
    assert all(m["role"] == "user" for m in second_call_messages)


def test_submit_call_missing_fields_is_sent_back_as_a_tool_error():
    """Seen once live: a long reply lost its structure and the call came
    back without a narrative. That must be answered, not crashed on."""
    incomplete = {k: v for k, v in a_verdict().items() if k != "narrative"}
    client = ScriptedClient([
        message([tool_use_block("submit_verdict", incomplete, "toolu_bad")], stop_reason="tool_use"),
        message([tool_use_block("submit_verdict", a_verdict(), "toolu_ok")], stop_reason="tool_use"),
    ])
    df, _ = ingest(str(MERIDIAN_CSV))
    pack = build_evidence_pack(df, DEFECTED_ACCOUNT)
    assert investigate(client, df, DEFECTED_ACCOUNT, pack) == a_verdict()

    tool_result = client.calls[1]["messages"][-1]["content"][0]
    assert tool_result["tool_use_id"] == "toolu_bad"
    assert tool_result["is_error"] is True
    assert "narrative" in tool_result["content"]


def test_raises_when_max_iterations_exceeded():
    endless_drilldown = message(
        [tool_use_block("get_category_monthly_series", {"category": DEFECTED_CATEGORY}, "toolu_z")],
        stop_reason="tool_use",
    )
    client = ScriptedClient([endless_drilldown] * 6)
    df, _ = ingest(str(MERIDIAN_CSV))
    pack = build_evidence_pack(df, DEFECTED_ACCOUNT)
    with pytest.raises(AgentError, match="Exceeded max_iterations"):
        investigate(client, df, DEFECTED_ACCOUNT, pack, max_iterations=6)
    assert client.call_count == 6


def test_drilldown_tools_return_real_data_from_the_pipeline():
    """The mock only stands in for the model — the tool executor underneath
    must still hit the real deterministic analytics functions."""
    client = ScriptedClient([
        message(
            [tool_use_block("get_category_monthly_series", {"category": DEFECTED_CATEGORY}, "toolu_c")],
            stop_reason="tool_use",
        ),
        message([tool_use_block("submit_verdict", a_verdict(), "toolu_d")], stop_reason="tool_use"),
    ])
    df, _ = ingest(str(MERIDIAN_CSV))
    pack = build_evidence_pack(df, DEFECTED_ACCOUNT)
    investigate(client, df, DEFECTED_ACCOUNT, pack)

    tool_result = json.loads(client.calls[1]["messages"][-1]["content"][0]["content"])
    # the real series should have 20 months and end at 0 (post-collapse)
    assert len(tool_result) >= 15
    last_month = sorted(tool_result.keys())[-1]
    assert tool_result[last_month] == 0.0
