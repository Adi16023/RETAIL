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
