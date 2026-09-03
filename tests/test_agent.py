import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pipeline.agent import AgentError, SUBMIT_VERDICT_TOOL, investigate
from pipeline.evidence import build_evidence_pack
from pipeline.ingest import ingest

from fakes import ScriptedClient, message, text_block, tool_use_block

REPO_ROOT = Path(__file__).resolve().parents[1]
GENERATED_CSV = REPO_ROOT / "synthetic_data" / "transactions.csv"


def a_verdict(**overrides):
    v = {
        "verdict": "leakage_detected",
        "temporary_or_structural": "structural",
        "confidence": "high",
        "defer": False,
        "attributed_categories": ["Industrial Equipment"],
        "cited_facts": ["Industrial Equipment: change-point 2025-05, 100% decline, recovered=false"],
        "narrative": "Industrial Equipment collapsed while total revenue stayed flat.",
        "recommended_actions": ["Reach out to the account owner about Industrial Equipment."],
        "data_needed_if_deferring": [],
    }
    v.update(overrides)
    return v


def test_happy_path_one_drilldown_then_submit():
    verdict = a_verdict()
    client = ScriptedClient([
        message(
            [text_block("Let me check the seasonal pattern."),
             tool_use_block("get_category_seasonal_breakdown", {"category": "Industrial Equipment"}, "toolu_a")],
            stop_reason="tool_use",
        ),
        message(
            [tool_use_block("submit_verdict", verdict, "toolu_b")],
            stop_reason="tool_use",
        ),
    ])

    df, _ = ingest(str(GENERATED_CSV))
    pack = build_evidence_pack(df, "ACC-0001")
    result = investigate(client, df, "ACC-0001", pack)

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
    df, _ = ingest(str(GENERATED_CSV))
    pack = build_evidence_pack(df, "ACC-0001")
    result = investigate(client, df, "ACC-0001", pack)
    assert result == a_verdict()

    tool_result_msg = client.calls[1]["messages"][-1]
    assert tool_result_msg["content"][0]["is_error"] is True


def test_raises_when_model_stops_without_submitting():
    client = ScriptedClient([
        message([text_block("I'm not sure, here's my analysis in prose.")], stop_reason="end_turn"),
    ])
    df, _ = ingest(str(GENERATED_CSV))
    pack = build_evidence_pack(df, "ACC-0001")
    with pytest.raises(AgentError, match="without calling submit_verdict"):
        investigate(client, df, "ACC-0001", pack)


def test_raises_when_max_iterations_exceeded():
    endless_drilldown = message(
        [tool_use_block("get_category_monthly_series", {"category": "Industrial Equipment"}, "toolu_z")],
        stop_reason="tool_use",
    )
    client = ScriptedClient([endless_drilldown] * 6)
    df, _ = ingest(str(GENERATED_CSV))
    pack = build_evidence_pack(df, "ACC-0001")
    with pytest.raises(AgentError, match="Exceeded max_iterations"):
        investigate(client, df, "ACC-0001", pack, max_iterations=6)
    assert client.call_count == 6


def test_drilldown_tools_return_real_data_from_the_pipeline():
    """The mock only stands in for the model — the tool executor underneath
    must still hit the real deterministic analytics functions."""
    client = ScriptedClient([
        message(
            [tool_use_block("get_category_monthly_series", {"category": "Industrial Equipment"}, "toolu_c")],
            stop_reason="tool_use",
        ),
        message([tool_use_block("submit_verdict", a_verdict(), "toolu_d")], stop_reason="tool_use"),
    ])
    df, _ = ingest(str(GENERATED_CSV))
    pack = build_evidence_pack(df, "ACC-0001")
    investigate(client, df, "ACC-0001", pack)

    tool_result = json.loads(client.calls[1]["messages"][-1]["content"][0]["content"])
    # the real series should have 20 months and end at 0 (post-collapse)
    assert len(tool_result) >= 15
    last_month = sorted(tool_result.keys())[-1]
    assert tool_result[last_month] == 0.0
