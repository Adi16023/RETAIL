"""
Tests for the temporary Groq shim (src/pipeline/groq_client.py). These
prove two things: the Anthropic<->Groq format conversion is correct in
isolation, and — more importantly — that agent.investigate() runs
UNCHANGED through the shim, which is the whole point of building it this
way (swap back to real Anthropic later with no pipeline changes).
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pipeline.agent import SUBMIT_VERDICT_TOOL, investigate
from pipeline.evidence import build_evidence_pack
from pipeline.groq_client import (
    DEFAULT_GROQ_MODEL,
    GROQ_MAX_OUTPUT_TOKENS,
    _GroqMessages,
    _anthropic_messages_to_groq,
    _anthropic_tool_to_groq,
    _groq_response_to_anthropic,
)
from pipeline.ingest import ingest

from datasets import DEFECTED_ACCOUNT, DEFECTED_CATEGORY, MERIDIAN_CSV


def groq_tool_call(call_id: str, name: str, arguments: dict):
    return SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=json.dumps(arguments)))


def groq_response(content=None, tool_calls=None, finish_reason="stop"):
    message = SimpleNamespace(content=content, tool_calls=tool_calls or [])
    return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason=finish_reason)])


class FakeGroqChatCompletions:
    def __init__(self, responses):
        self._responses = list(responses)
        self.call_count = 0
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        response = self._responses[self.call_count]
        self.call_count += 1
        return response


class FakeGroqClient:
    def __init__(self, responses):
        self.chat = SimpleNamespace(completions=FakeGroqChatCompletions(responses))


class GroqBackedTestClient:
    """Same shape as GroqShimClient but skips real groq.Groq() construction."""

    def __init__(self, fake_groq_client, model_override=None):
        self.messages = _GroqMessages(fake_groq_client, model_override)


def test_tool_conversion():
    groq_tool = _anthropic_tool_to_groq(SUBMIT_VERDICT_TOOL)
    assert groq_tool["type"] == "function"
    assert groq_tool["function"]["name"] == "submit_verdict"
    assert groq_tool["function"]["parameters"] == SUBMIT_VERDICT_TOOL["input_schema"]


def test_message_conversion_user_then_tool_use_then_tool_result():
    anthropic_messages = [
        {"role": "user", "content": '{"account_id": "ACC-1"}'},
        {"role": "assistant", "content": [
            SimpleNamespace(type="text", text="checking"),
            SimpleNamespace(type="tool_use", name="get_category_monthly_series",
                             input={"category": "Widgets"}, id="toolu_1"),
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "toolu_1", "content": '{"2025-01": 100.0}'},
        ]},
    ]
    groq_messages = _anthropic_messages_to_groq("system prompt text", anthropic_messages)

    assert groq_messages[0] == {"role": "system", "content": "system prompt text"}
    assert groq_messages[1] == {"role": "user", "content": '{"account_id": "ACC-1"}'}
    assert groq_messages[2]["role"] == "assistant"
    assert groq_messages[2]["content"] == "checking"
    assert groq_messages[2]["tool_calls"] == [{
        "id": "toolu_1", "type": "function",
        "function": {"name": "get_category_monthly_series", "arguments": '{"category": "Widgets"}'},
    }]
    assert groq_messages[3] == {"role": "tool", "tool_call_id": "toolu_1", "content": '{"2025-01": 100.0}'}


def test_response_conversion_tool_calls():
    response = groq_response(
        content=None,
        tool_calls=[groq_tool_call("call_1", "submit_verdict", {"verdict": "healthy"})],
        finish_reason="tool_calls",
    )
    converted = _groq_response_to_anthropic(response)
    assert converted.stop_reason == "tool_use"
    assert len(converted.content) == 1
    assert converted.content[0].type == "tool_use"
    assert converted.content[0].name == "submit_verdict"
    assert converted.content[0].input == {"verdict": "healthy"}
    assert converted.content[0].id == "call_1"


def test_response_conversion_plain_stop():
    response = groq_response(content="I'm done", tool_calls=[], finish_reason="stop")
    converted = _groq_response_to_anthropic(response)
    assert converted.stop_reason == "end_turn"
    assert converted.content[0].type == "text"
    assert converted.content[0].text == "I'm done"


def test_investigate_runs_unchanged_through_groq_shim():
    """The real proof: agent.investigate() needs zero changes to run
    against a Groq-shaped client instead of an Anthropic one."""
    verdict_input = {
        "verdict": "leakage_detected", "temporary_or_structural": "structural",
        "confidence": "high", "defer": False, "leak_dimensions": ["category_mix"],
        "attributed_categories": [DEFECTED_CATEGORY],
        "cited_facts": ["fact"], "narrative": "narrative", "recommended_actions": [],
        "data_needed_if_deferring": [], "model_opinion_response": "",
    }
    fake_groq = FakeGroqClient([
        groq_response(
            tool_calls=[groq_tool_call("call_a", "get_category_monthly_series", {"category": DEFECTED_CATEGORY})],
            finish_reason="tool_calls",
        ),
        groq_response(
            tool_calls=[groq_tool_call("call_b", "submit_verdict", verdict_input)],
            finish_reason="tool_calls",
        ),
    ])
    client = GroqBackedTestClient(fake_groq)

    df, _ = ingest(str(MERIDIAN_CSV))
    pack = build_evidence_pack(df, DEFECTED_ACCOUNT)
    result = investigate(client, df, DEFECTED_ACCOUNT, pack, model="claude-opus-5")

    assert result == verdict_input
    # model substitution: an Anthropic model ID must not be sent to Groq
    first_call_model = fake_groq.chat.completions.calls[0]["model"]
    assert first_call_model == DEFAULT_GROQ_MODEL
    # the output ceiling agent.py asks for is sized for Claude's thinking;
    # Groq bills the requested ceiling against its per-minute cap, so the
    # shim must clamp it rather than pass it through
    assert fake_groq.chat.completions.calls[0]["max_tokens"] == GROQ_MAX_OUTPUT_TOKENS
    # the tool result from call_a must have reached the second request as a "tool" message
    second_call_messages = fake_groq.chat.completions.calls[1]["messages"]
    tool_messages = [m for m in second_call_messages if m["role"] == "tool"]
    assert len(tool_messages) == 1
    assert tool_messages[0]["tool_call_id"] == "call_a"


def test_shim_clamps_the_output_ceiling_but_never_raises_it():
    fake_groq = FakeGroqClient([groq_response(content="ok"), groq_response(content="ok")])
    client = GroqBackedTestClient(fake_groq)
    client.messages.create(model="x", max_tokens=16000, system="s", tools=[], messages=[])
    client.messages.create(model="x", max_tokens=512, system="s", tools=[], messages=[])
    sent = [c["max_tokens"] for c in fake_groq.chat.completions.calls]
    assert sent == [GROQ_MAX_OUTPUT_TOKENS, 512]


def groq_chunk(content=None, tool_calls=None, finish_reason=None, usage=None):
    """One streamed chunk. `tool_calls` entries are (index, id, name, arguments_fragment)."""
    delta = SimpleNamespace(
        content=content,
        tool_calls=[
            SimpleNamespace(index=i, id=call_id, function=SimpleNamespace(name=name, arguments=fragment))
            for i, call_id, name, fragment in (tool_calls or [])
        ] or None,
    )
    chunk = SimpleNamespace(choices=[SimpleNamespace(delta=delta, finish_reason=finish_reason)])
    if usage is not None:
        chunk.x_groq = SimpleNamespace(usage=usage)
    return chunk


def test_stream_forwards_text_deltas_and_assembles_tool_call_fragments():
    """AryaChat streams: text must arrive as it is written, while a tool
    call's arguments — which Groq sends in pieces — must only ever appear
    assembled, in the final message, in the Anthropic shape."""
    chunks = iter([
        groq_chunk(content="Let me "),
        groq_chunk(content="check."),
        groq_chunk(tool_calls=[(0, "call_a", "get_product_changes", '{"cate')]),
        groq_chunk(tool_calls=[(0, None, None, 'gory": null}')]),
        groq_chunk(finish_reason="tool_calls",
                   usage=SimpleNamespace(prompt_tokens=900, completion_tokens=30)),
    ])
    fake_groq = FakeGroqClient([chunks])
    client = GroqBackedTestClient(fake_groq)

    seen = []
    with client.messages.stream(model="x", max_tokens=16000, system="s", tools=[], messages=[]) as stream:
        for delta in stream.text_stream:
            seen.append(delta)
        final = stream.get_final_message()

    assert seen == ["Let me ", "check."]
    assert fake_groq.chat.completions.calls[0]["stream"] is True
    assert fake_groq.chat.completions.calls[0]["max_tokens"] == GROQ_MAX_OUTPUT_TOKENS
    assert final.stop_reason == "tool_use"
    assert [b.type for b in final.content] == ["text", "tool_use"]
    assert final.content[0].text == "Let me check."
    assert final.content[1].name == "get_product_changes" and final.content[1].id == "call_a"
    assert final.content[1].input == {"category": None}
    assert (final.usage.input_tokens, final.usage.output_tokens) == (900, 30)


def test_stream_final_message_works_without_reading_the_text_first():
    fake_groq = FakeGroqClient([iter([groq_chunk(content="done"), groq_chunk(finish_reason="stop")])])
    client = GroqBackedTestClient(fake_groq)
    with client.messages.stream(model="x", max_tokens=100, system="s", tools=[], messages=[]) as stream:
        final = stream.get_final_message()
    assert final.stop_reason == "end_turn" and final.content[0].text == "done"


def test_usage_rides_on_the_converted_response_when_groq_sends_it():
    response = groq_response(content="ok")
    assert _groq_response_to_anthropic(response).usage is None
    response.usage = SimpleNamespace(prompt_tokens=10, completion_tokens=4)
    converted = _groq_response_to_anthropic(response)
    assert (converted.usage.input_tokens, converted.usage.output_tokens) == (10, 4)


def test_string_assistant_content_replays_as_plain_text():
    """A chat history replays earlier answers as plain strings, the shape
    Anthropic accepts; the shim must not iterate the characters."""
    groq_messages = _anthropic_messages_to_groq("sys", [
        {"role": "user", "content": "Why is margin down?"},
        {"role": "assistant", "content": "It fell from 32.2% to 19.9%."},
        {"role": "user", "content": "Since when?"},
    ])
    assert groq_messages[2] == {"role": "assistant", "content": "It fell from 32.2% to 19.9%."}
    assert groq_messages[3] == {"role": "user", "content": "Since when?"}


def test_dict_content_blocks_are_accepted_too():
    groq_messages = _anthropic_messages_to_groq("sys", [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": [{"type": "text", "text": "a"},
                                          {"type": "tool_use", "name": "get_kpis", "input": {}, "id": "t1"}]},
    ])
    assert groq_messages[2]["content"] == "a"
    assert groq_messages[2]["tool_calls"][0]["function"]["name"] == "get_kpis"
