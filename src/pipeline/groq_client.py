"""
A Groq-backed stand-in for the Anthropic client, used only because no
Claude API key is available yet (see plan.md Phase 3). It exposes the same
minimal interface agent.py's investigate() actually calls —
`client.messages.create(model, max_tokens, system, tools, messages) ->
response` where `response.stop_reason` and `response.content` (blocks with
`.type`/`.text`/`.name`/`.input`/`.id`) match Anthropic's shape — so
agent.py, impact.py, prioritize.py, and report.py needed ZERO changes.
Swapping back to Claude later is a one-line change at the call site
(construct anthropic.Anthropic() instead of GroqShimClient()); nothing in
the pipeline itself is Groq-aware.

Internally this translates Anthropic-shaped tool defs and message history
to Groq's OpenAI-compatible chat.completions format on the way in, and
translates the OpenAI-shaped response back to Anthropic-shaped content
blocks on the way out. This is a deliberate, isolated, swappable shim —
not a permanent provider abstraction; delete this file once real Claude
access exists and nothing else in the pipeline needs to change.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"

# Groq bills the REQUESTED output ceiling against its free-tier limit of
# 8,000 tokens per minute, and one evidence pack is already most of that.
# Call sites ask for what an Anthropic model needs (Claude 5-family
# thinking is charged to the same ceiling, so agent.py asks for 16,000);
# this shim clamps the request to what Groq will accept. 4,096 is the
# value every 18/18 live run was scored at.
GROQ_MAX_OUTPUT_TOKENS = 4096


def _anthropic_tool_to_groq(tool: dict) -> dict:
    return {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool["description"],
            "parameters": tool["input_schema"],
        },
    }


def _content_blocks_to_groq_assistant_message(content_blocks) -> dict:
    text_parts = []
    tool_calls = []
    for block in content_blocks:
        block_type = getattr(block, "type", None)
        if block_type == "text":
            text_parts.append(block.text)
        elif block_type == "tool_use":
            tool_calls.append({
                "id": block.id,
                "type": "function",
                "function": {"name": block.name, "arguments": json.dumps(block.input)},
            })
    msg = {"role": "assistant", "content": "\n".join(text_parts) if text_parts else None}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return msg


def _anthropic_messages_to_groq(system: str, messages: list) -> list[dict]:
    groq_messages = [{"role": "system", "content": system}]
    for msg in messages:
        role, content = msg["role"], msg["content"]

        if role == "user" and isinstance(content, str):
            groq_messages.append({"role": "user", "content": content})

        elif role == "user" and isinstance(content, list):
            # Anthropic bundles tool_result blocks into one user message;
            # Groq/OpenAI wants one separate "tool" role message each.
            for item in content:
                groq_messages.append({
                    "role": "tool",
                    "tool_call_id": item["tool_use_id"],
                    "content": item["content"],
                })

        elif role == "assistant":
            groq_messages.append(_content_blocks_to_groq_assistant_message(content))

        else:
            raise ValueError(f"Unexpected message shape for Groq conversion: role={role!r}")

    return groq_messages


def _groq_response_to_anthropic(response) -> SimpleNamespace:
    choice = response.choices[0]
    message = choice.message

    blocks = []
    if message.content:
        blocks.append(SimpleNamespace(type="text", text=message.content))

    tool_calls = getattr(message, "tool_calls", None) or []
    for call in tool_calls:
        blocks.append(SimpleNamespace(
            type="tool_use",
            name=call.function.name,
            input=json.loads(call.function.arguments),
            id=call.id,
        ))

    stop_reason = "tool_use" if choice.finish_reason == "tool_calls" else (
        "end_turn" if choice.finish_reason == "stop" else choice.finish_reason
    )
    return SimpleNamespace(content=blocks, stop_reason=stop_reason)


class _GroqMessages:
    def __init__(self, groq_client, model_override: str | None):
        self._groq_client = groq_client
        self._model_override = model_override

    def create(self, model, max_tokens, system, tools, messages):
        groq_model = self._model_override or model
        # Anthropic model IDs (e.g. "claude-opus-5") aren't valid Groq
        # model IDs — if the caller passed one through, fall back to the
        # Groq default rather than sending a request guaranteed to 404.
        if groq_model.startswith("claude-"):
            groq_model = DEFAULT_GROQ_MODEL

        response = self._groq_client.chat.completions.create(
            model=groq_model,
            max_tokens=min(max_tokens, GROQ_MAX_OUTPUT_TOKENS),
            messages=_anthropic_messages_to_groq(system, messages),
            tools=[_anthropic_tool_to_groq(t) for t in tools],
            tool_choice="auto",
        )
        return _groq_response_to_anthropic(response)


class GroqShimClient:
    """Drop-in stand-in for anthropic.Anthropic() backed by Groq. Usage:

        client = GroqShimClient()  # reads GROQ_API_KEY from env
        investigate(client, df, account_id, evidence_pack, model="claude-opus-5")

    The `model` argument passed to investigate() is ignored if it's an
    Anthropic model ID (see _GroqMessages.create) — pass model_override
    explicitly here to pin a specific Groq model instead of the default.
    """

    def __init__(self, api_key: str | None = None, model_override: str | None = None):
        import groq
        self._groq_client = groq.Groq(api_key=api_key)
        self.messages = _GroqMessages(self._groq_client, model_override)
