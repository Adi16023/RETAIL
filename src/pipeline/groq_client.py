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

`messages.stream(...)` mirrors the Anthropic SDK's streaming entry point
(`with ... as s: for delta in s.text_stream`, then `s.get_final_message()`)
for AryaChat, which streams an answer as it is written. The investigator
and the other agents keep using `create`.
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
    # Anthropic accepts a plain string as assistant content (how a chat
    # history replays earlier answers) as well as content blocks; the blocks
    # may be SDK objects or plain dicts. All three shapes are valid input.
    if isinstance(content_blocks, str):
        return {"role": "assistant", "content": content_blocks}
    text_parts = []
    tool_calls = []
    for block in content_blocks:
        if isinstance(block, dict):
            block_type, text = block.get("type"), block.get("text")
            name, tool_input, call_id = block.get("name"), block.get("input"), block.get("id")
        else:
            block_type, text = getattr(block, "type", None), getattr(block, "text", None)
            name, tool_input, call_id = (getattr(block, "name", None), getattr(block, "input", None),
                                         getattr(block, "id", None))
        if block_type == "text":
            text_parts.append(text)
        elif block_type == "tool_use":
            tool_calls.append({
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(tool_input)},
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


def _groq_usage_to_anthropic(usage) -> SimpleNamespace | None:
    """Groq's prompt/completion counts in the Anthropic attribute names, so a
    caller summing `response.usage.input_tokens` works on either client.
    None when the response carried no usage (older shapes, mid-stream)."""
    if usage is None:
        return None
    return SimpleNamespace(
        input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
        output_tokens=getattr(usage, "completion_tokens", 0) or 0,
    )


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
    return SimpleNamespace(
        content=blocks, stop_reason=stop_reason,
        usage=_groq_usage_to_anthropic(getattr(response, "usage", None)),
    )


class _GroqStream:
    """Groq's chunked completion, in the shape of the Anthropic SDK's
    MessageStream — `with client.messages.stream(...) as s`, `s.text_stream`
    yielding text deltas, `s.get_final_message()` returning the finished
    message in the same Anthropic shape `create()` returns.

    Text deltas are forwarded as they arrive; tool-call fragments (Groq
    streams a call's arguments in pieces, keyed by index) are accumulated
    and only become tool_use blocks in the final message. That is the
    contract the chat loop relies on: what streams is prose, what it acts
    on is the finished message.
    """

    def __init__(self, chunks):
        self._chunks = chunks
        self._text: list[str] = []
        self._calls: dict[int, dict] = {}
        self._finish_reason = None
        self._usage = None
        self._drained = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        close = getattr(self._chunks, "close", None)
        if callable(close):
            close()
        return False

    @property
    def text_stream(self):
        return self._iterate()

    def _iterate(self):
        if self._drained:
            return
        for chunk in self._chunks:
            # Usage rides on the last chunk, under x_groq on Groq's SDK.
            usage = getattr(getattr(chunk, "x_groq", None), "usage", None) or getattr(chunk, "usage", None)
            if usage is not None:
                self._usage = usage
            choices = getattr(chunk, "choices", None) or []
            if not choices:
                continue
            choice = choices[0]
            delta = getattr(choice, "delta", None)
            if delta is not None:
                content = getattr(delta, "content", None)
                if content:
                    self._text.append(content)
                    yield content
                for call in getattr(delta, "tool_calls", None) or []:
                    slot = self._calls.setdefault(
                        getattr(call, "index", 0) or 0, {"id": None, "name": None, "arguments": ""},
                    )
                    if getattr(call, "id", None):
                        slot["id"] = call.id
                    function = getattr(call, "function", None)
                    if function is not None:
                        if getattr(function, "name", None):
                            slot["name"] = function.name
                        if getattr(function, "arguments", None):
                            slot["arguments"] += function.arguments
            if getattr(choice, "finish_reason", None):
                self._finish_reason = choice.finish_reason
        self._drained = True

    def get_final_message(self) -> SimpleNamespace:
        for _ in self._iterate():  # a caller that never read text_stream still gets the message
            pass
        tool_calls = [
            SimpleNamespace(id=c["id"], function=SimpleNamespace(name=c["name"], arguments=c["arguments"] or "{}"))
            for _, c in sorted(self._calls.items())
        ]
        message = SimpleNamespace(content="".join(self._text) or None, tool_calls=tool_calls)
        finish_reason = self._finish_reason or ("tool_calls" if tool_calls else "stop")
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=message, finish_reason=finish_reason)],
            usage=self._usage,
        )
        return _groq_response_to_anthropic(response)


class _GroqMessages:
    def __init__(self, groq_client, model_override: str | None):
        self._groq_client = groq_client
        self._model_override = model_override

    def _request(self, model, max_tokens, system, tools, messages) -> dict:
        groq_model = self._model_override or model
        # Anthropic model IDs (e.g. "claude-opus-5") aren't valid Groq
        # model IDs — if the caller passed one through, fall back to the
        # Groq default rather than sending a request guaranteed to 404.
        if groq_model.startswith("claude-"):
            groq_model = DEFAULT_GROQ_MODEL
        return {
            "model": groq_model,
            "max_tokens": min(max_tokens, GROQ_MAX_OUTPUT_TOKENS),
            "messages": _anthropic_messages_to_groq(system, messages),
            "tools": [_anthropic_tool_to_groq(t) for t in tools],
            "tool_choice": "auto",
        }

    def create(self, model, max_tokens, system, tools, messages):
        response = self._groq_client.chat.completions.create(
            **self._request(model, max_tokens, system, tools, messages)
        )
        return _groq_response_to_anthropic(response)

    def stream(self, model, max_tokens, system, tools, messages) -> _GroqStream:
        """Same request, streamed. Mirrors `anthropic.Anthropic().messages.stream`
        so the chat loop can stream on either provider without knowing which."""
        chunks = self._groq_client.chat.completions.create(
            **self._request(model, max_tokens, system, tools, messages), stream=True,
        )
        return _GroqStream(chunks)


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
