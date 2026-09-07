"""
A fake Anthropic client for testing the Stage 4 agent loop without hitting
the real API. Mimics the subset of the `anthropic.Anthropic` client shape
that agent.py actually uses: `client.messages.create(...) -> response`
where `response.stop_reason` and `response.content` (a list of blocks with
`.type`, and for tool_use blocks `.name`/`.input`/`.id`) match the real SDK.

It also mimics `client.messages.stream(...)` — the entry point AryaChat
uses to stream an answer — as a context manager whose `text_stream` yields
the scripted text in two halves (so a tag split across deltas is exercised)
and whose `get_final_message()` returns the scripted response. Construct
with `streaming=False` to get a client that only has `create`, the way a
provider without streaming would look.
"""

from types import SimpleNamespace


def text_block(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=text)


def tool_use_block(name: str, tool_input: dict, block_id: str = "toolu_1") -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", name=name, input=tool_input, id=block_id)


def message(content: list, stop_reason: str, usage: dict | None = None) -> SimpleNamespace:
    """`usage`, when given, is {"input_tokens": n, "output_tokens": m} — the
    two counts the chat loop sums."""
    return SimpleNamespace(
        content=content, stop_reason=stop_reason,
        usage=SimpleNamespace(**usage) if usage else None,
    )


class _ScriptedStream:
    def __init__(self, response):
        self._response = response

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    @property
    def text_stream(self):
        for block in self._response.content:
            text = getattr(block, "text", None) if getattr(block, "type", None) == "text" else None
            if not text:
                continue
            mid = max(1, len(text) // 2)
            yield text[:mid]
            yield text[mid:]

    def get_final_message(self):
        return self._response


class ScriptedClient:
    """client.messages.create(...) returns the next scripted response each
    call, in order. Raises if the script runs out (means the code under
    test made more calls than expected)."""

    def __init__(self, responses: list, streaming: bool = True):
        self._responses = list(responses)
        self._streaming = streaming
        self.call_count = 0
        self.calls = []  # records (model, tools, messages) per call for assertions
        self.streamed = 0  # how many of those calls went through stream()

    def _next(self, model, max_tokens, system, tools, messages):
        if self.call_count >= len(self._responses):
            raise AssertionError(
                f"ScriptedClient exhausted after {self.call_count} calls — "
                "the agent loop made more requests than scripted."
            )
        self.calls.append({"model": model, "max_tokens": max_tokens, "system": system,
                           "tools": tools, "messages": messages})
        response = self._responses[self.call_count]
        self.call_count += 1
        return response

    class _Messages:
        def __init__(self, outer):
            self._outer = outer

        def create(self, model, max_tokens, system, tools, messages):
            return self._outer._next(model, max_tokens, system, tools, messages)

    class _StreamingMessages(_Messages):
        def stream(self, model, max_tokens, system, tools, messages):
            self._outer.streamed += 1
            return _ScriptedStream(self._outer._next(model, max_tokens, system, tools, messages))

    @property
    def messages(self):
        if self._streaming:
            return ScriptedClient._StreamingMessages(self)
        return ScriptedClient._Messages(self)
