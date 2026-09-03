"""
A fake Anthropic client for testing the Stage 4 agent loop without hitting
the real API. Mimics the subset of the `anthropic.Anthropic` client shape
that agent.py actually uses: `client.messages.create(...) -> response`
where `response.stop_reason` and `response.content` (a list of blocks with
`.type`, and for tool_use blocks `.name`/`.input`/`.id`) match the real SDK.
"""

from types import SimpleNamespace


def text_block(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=text)


def tool_use_block(name: str, tool_input: dict, block_id: str = "toolu_1") -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", name=name, input=tool_input, id=block_id)


def message(content: list, stop_reason: str) -> SimpleNamespace:
    return SimpleNamespace(content=content, stop_reason=stop_reason)


class ScriptedClient:
    """client.messages.create(...) returns the next scripted response each
    call, in order. Raises if the script runs out (means the code under
    test made more calls than expected)."""

    def __init__(self, responses: list):
        self._responses = list(responses)
        self.call_count = 0
        self.calls = []  # records (model, tools, messages) per call for assertions

    class _Messages:
        def __init__(self, outer):
            self._outer = outer

        def create(self, model, max_tokens, system, tools, messages):
            outer = self._outer
            if outer.call_count >= len(outer._responses):
                raise AssertionError(
                    f"ScriptedClient exhausted after {outer.call_count} calls — "
                    "the agent loop made more requests than scripted."
                )
            outer.calls.append({"model": model, "system": system, "tools": tools, "messages": messages})
            response = outer._responses[outer.call_count]
            outer.call_count += 1
            return response

    @property
    def messages(self):
        return ScriptedClient._Messages(self)
