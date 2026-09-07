"""
AryaChat (src/pipeline/aryachat.py) and its JSON chat store (src/ui/chatstore.py).

What must hold:
- The model is shown a small ACCOUNT INDEX and, when there is one, the brief
  of the ACCOUNT IN FOCUS — never a pack: no underscore keys, no rows, and it
  fits a token budget.
- The loop works like an MCP tool loop: LLM -> tools -> results -> LLM until
  the model answers in prose; every tool takes the account it is about; the
  tools it ran are recorded with a preview; no answer at all is a clear
  error; the step ceiling returns what it had, marked truncated.
- Focus is sticky: the last account a tool touched becomes the chat's focus.
- Progress is emitted as events: text deltas as the model writes (through a
  streaming client, or as one delta through a client that cannot stream), a
  tool_start / tool_end pair per call, a retry when a draft is refused, and
  the next-question chips once, after the text.
- Next-question chips are parsed out of <follow_up> tags on every exit path
  and never reach the text — not even when a tag is split across deltas.
- Every figure in an answer is checked against the opening context and this
  turn's tool results; anything found nowhere is reported.
- The book tools (find, book, brief, compare) and the pattern tools (orders,
  products, category mix, monthly table) return real figures from the pipeline.
- Context is managed inside one chat: only the last KEEP_LAST_TURNS are
  replayed verbatim, the compacted summary rides in the opening message, and
  a question whose answer never arrived is never replayed.
- Compaction is one call that returns a summary.
- The store round-trips a chat with its focus, isolates by scope +
  fingerprint, and titles a chat from its first question.
"""

import functools
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pipeline.aryachat import (
    CHAT_TOOLS,
    KEEP_LAST_TURNS,
    MAX_TOOL_RESULT_CHARS,
    SYSTEM_PROMPT,
    ChatError,
    FollowUpStreamFilter,
    _execute_chat_tool,
    ask,
    build_account_brief,
    build_account_index,
    build_messages,
    compact_conversation,
    find_accounts,
    required_tools_for,
    split_follow_ups,
    unsourced_figures,
)
from pipeline.evidence import build_evidence_pack
from pipeline.ingest import ingest
from datetime import date

from ui import chatstore

from datasets import MERIDIAN_CSV
from fakes import ScriptedClient, message, text_block, tool_use_block


@pytest.fixture(scope="module")
def df():
    clean, _ = ingest(str(MERIDIAN_CSV))
    return clean


@pytest.fixture(scope="module")
def pack_for(df):
    return functools.lru_cache(maxsize=None)(lambda account_id: build_evidence_pack(df, account_id))


@pytest.fixture(scope="module")
def pack(pack_for):
    return pack_for("ACC-101")


ANSWER = "Margin rate fell from 32.1% to 19.9% while revenue held flat."
CHIPS = "\n<follow_up>Which products fell in April 2026?</follow_up>\n<follow_up>Is this seasonal?</follow_up>"
REPORT = {
    "verdict": "leakage_detected", "temporary_or_structural": "structural", "confidence": "high",
    "defer": False, "leak_dimensions": ["margin"], "attributed_categories": [], "narrative": "x",
    "recommended_actions": ["Call them"],
    "financial_impact": {"total_monthly_revenue_at_risk": 121386.0, "margin_impact": {"monthly_margin_at_risk": 34729.0},
                         "overall_severity_pct_of_baseline": 0.44},
    "prioritization": {"priority": "High", "churn_risk_projection": {"projected_12_month_loss_if_unaddressed": 1456633.0}},
}


def prose(text: str = ANSWER + CHIPS, stop_reason: str = "end_turn", usage=None):
    return message([text_block(text)], stop_reason=stop_reason, usage=usage)


def kpis_call(account_id="ACC-101", block_id="t1"):
    return message([text_block("Let me check."), tool_use_block("get_kpis", {"account_id": account_id}, block_id)],
                   stop_reason="tool_use")


# --- The opening context ---------------------------------------------------------------

def _walk(value, seen):
    if isinstance(value, dict):
        for k, v in value.items():
            seen.append(k)
            _walk(v, seen)
    elif isinstance(value, list):
        for v in value:
            _walk(v, seen)


def test_brief_is_a_digest_not_the_pack(pack):
    brief = build_account_brief(pack)
    keys = []
    _walk(brief, keys)
    assert not any(k.startswith("_") for k in keys), "underscore (presentation) keys leaked"
    for forbidden in ("monthly_series", "significance", "product_changes", "_presentation", "p_value"):
        assert forbidden not in keys
    assert brief["verdict"].startswith("not yet investigated")
    assert brief["signals"]["margin"] == "margin rate has fallen"
    assert brief["signals"]["tier_mix"] == "spend is moving into cheaper lines"
    kpis = brief["kpis_baseline_vs_recent"]
    assert kpis["margin_rate"]["recent"] == 19.9
    # The change is taken on the raw values, then rounded — the same order
    # the dashboard tile uses — so an answer and the tile never disagree.
    assert kpis["margin_rate"]["change"] == -12.3
    assert kpis["high_tier_share"]["change"] == -43.8
    assert kpis["revenue_per_month"]["change"] == 7030
    assert len(json.dumps(brief)) < 6000, "the brief must stay a fraction of the pack"


def test_brief_carries_the_verdict_and_money_when_investigated(pack):
    brief = build_account_brief(pack, REPORT)
    assert brief["verdict"]["verdict"] == "leakage_detected"
    assert brief["money"]["revenue_at_risk_per_month_rs"] == 121386
    assert brief["money"]["exposure_over_12_months_rs"] == 1456633


def test_the_index_is_one_short_line_per_account(df):
    index = build_account_index(df, lambda a: REPORT if a == "ACC-101" else None)
    assert index["accounts_in_book"] == 18 and len(index["accounts"]) == 18
    first = index["accounts"][0]
    assert first == {"account_id": "ACC-101", "name": "Northgate Traders", "region": "North",
                     "verdict": "leakage detected, structural, priority High"}
    assert index["accounts"][1]["verdict"] == "not analysed"
    assert len(json.dumps(index)) < 2500, "the index rides on every turn; it must stay small"


def test_opening_message_carries_the_index_then_the_focus_then_the_question(df, pack):
    messages = build_messages(build_account_index(df), build_account_brief(pack), None, [], "Why?")
    opening = messages[0]["content"]
    assert opening.startswith("ACCOUNT INDEX:") and "ACCOUNT IN FOCUS:\n{" in opening
    assert messages[-1] == {"role": "user", "content": "Why?"}
    no_focus = build_messages(build_account_index(df), None, None, [], "Why?")[0]["content"]
    assert "ACCOUNT IN FOCUS: none yet" in no_focus


# --- The loop ---------------------------------------------------------------------------

def test_ask_calls_a_tool_on_the_focus_account_then_answers_in_prose(df, pack_for):
    client = ScriptedClient([kpis_call(), prose()])
    result = ask(client, df, "Why is margin down?", pack_for=pack_for, focus_account="ACC-101")
    assert result["answer"] == ANSWER
    assert result["follow_ups"] == ["Which products fell in April 2026?", "Is this seasonal?"]
    assert result["tools_used"] == ["get_kpis"]
    assert result["accounts_touched"] == ["ACC-101"] and result["focus_account"] == "ACC-101"
    assert result["truncated"] is False and result["iterations"] == 2
    log = result["tool_log"][0]
    assert log["ok"] is True and log["accounts"] == ["ACC-101"] and log["result_chars"] > 100
    assert log["result_preview"].startswith('{"revenue_per_month"') and len(log["result_preview"]) <= 500
    # Every figure in the answer is in the focus brief, so nothing is flagged.
    assert result["unsourced_figures"] == []
    # Without an event callback the loop uses create(), never stream().
    assert client.streamed == 0
    first = client.calls[0]["messages"]
    assert first[0]["content"].startswith("ACCOUNT INDEX:") and "ACC-101" in first[0]["content"]
    assert first[2] == {"role": "user", "content": "Why is margin down?"}
    second = client.calls[1]["messages"]
    assert second[-1]["content"][0]["tool_use_id"] == "t1"
    assert '"margin_rate"' in second[-1]["content"][0]["content"]
    # Every tool the model can call takes the account it is about, except the book-wide ones.
    for tool in CHAT_TOOLS:
        if tool["name"] not in ("find_account", "get_account_book", "compare_accounts"):
            assert tool["input_schema"]["required"][0] == "account_id", tool["name"]


def test_a_named_account_becomes_the_focus(df, pack_for):
    """No account in focus; the manager asks about ACC-104 by name; the model
    resolves it and calls its brief. The chat is now about ACC-104."""
    client = ScriptedClient([
        message([tool_use_block("find_account", {"query": "Harbor"}, "f1")], stop_reason="tool_use"),
        message([tool_use_block("get_account_brief", {"account_id": "ACC-104"}, "b1")], stop_reason="tool_use"),
        prose("Revenue is down sharply." + CHIPS),
    ])
    result = ask(client, df, "What is happening on Harbor?", pack_for=pack_for)
    assert result["focus_account"] == "ACC-104" and result["accounts_touched"] == ["ACC-104"]
    assert result["tools_used"] == ["find_account", "get_account_brief"]
    assert client.calls[0]["messages"][0]["content"].count("ACCOUNT IN FOCUS: none yet") == 1


def test_focus_is_kept_when_no_tool_names_an_account(df, pack_for):
    client = ScriptedClient([prose("It is structural." + CHIPS)])
    result = ask(client, df, "Is that structural?", pack_for=pack_for, focus_account="ACC-107")
    assert result["focus_account"] == "ACC-107" and result["accounts_touched"] == []
    # A comparison touches several accounts but is still about the one in focus.
    client = ScriptedClient([
        message([tool_use_block("compare_accounts", {"account_ids": ["ACC-107", "ACC-103"]}, "c1")], stop_reason="tool_use"),
        prose("Harbor Point is healthier." + CHIPS),
    ])
    result = ask(client, df, "compare it with Harbor", pack_for=pack_for, focus_account="ACC-107")
    assert result["focus_account"] == "ACC-107" and result["accounts_touched"] == ["ACC-107", "ACC-103"]
    # An unknown focus is dropped rather than sent.
    result = ask(ScriptedClient([prose("Which account?")]), df, "Why?", pack_for=pack_for, focus_account="ACC-999")
    assert result["focus_account"] is None


def test_a_turn_streams_text_and_brackets_each_tool_with_events(df, pack_for):
    client = ScriptedClient([kpis_call(), prose()])
    events = []
    result = ask(client, df, "Why?", pack_for=pack_for, focus_account="ACC-101", on_event=events.append)
    types = [e.type for e in events]
    assert client.streamed == 2, "with a callback, every LLM call streams"
    assert types[0] == "text" and "tool_start" in types and "tool_end" in types
    assert types.index("tool_start") < types.index("tool_end")
    assert types[-1] == "follow_ups" and types.count("follow_ups") == 1
    assert types[types.index("tool_end") + 1:].count("text") >= 2
    start = next(e for e in events if e.type == "tool_start")
    end = next(e for e in events if e.type == "tool_end")
    assert start.tool_name == end.tool_name == "get_kpis" and start.tool_id == end.tool_id == "t1"
    assert start.arguments == {"account_id": "ACC-101"}
    assert end.ok is True and end.result_chars > 100 and end.result_preview
    streamed = "".join(e.text for e in events[types.index("tool_end") + 1:] if e.type == "text")
    assert streamed.strip() == ANSWER and "<follow_up" not in streamed
    assert events[-1].follow_ups == result["follow_ups"]


def test_a_client_that_cannot_stream_still_emits_the_text_once(df, pack_for):
    client = ScriptedClient([prose()], streaming=False)
    events = []
    result = ask(client, df, "Why?", pack_for=pack_for, focus_account="ACC-101", on_event=events.append)
    texts = [e for e in events if e.type == "text"]
    assert len(texts) == 1 and texts[0].text.strip() == ANSWER
    assert result["answer"] == ANSWER and result["follow_ups"]


def test_ask_raises_when_the_model_returns_nothing_at_all(df, pack_for):
    with pytest.raises(ChatError, match="without answering"):
        ask(ScriptedClient([message([], stop_reason="end_turn")]), df, "Why?", pack_for=pack_for)


def test_running_out_of_room_gets_one_brief_retry(df, pack_for):
    client = ScriptedClient([message([], stop_reason="length"), prose()])
    result = ask(client, df, "Why?", pack_for=pack_for, focus_account="ACC-101")
    assert result["answer"] == ANSWER
    retry = client.calls[1]["messages"][-1]
    assert retry["role"] == "user" and "ran out of room" in retry["content"]
    assert client.calls[0]["max_tokens"] >= 4096, "must ask for at least the shim's ceiling"


def test_a_bad_tool_call_is_reported_to_the_model_not_the_caller(df, pack_for):
    client = ScriptedClient([
        message([tool_use_block("get_kpis", {"account_id": "Northgate"}, "t1")], stop_reason="tool_use"),
        prose(),
    ])
    events = []
    result = ask(client, df, "Why?", pack_for=pack_for, focus_account="ACC-101", on_event=events.append)
    assert result["tool_log"][0]["ok"] is False
    assert "Did you mean: ACC-101 (Northgate Traders)" in result["tool_log"][0]["error"]
    assert next(e for e in events if e.type == "tool_end").ok is False
    sent_back = client.calls[1]["messages"][-1]["content"][0]
    assert sent_back["is_error"] is True and sent_back["content"].startswith("Error:")
    assert result["focus_account"] == "ACC-101", "a failed call touches no account"


def test_a_product_question_about_the_focus_account_is_sent_back_until_a_product_tool_runs(df, pack_for):
    client = ScriptedClient([
        prose("Two products stopped."),
        message([tool_use_block("get_product_changes", {"account_id": "ACC-101", "category": None}, "t1")], stop_reason="tool_use"),
        prose("Seven products moved." + CHIPS),
    ])
    events = []
    result = ask(client, df, "Which products dropped, and when?", pack_for=pack_for,
                 focus_account="ACC-101", on_event=events.append)
    assert result["answer"] == "Seven products moved."
    assert result["tools_used"] == ["get_product_changes"]
    types = [e.type for e in events]
    assert types.count("retry") == 1
    assert "".join(e.text for e in events[:types.index("retry")] if e.type == "text") == "Two products stopped."
    pushback = client.calls[0]["messages"][4]
    assert pushback["role"] == "user" and pushback["content"].startswith("Not accepted yet")


def test_no_push_back_when_the_question_has_no_account_yet_or_spans_the_book(df, pack_for):
    """With nothing in focus the right answer to 'which products' may be
    'which account?', and a book-level question is answered from the book."""
    result = ask(ScriptedClient([prose("Which account do you mean?")]), df, "Which products dropped?", pack_for=pack_for)
    assert result["answer"] == "Which account do you mean?" and result["tools_used"] == []
    assert required_tools_for("Which accounts lost products?") == []
    assert [t for t, _ in required_tools_for("Which products dropped?")] == [("get_product_changes", "get_product_revenue")]
    assert [t for t, _ in required_tools_for("Is this just seasonal?")] == [("get_ruled_out_checks",)]
    assert [t for t, _ in required_tools_for("What could we do, and what is it worth?")] == [("get_priced_options",)]


def test_the_push_back_happens_once(df, pack_for):
    client = ScriptedClient([prose("First."), prose("Still no tool.")])
    result = ask(client, df, "Is this seasonal?", pack_for=pack_for, focus_account="ACC-101")
    assert result["answer"] == "Still no tool." and result["tools_used"] == []
    assert client.call_count == 2


def test_the_step_ceiling_returns_what_it_had_marked_truncated(df, pack_for):
    client = ScriptedClient([
        message([text_block("Looking at the tiles." + CHIPS), tool_use_block("get_kpis", {"account_id": "ACC-101"}, "t1")],
                stop_reason="tool_use"),
        message([tool_use_block("get_kpis", {"account_id": "ACC-101"}, "t2")], stop_reason="tool_use"),
    ])
    result = ask(client, df, "Why?", pack_for=pack_for, focus_account="ACC-101", max_iterations=2)
    assert result["truncated"] is True and result["iterations"] == 2
    assert result["answer"] == "Looking at the tiles." and result["follow_ups"]
    # With no text written at all before the ceiling, there is nothing to stand.
    silent_call = message([tool_use_block("get_kpis", {"account_id": "ACC-101"}, "t1")], stop_reason="tool_use")
    with pytest.raises(ChatError, match="max_iterations"):
        ask(ScriptedClient([silent_call]), df, "Why?", pack_for=pack_for, max_iterations=1)


def test_usage_is_summed_across_the_calls_of_a_turn(df, pack_for):
    client = ScriptedClient([
        message([tool_use_block("get_kpis", {"account_id": "ACC-101"}, "t1")], stop_reason="tool_use",
                usage={"input_tokens": 1000, "output_tokens": 20}),
        prose(usage={"input_tokens": 1500, "output_tokens": 80}),
    ])
    assert ask(client, df, "Why?", pack_for=pack_for)["usage"] == {"input_tokens": 2500, "output_tokens": 100}
    assert ask(ScriptedClient([prose()]), df, "Why?", pack_for=pack_for)["usage"] == {}


def test_a_huge_tool_result_is_capped_with_a_marker(df, pack_for, monkeypatch):
    import pipeline.aryachat as module
    monkeypatch.setattr(module, "_tool_get_monthly_table", lambda *_: [{"row": i} for i in range(5000)])
    client = ScriptedClient([
        message([tool_use_block("get_monthly_table", {"account_id": "ACC-101"}, "t1")], stop_reason="tool_use"),
        prose(),
    ])
    result = ask(client, df, "What happened in April 2026?", pack_for=pack_for, focus_account="ACC-101")
    sent = client.calls[1]["messages"][-1]["content"][0]["content"]
    assert len(sent) < MAX_TOOL_RESULT_CHARS + 200 and "[truncated:" in sent
    assert result["tool_log"][0]["result_chars"] == len(sent)


def test_prompt_forbids_arithmetic_and_internal_vocabulary_and_asks_for_chips():
    assert "never calculate" in SYSTEM_PROMPT.lower()
    assert "erosion_detected" in SYSTEM_PROMPT  # named so it can be forbidden
    assert "NEVER PREDICT THE CUSTOMER" in SYSTEM_PROMPT
    assert "<follow_up>" in SYSTEM_PROMPT and "ACCOUNT IN FOCUS" in SYSTEM_PROMPT
    assert "submit_answer" not in SYSTEM_PROMPT


# --- Next-question chips --------------------------------------------------------------

def test_chips_are_parsed_out_capped_and_deduplicated():
    text = ("Answer.\n<follow_up>One?</follow_up>\n<follow_up>Two?</follow_up>"
            "<follow_up> one? </follow_up><follow_up>Three?</follow_up><follow_up>Four?</follow_up>")
    clean, chips = split_follow_ups(text)
    assert clean == "Answer."
    assert chips == ["One?", "Two?", "Three?"], "duplicates dropped, capped at three, all tags stripped"
    assert split_follow_ups("No chips here.") == ("No chips here.", [])


def test_the_stream_filter_withholds_tags_split_across_deltas_and_releases_a_plain_bracket():
    out = []
    flt = FollowUpStreamFilter(out.append)
    for delta in ["Margin a < b fell.", " <fol", "low_up>Which prod", "ucts?</follow_up>", " <follow_up>never closed"]:
        flt.push(delta)
    flt.flush()
    # Two spaces: one preceded each dropped tag, and ordinary text is never eaten.
    assert "".join(out) == "Margin a < b fell.  "
    assert not any("follow_up" in piece for piece in out)


# --- The figure check ------------------------------------------------------------------

def test_figures_absent_from_every_source_are_reported(pack):
    brief = build_account_brief(pack)
    answer = ("Revenue held at Rs 283,313 a month while margin fell from 32.1% to 19.9%, "
              "down 12 points; high-tier share is 22.5%. A mystery 45% and Rs 9,999 came from nowhere.")
    assert unsourced_figures(answer, [brief]) == ["45", "9,999"]


def test_dates_small_counts_and_rounded_forms_are_not_false_alarms(pack):
    brief = build_account_brief(pack)
    answer = "In April 2026 (2026-04) 3 products fell over 6 months; revenue Rs 283,313, margin 32.2%."
    tool_result = '{"2026-04":{"margin_pct":0.3215}}'
    assert unsourced_figures(answer, [brief, tool_result]) == []
    assert unsourced_figures("Discount rose 7 points to 9%.", ['{"x":1}']) == ["7", "9"]
    assert unsourced_figures("Discount is 5%.", ['{"discount":0.05}']) == []


def test_an_earlier_answer_is_a_legitimate_source_for_a_figure(df, pack_for):
    turns = [{"role": "user", "text": "Which products?"},
             {"role": "assistant", "text": "PT-Cordless Drill 18V fell from Rs 45,460 to Rs 5,609."}]
    client = ScriptedClient([prose("As said, the drill fell to Rs 5,609 a month.")])
    result = ask(client, df, "Remind me of the drill?", pack_for=pack_for, focus_account="ACC-101", turns=turns)
    assert result["unsourced_figures"] == []


# --- Tools return real figures ---------------------------------------------------------

def test_book_tools_resolve_compare_and_tabulate(df, pack_for):
    assert [m["account_id"] for m in find_accounts(df, "northgate")] == ["ACC-101"]
    assert len(find_accounts(df, "")) == 10, "an empty query lists the first ten"

    book, touched = _execute_chat_tool(df, pack_for, lambda a: REPORT if a == "ACC-101" else None, "get_account_book", {})
    assert touched == [] and len(book["accounts"]) == 18
    row = book["accounts"][0]
    assert row["account_id"] == "ACC-101" and row["verdict"].startswith("leakage detected")
    assert row["signals"]["margin"] == "margin rate has fallen" and row["revenue_per_month_recent_rs"] == 283313
    # An investigated account carries what the Prioritise page ranks on.
    assert row["priority"] == "High" and row["revenue_at_risk_per_month_rs"] == 121386
    assert row["exposure_over_12_months_rs"] == 1456633
    assert "priority" not in book["accounts"][1], "an uninvestigated account has no money at risk to quote"

    brief, touched = _execute_chat_tool(df, pack_for, None, "get_account_brief", {"account_id": "acc-101"})
    assert touched == ["ACC-101"] and brief["account"]["account_name"] == "Northgate Traders"

    digest, touched = _execute_chat_tool(df, pack_for, None, "compare_accounts", {"account_ids": ["ACC-101", "ACC-107", "ACC-101"]})
    assert touched == ["ACC-101", "ACC-107"]
    assert [a["account_id"] for a in digest["accounts"]] == ["ACC-101", "ACC-107"]
    with pytest.raises(ValueError, match="at least two"):
        _execute_chat_tool(df, pack_for, None, "compare_accounts", {"account_ids": ["ACC-101"]})


def test_pattern_tools_read_the_order_file(df, pack_for):
    orders, _ = _execute_chat_tool(df, pack_for, None, "get_orders", {"account_id": "ACC-101"})
    assert orders["orders_in_history"] == 30 and len(orders["orders"]) == 30
    first = orders["orders"][0]
    assert first["order_id"] == "ORD-00001" and first["date"] == "2024-09-15"
    assert first["lines"] == 8 and first["revenue"] == 291827.76 and "Power Tools" in first["categories"]

    products, _ = _execute_chat_tool(df, pack_for, None, "get_product_revenue", {"account_id": "ACC-101"})
    assert products["products_in_history"] == 20 and products["recent_window_starts"] == "2026-03"
    drill = next(p for p in products["products"] if p["product_id"] == "PT-Cordless Drill 18V")
    assert drill["category"] == "Power Tools" and drill["tier"] == "High"
    assert drill["baseline_monthly_avg"] > drill["recent_monthly_avg"] > 0

    mix, _ = _execute_chat_tool(df, pack_for, None, "get_category_mix", {"account_id": "ACC-101"})
    power = next(c for c in mix["categories"] if c["category"] == "Power Tools")
    assert power["baseline_share_percent"] == 30.3 and power["recent_share_percent"] < 15

    table, _ = _execute_chat_tool(df, pack_for, None, "get_monthly_table", {"account_id": "ACC-101"})
    assert len(table) == 24 and table[-1]["month"] == "2026-08" and table[-1]["margin_pct_percent"] == 18.6

    kpis, _ = _execute_chat_tool(df, pack_for, None, "get_kpis", {"account_id": "ACC-101"})
    assert kpis["high_tier_share"]["baseline"] == 66.2 and kpis["high_tier_share"]["recent"] == 22.5

    checks, _ = _execute_chat_tool(df, pack_for, None, "get_ruled_out_checks", {"account_id": "ACC-101"})
    assert checks["product_mix_direction"].startswith("unfavourable") and checks["seasonal_repeat"]["confirmed"] is False

    options, _ = _execute_chat_tool(df, pack_for, None, "get_priced_options", {"account_id": "ACC-101"})
    assert options and options[0]["lever"] == "mix_recovery"
    nothing, _ = _execute_chat_tool(df, pack_for, None, "get_priced_options", {"account_id": "ACC-102"})
    assert nothing["priced_options"] == []

    # The investigator's drill-downs run on the named account.
    changes, touched = _execute_chat_tool(df, pack_for, None, "get_product_changes", {"account_id": "ACC-101", "category": "Power Tools"})
    assert touched == ["ACC-101"] and all(c["category"] == "Power Tools" for c in changes)


# --- Context inside a chat ------------------------------------------------------------

def _turns(n: int) -> list[dict]:
    out = []
    for i in range(n):
        out.append({"role": "user", "text": f"Q{i}"})
        out.append({"role": "assistant", "text": f"A{i}"})
    return out


def test_only_the_last_turns_are_replayed_and_the_summary_rides_up_front(df):
    turns = _turns(10)  # 20 turns
    messages = build_messages(build_account_index(df), None, "Earlier: margin fell.", turns, "Next?")
    assert "EARLIER IN THIS CHAT" in messages[0]["content"] and "margin fell" in messages[0]["content"]
    replayed = [m["content"] for m in messages[2:-1]]
    assert len(replayed) == KEEP_LAST_TURNS
    assert replayed[0] == f"Q{10 - KEEP_LAST_TURNS // 2}"       # the oldest replayed question
    assert "Q0" not in replayed and "A0" not in replayed
    assert messages[-1]["content"] == "Next?"


def test_an_unanswered_question_is_never_replayed(df):
    turns = [{"role": "user", "text": "Q0"}, {"role": "assistant", "text": "A0"},
             {"role": "user", "text": "dangling"}]
    messages = build_messages(build_account_index(df), None, None, turns, "Next?")
    contents = [m["content"] for m in messages]
    assert "dangling" not in contents
    assert contents[-3:] == ["Q0", "A0", "Next?"]


def test_compact_conversation_returns_the_submitted_summary():
    client = ScriptedClient([
        message([tool_use_block("submit_summary", {"summary": "Manager asked why margin fell; it went 32.1% to 19.9%."}, "s1")],
                stop_reason="tool_use"),
    ])
    summary = compact_conversation(client, _turns(4), summary="older", model="test-model")
    assert summary.startswith("Manager asked")
    body = client.calls[0]["messages"][0]["content"]
    assert "EXISTING SUMMARY" in body and "older" in body and "Manager: Q0" in body
    assert [t["name"] for t in client.calls[0]["tools"]] == ["submit_summary"]


def test_compact_raises_without_a_summary_tool_call():
    client = ScriptedClient([message([text_block("summary in prose")], stop_reason="end_turn")])
    with pytest.raises(ChatError, match="submit_summary"):
        compact_conversation(client, _turns(2))


# --- The store ---------------------------------------------------------------------------

@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(chatstore, "CHAT_DIR", tmp_path / "aryachat")
    return chatstore


def test_store_round_trips_a_chat_with_its_focus_and_titles_it_from_the_first_question(store):
    chat = store.new_chat("book", "abc123", "test-model", focus_account="ACC-101")
    assert store.list_chats("book", "abc123")[0]["title"] == "New chat"
    assert store.list_chats("book", "abc123")[0]["focus_account"] == "ACC-101"
    store.append_turn(chat, "user", "Why is the margin on this account falling so fast lately?")
    store.append_turn(chat, "assistant", "Because...", tools_used=["get_kpis"], accounts_touched=["ACC-104"],
                      tool_log=[{"name": "get_kpis", "input": {"account_id": "ACC-104"}, "ok": True,
                                 "accounts": ["ACC-104"], "result_chars": 12, "result_preview": "{}"}],
                      unsourced_figures=[], usage={"input_tokens": 1, "output_tokens": 2})
    store.set_focus(chat, "ACC-104")
    store.save_chat(chat)

    loaded = store.load_chat("book", "abc123", chat["chat_id"])
    assert loaded["title"].startswith("Why is the margin") and loaded["title"].endswith("…")
    assert loaded["focus_account"] == "ACC-104" and loaded["scope"] == "book"
    assert [t["role"] for t in loaded["turns"]] == ["user", "assistant"]
    assert loaded["turns"][1]["tool_log"][0]["result_preview"] == "{}"
    assert store.list_chats("book", "abc123")[0]["turn_count"] == 2


def test_chats_are_isolated_by_scope_and_fingerprint(store):
    store.new_chat("book", "abc123", "m")
    store.new_chat("book", "abc123", "m")
    store.new_chat("book", "other", "m")
    store.new_chat("ACC-102", "abc123", "m")
    assert len(store.list_chats("book", "abc123")) == 2
    assert len(store.list_chats("book", "other")) == 1
    assert len(store.list_chats("ACC-102", "abc123")) == 1
    assert store.list_chats("nothing", "abc123") == []


def test_summary_marks_which_turns_it_replaces(store):
    chat = store.new_chat("book", "abc123", "m")
    for t in _turns(5):
        store.append_turn(chat, t["role"], t["text"])
    assert len(store.unsummarised_turns(chat)) == 10
    store.apply_summary(chat, "the gist", through=4)
    store.save_chat(chat)
    loaded = store.load_chat("book", "abc123", chat["chat_id"])
    assert loaded["summary"] == "the gist" and loaded["summarised_through"] == 4
    assert [t["text"] for t in store.unsummarised_turns(loaded)][:2] == ["Q2", "A2"]


def test_conversations_group_like_youkti():
    today = date(2026, 9, 7)
    rows = [
        {"title": "A", "updated_at": "2026-09-07T10:00:00"},
        {"title": "B", "updated_at": "2026-09-06T10:00:00"},
        {"title": "C", "updated_at": "2026-09-03T10:00:00"},
        {"title": "D", "updated_at": "2026-08-01T10:00:00"},
    ]
    grouped = dict(chatstore.group_conversations(rows, today=today))
    assert [row["title"] for row in grouped["Today"]] == ["A"]
    assert [row["title"] for row in grouped["Yesterday"]] == ["B"]
    assert [row["title"] for row in grouped["Previous 7 days"]] == ["C"]
    assert [row["title"] for row in grouped["Older"]] == ["D"]


def test_a_dangling_question_can_be_dropped(store):
    chat = store.new_chat("book", "abc123", "m")
    store.append_turn(chat, "user", "Q")
    store.drop_last_turn(chat)
    assert chat["turns"] == []
    store.delete_chat("book", "abc123", chat["chat_id"])
    assert store.load_chat("book", "abc123", chat["chat_id"]) is None
