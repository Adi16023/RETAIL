"""
The AryaChat panel: a manager asks about any account, or about the whole
book, in as many separate chats as they like, each with its own context.

The screen follows the ARYA MCP chat: a sidebar of conversations, a message
thread and a composer. Nothing is pre-selected and nothing is pre-written —
the manager types the question, the model works out which account (if any)
it is about, calls the tool that holds the answer, and answers from it. A
turn is shown as it happens: tool chips appear while a look-up runs and flip
to done when it returns (open one to see what came back), the answer streams
in as the model writes it, and 2-3 next-question chips land under the
finished answer; tapping one sends it verbatim. Chips are ephemeral — they
belong to the latest answer only and are never stored.

The account the chat is currently about is shown above the composer and
carried into the next question, so "and the margin?" needs no repetition.

Every answer shows what it rests on — the tools it drew figures from — and
a figure that appears in none of them is flagged, because that figure came
from nowhere.
"""

from __future__ import annotations

import time

import streamlit as st

from pipeline.aryachat import (
    COMPACT_AFTER_TURNS,
    KEEP_LAST_TURNS,
    AgentEvent,
    ChatError,
    ask,
    clean_text,
    compact_conversation,
)

from . import chatstore

TOOL_LABELS = {
    "find_account": "account look-up",
    "get_account_book": "the account book",
    "get_account_brief": "account brief",
    "compare_accounts": "account comparison",
    "get_orders": "orders",
    "get_product_revenue": "products bought",
    "get_category_mix": "category mix",
    "get_category_monthly_series": "category by month",
    "get_category_seasonal_breakdown": "seasonal breakdown",
    "get_product_changes": "product changes",
    "get_monthly_economics": "monthly economics",
    "get_tier_monthly_series": "tier mix by month",
    "get_monthly_table": "monthly table",
    "get_kpis": "headline tiles",
    "get_ruled_out_checks": "harmless-explanation checks",
    "get_priced_options": "priced options",
}

# Re-render the streaming bubble at most this often. Groq writes several
# hundred tokens a second; a frame per token is a websocket message per
# token for nothing the eye can follow.
STREAM_FRAME_SECONDS = 0.05

SCOPE = "book"


def _label(tool: str) -> str:
    return TOOL_LABELS.get(tool, tool.replace("_", " "))


def _state_key(fingerprint: str) -> str:
    return f"aryachat_active_{fingerprint}"


def _chips_key(key: str) -> str:
    return f"{key}-chips"


def _account_label(account_id: str | None, names: dict) -> str:
    if not account_id:
        return ""
    return f"{account_id} · {names[account_id]}" if names.get(account_id) else account_id


# --- Rendering a stored turn -------------------------------------------------------------

def _describe_arguments(arguments: dict) -> str:
    parts = []
    for key, value in (arguments or {}).items():
        if value in (None, "", [], {}):
            continue
        parts.append(f"{key} = {', '.join(map(str, value)) if isinstance(value, list) else value}")
    return ", ".join(parts)


def _render_tool_log(log: list[dict], unsourced: list[str]) -> None:
    """One collapsed row per turn — "Based on: …" — that opens to show each
    call's arguments and a preview of what it returned. The figure check's
    findings live in here too, as a quiet line for whoever opens it — never
    as a warning in the thread, which read as an alarm on answers that were
    fine apart from a rounded aside."""
    names = list(dict.fromkeys(entry["name"] for entry in log))
    with st.expander("Based on: " + ", ".join(_label(n) for n in names)):
        if unsourced:
            st.caption("Figures the analyst stated that no look-up returned: " + ", ".join(unsourced))
        for entry in log:
            status = "returned" if entry.get("ok") else "failed"
            chars = entry.get("result_chars") or 0
            st.markdown(f"**{_label(entry['name'])}** · {status}" + (f" · {chars:,} characters" if chars else ""))
            asked = _describe_arguments(entry.get("input") or {})
            if asked:
                st.caption("Asked for: " + asked)
            if entry.get("error"):
                st.caption(f"Error: {entry['error']}")
            if entry.get("result_preview"):
                st.code(entry["result_preview"], language="json", wrap_lines=True)


def _render_turn(turn: dict, names: dict) -> None:
    role = "user" if turn.get("role") == "user" else "assistant"
    with st.chat_message(role):
        st.markdown(turn.get("text", ""))
        if role != "assistant":
            return
        # Chats saved by the earlier, structured version of this loop carried
        # a "what the data cannot answer" field; still shown if present.
        gap = clean_text(turn.get("cannot_answer_because"))
        if gap:
            st.caption(f"⚠ Not in the data: {gap}")
        log = turn.get("tool_log") or []
        if log:
            _render_tool_log(log, turn.get("unsourced_figures") or [])
        elif turn.get("tools_used") is not None:
            st.caption("Answered from the standing context, no look-up needed")
        if turn.get("truncated"):
            st.caption("Stopped at the step limit — this is what it had found by then.")


# --- The live turn ----------------------------------------------------------------------

class _LiveTurn:
    """Turns loop events into screen updates while a question is answered.

    Tool chips go in a container above the answer so the order on screen is
    the order things happened: look something up, then write. Text is
    accumulated and re-rendered with a cursor; a `retry` clears it, because
    the draft it held was refused and a fresh one is about to stream.
    """

    def __init__(self, activity, slot):
        self._activity = activity
        self._slot = slot
        self._buffer: list[str] = []
        self._statuses: dict[str, object] = {}
        self._last_frame = 0.0
        self.follow_ups: list[str] = []
        self._slot.caption("Thinking…")

    def __call__(self, event: AgentEvent) -> None:
        if event.type == "text":
            self._buffer.append(event.text)
            self._render(final=False)
        elif event.type == "tool_start":
            status = self._activity.status(f"Looking up {_label(event.tool_name)}…", state="running")
            asked = _describe_arguments(event.arguments)
            if asked:
                status.caption("Asked for: " + asked)
            self._statuses[event.tool_id] = status
        elif event.type == "tool_end":
            status = self._statuses.get(event.tool_id)
            if status is not None:
                label = _label(event.tool_name)
                if event.ok:
                    status.update(label=f"{label} · {event.result_chars:,} characters", state="complete")
                    if event.result_preview:
                        status.code(event.result_preview, language="json", wrap_lines=True)
                else:
                    status.update(label=f"{label} · failed", state="error")
        elif event.type == "retry":
            self._buffer.clear()
            self._slot.caption("Checking the detail first…")
        elif event.type == "follow_ups":
            self.follow_ups = list(event.follow_ups)

    def _render(self, final: bool) -> None:
        now = time.monotonic()
        if not final and now - self._last_frame < STREAM_FRAME_SECONDS:
            return
        self._last_frame = now
        text = "".join(self._buffer)
        self._slot.markdown(text if final else text + " ▌")

    def finish(self, text: str) -> None:
        self._buffer = [text]
        self._render(final=True)


# --- Sidebar ------------------------------------------------------------------------------

def _chat_list(fingerprint: str, model_id: str, key: str, initial_focus: str | None, names: dict) -> None:
    if st.button("New chat", icon=":material/add:", type="primary", width="stretch", key=f"{key}-new"):
        chat = chatstore.new_chat(SCOPE, fingerprint, model_id, focus_account=initial_focus)
        st.session_state[key] = chat["chat_id"]
        st.session_state.pop(_chips_key(key), None)
        st.rerun()
    chats = chatstore.list_chats(SCOPE, fingerprint)
    if not chats:
        st.caption("No chats yet.")
        return
    active_id = st.session_state.get(key)
    for row in chats:
        focus = _account_label(row.get("focus_account"), names)
        st.button(
            row["title"],
            key=f"{key}-open-{row['chat_id']}",
            type="primary" if row["chat_id"] == active_id else "secondary",
            width="stretch",
            help=f"{row['turn_count']} turns · last {row['updated_at'][:16].replace('T', ' ')}"
                 + (f" · about {focus}" if focus else ""),
            on_click=lambda k=key, c=row["chat_id"]: st.session_state.__setitem__(k, c),
        )
    if active_id and any(row["chat_id"] == active_id for row in chats):
        if st.button("Delete this chat", icon=":material/delete:", type="tertiary",
                     key=f"{key}-delete", help="Removes this chat and its context. Cannot be undone."):
            chatstore.delete_chat(SCOPE, fingerprint, active_id)
            st.session_state.pop(key, None)
            st.session_state.pop(_chips_key(key), None)
            st.rerun()


def _compact(chat: dict, make_client, model_id: str) -> str | None:
    """Fold everything except the last KEEP_LAST_TURNS into the summary."""
    turns = chat.get("turns") or []
    through = len(turns) - KEEP_LAST_TURNS
    start = chat.get("summarised_through", 0)
    if through <= start:
        return None
    client = make_client()
    summary = compact_conversation(client, turns[start:through], chat.get("summary"), model=model_id)
    chatstore.apply_summary(chat, summary, through)
    chatstore.save_chat(chat)
    return summary


# --- Next-question chips ----------------------------------------------------------------
#
# Kept in session state, never on disk: they belong to the latest answer of
# one chat and are consumed the moment one is tapped. Keyed by the turn count
# they were produced at, so a chip from an earlier answer can never surface
# under a later one.

def _remember_chips(key: str, chat: dict, chips: list[str]) -> None:
    if chips:
        st.session_state[_chips_key(key)] = {
            "chat_id": chat["chat_id"], "after_turns": len(chat.get("turns") or []), "chips": chips,
        }
    else:
        st.session_state.pop(_chips_key(key), None)


def _chips_for(key: str, chat: dict) -> list[str]:
    held = st.session_state.get(_chips_key(key)) or {}
    if held.get("chat_id") != chat["chat_id"] or held.get("after_turns") != len(chat.get("turns") or []):
        return []
    return list(held.get("chips") or [])


def _render_chips(key: str, chat: dict) -> str | None:
    """The chips under the latest answer. Returns the tapped one, if any —
    and forgets the set, so a rerun cannot send it twice."""
    chips = _chips_for(key, chat)
    if not chips:
        return None
    picked = st.pills(
        "Ask next", chips, selection_mode="single", label_visibility="collapsed",
        key=f"{key}-pills-{chat['chat_id']}-{len(chat.get('turns') or [])}",
    )
    if picked:
        st.session_state.pop(_chips_key(key), None)
    return picked


# --- The panel ------------------------------------------------------------------------------

def render_aryachat(df, fingerprint: str, model_id: str, choice_label: str, make_client,
                    pack_for, report_for, names: dict, initial_focus: str | None = None) -> None:
    """`pack_for(account_id)` / `report_for(account_id)` reach the pipeline's
    cached outputs; `initial_focus` seeds a new chat with the account the
    manager has open elsewhere in the app, if any."""
    key = _state_key(fingerprint)

    side, main = st.columns([1, 3], gap="large")
    with side:
        st.markdown("**Chats**")
        _chat_list(fingerprint, model_id, key, initial_focus, names)

    with main:
        chat_id = st.session_state.get(key)
        chat = chatstore.load_chat(SCOPE, fingerprint, chat_id) if chat_id else None
        if chat is None:
            st.info("Start a new chat, or pick one on the left.")
            return

        turns = chat.get("turns") or []
        for turn in turns:
            _render_turn(turn, names)
        tapped = _render_chips(key, chat)

        # No context line (focus, turns in context, tokens) — removed at the
        # user's request; it is plumbing, and the focus is visible in the chat
        # list's hover text. Compaction stays automatic and silent, and the
        # summary it produces stays readable behind one collapsed row.
        if chat.get("summary"):
            with st.expander("What the earlier turns established"):
                st.markdown(chat["summary"])

        question = st.chat_input("Ask about an account, or about the book…", key=f"{key}-input")
        question = question or tapped
        if not question:
            return

        # Auto-compact when the un-summarised tail has grown past the threshold.
        if len(chatstore.unsummarised_turns(chat)) > COMPACT_AFTER_TURNS:
            try:
                _compact(chat, make_client, model_id)
            except Exception as e:
                st.warning(f"Could not compact the earlier turns ({e}); answering with the recent ones only.")

        chatstore.append_turn(chat, "user", question)
        chatstore.save_chat(chat)
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            activity = st.container()
            slot = st.empty()
            live_turn = _LiveTurn(activity, slot)
            try:
                client = make_client()
                answer = ask(
                    client, df, question,
                    pack_for=pack_for, report_for=report_for,
                    focus_account=chat.get("focus_account"),
                    turns=chat["turns"][chat.get("summarised_through", 0):-1],
                    summary=chat.get("summary"), model=model_id,
                    on_event=live_turn,
                )
            except ChatError as e:
                error = f"The model did not answer: {e}"
            except Exception as e:  # provider errors surface as text, never a crash
                error = f"Could not reach the model: {e}"
            else:
                error = None
            if error:
                slot.error(error)
                chatstore.drop_last_turn(chat)
                chatstore.save_chat(chat)
                return
            live_turn.finish(answer["answer"])
            chatstore.append_turn(
                chat, "assistant", answer["answer"], model=model_id,
                tools_used=answer.get("tools_used") or [],
                tool_log=answer.get("tool_log") or [],
                accounts_touched=answer.get("accounts_touched") or [],
                unsourced_figures=answer.get("unsourced_figures") or [],
                usage=answer.get("usage") or {},
                truncated=bool(answer.get("truncated")),
            )
            chatstore.set_focus(chat, answer.get("focus_account"))
            chatstore.save_chat(chat)
            _remember_chips(key, chat, answer.get("follow_ups") or [])
        st.rerun()
