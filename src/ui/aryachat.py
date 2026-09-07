"""
Floating AryaChat widget: a manager asks about any account, or the whole
book, from a bottom-right chatbot that expands to fullscreen on its own.

Closed -> a launcher button. Open -> a docked panel over whatever page they
are on. Fullscreen -> the same chat, edge to edge, still independent of
Detect / Investigate. Recent chats and New chat live inside the widget.

Chrome (FAB, rail, actions, composer, hero) uses streamlit-shadcn-ui so
layout is a real React tree instead of Streamlit CSS flex hacks. The
message thread stays on Streamlit chat_message for streaming tool chips.
"""

from __future__ import annotations

import time
from functools import partial

import streamlit as st
import streamlit_shadcn_ui as ui

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

STREAM_FRAME_SECONDS = 0.05
SCOPE = "book"


def _label(tool: str) -> str:
    return TOOL_LABELS.get(tool, tool.replace("_", " "))


def _state_key(fingerprint: str) -> str:
    return f"aryachat_active_{fingerprint}"


def _chips_key(key: str) -> str:
    return f"{key}-chips"


def _describe_arguments(arguments: dict) -> str:
    parts = []
    for key, value in (arguments or {}).items():
        if value in (None, "", [], {}):
            continue
        parts.append(f"{key} = {', '.join(map(str, value)) if isinstance(value, list) else value}")
    return ", ".join(parts)


def _render_tool_log(log: list[dict], unsourced: list[str]) -> None:
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
        gap = clean_text(turn.get("cannot_answer_because"))
        if gap:
            st.caption(f"Not in the data: {gap}")
        log = turn.get("tool_log") or []
        if log:
            _render_tool_log(log, turn.get("unsourced_figures") or [])
        elif turn.get("tools_used") is not None:
            st.caption("Answered from the standing context, no look-up needed")
        if turn.get("truncated"):
            st.caption("Stopped at the step limit — this is what it had found by then.")


class _LiveTurn:
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


def _consume_nav(key: str) -> None:
    if "delete_chat" in st.query_params:
        st.session_state["arya_delete_id"] = st.query_params["delete_chat"]
        del st.query_params["delete_chat"]
    if "chat" not in st.query_params:
        return
    requested = st.query_params["chat"]
    del st.query_params["chat"]
    st.session_state["arya_open"] = True
    if requested == "new":
        st.session_state[key] = None
        st.session_state.pop(_chips_key(key), None)
        return
    st.session_state[key] = requested


def _confirm_delete(fingerprint: str, key: str) -> None:
    chat_id = st.session_state.get("arya_delete_id")
    if not chat_id:
        return
    decision = ui.alert_dialog(
        show=True,
        title="Delete this chat?",
        description="This conversation and its messages will be removed. This can't be undone.",
        confirm_label="Delete",
        cancel_label="Cancel",
        key="arya-delete-dialog",
    )
    if decision is True:
        chatstore.delete_chat(SCOPE, fingerprint, chat_id)
        if st.session_state.get(key) == chat_id:
            st.session_state.pop(key, None)
        st.session_state.pop(_chips_key(key), None)
        st.session_state.pop("arya_delete_id", None)
        st.rerun()
    if decision is False:
        st.session_state.pop("arya_delete_id", None)


def sync_chat_nav(fingerprint: str) -> None:
    """Apply URL chat selection / delete before the page body renders."""
    key = _state_key(fingerprint)
    if st.session_state.pop("arya_force_idle", False):
        st.session_state[key] = None
        st.session_state.pop(_chips_key(key), None)
        st.session_state["arya_open"] = True
    _consume_nav(key)
    _confirm_delete(fingerprint, key)


def _compact(chat: dict, make_client, model_id: str) -> str | None:
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


def _open_widget() -> None:
    st.session_state["arya_open"] = True


def _close_widget() -> None:
    st.session_state["arya_open"] = False
    st.session_state["arya_fullscreen"] = False
    st.session_state.pop("arya_pending_question", None)


def _collapse_fullscreen() -> None:
    """Leave fullscreen and return to the docked panel."""
    st.session_state["arya_fullscreen"] = False
    st.session_state["arya_open"] = True


def _toggle_fullscreen() -> None:
    st.session_state["arya_fullscreen"] = not st.session_state.get("arya_fullscreen", False)
    if st.session_state["arya_fullscreen"]:
        st.session_state["arya_open"] = True


def _start_new_chat(key: str) -> None:
    st.session_state[key] = None
    st.session_state.pop(_chips_key(key), None)
    st.session_state.pop("arya_pending_question", None)
    st.session_state["arya_open"] = True


def _select_chat(key: str, chat_id: str) -> None:
    st.session_state[key] = chat_id
    st.session_state["arya_open"] = True


def _mark_delete(chat_id: str) -> None:
    st.session_state["arya_delete_id"] = chat_id


def _render_left_rail(chats: list[dict], *, key: str, active_id: str | None, fullscreen: bool) -> None:
    """History scrolls above; New chat + exit/close are pinned to the rail bottom."""
    with st.container(key="arya_left_scroll"):
        with ui.elements(key=f"arya-chats-{key}-{'full' if fullscreen else 'dock'}") as el:
            el.heading("Recent chats", level=4)
            with el.stack(key="chat-list", gap="xs"):
                if not chats:
                    el.text("No chats yet", variant="muted")
                else:
                    for group_label, items in chatstore.group_conversations(chats):
                        el.text(group_label, variant="caption")
                        for row in items:
                            cid = row["chat_id"]
                            title = (row.get("title") or "Chat")[:48]
                            with el.stack(
                                key=f"row-{cid}",
                                direction="horizontal",
                                gap="xs",
                                align="center",
                            ):
                                el.button(
                                    title,
                                    key=f"open-{cid}",
                                    variant="secondary" if cid == active_id else "ghost",
                                    stretch=True,
                                    size="sm",
                                    on_click=partial(_select_chat, key, cid),
                                )
                                el.button(
                                    "×",
                                    key=f"del-{cid}",
                                    variant="ghost",
                                    size="icon-sm",
                                    help="Delete chat",
                                    on_click=partial(_mark_delete, cid),
                                )

    # Bottom of the sidebar — always, regardless of shadcn flex quirks.
    with st.container(key="arya_left_foot"):
        st.button(
            "New chat",
            key="arya-rail-new",
            type="primary",
            use_container_width=True,
            on_click=_start_new_chat,
            args=(key,),
        )
        if fullscreen:
            st.button(
                "Exit fullscreen",
                key="arya-rail-exit-full",
                use_container_width=True,
                on_click=_collapse_fullscreen,
            )
        else:
            st.button(
                "Fullscreen",
                key="arya-rail-full",
                use_container_width=True,
                on_click=_toggle_fullscreen,
            )
            st.button(
                "Close",
                key="arya-rail-close",
                use_container_width=True,
                on_click=_close_widget,
            )


def _render_composer(key: str, turn_count: int) -> str | None:
    """Native Streamlit form — reliable send button, no missing chrome."""
    with st.form(key=f"{key}-composer-{turn_count}", clear_on_submit=True, border=False):
        c1, c2 = st.columns([8, 1], gap="small")
        with c1:
            typed = st.text_input(
                "Message",
                placeholder="Ask about an account, or about the book…",
                label_visibility="collapsed",
                key=f"{key}-typed-{turn_count}",
            )
        with c2:
            sent = st.form_submit_button("↑", type="primary", use_container_width=True)
    if sent:
        text = (typed or "").strip()
        return text or None
    return None


def _queue_question(question: str) -> None:
    """Defer the model call to the next fragment paint so it runs in the thread."""
    text = (question or "").strip()
    if not text:
        return
    st.session_state["arya_pending_question"] = text
    st.rerun(scope="fragment")


def _answer_question(
    question: str,
    *,
    chat: dict | None,
    key: str,
    fingerprint: str,
    model_id: str,
    make_client,
    df,
    pack_for,
    report_for,
    initial_focus: str | None,
) -> None:
    """Stream the reply inside the thread pane; always ends with a fragment rerun."""
    st.session_state.pop("arya_pending_question", None)

    if chat is None:
        chat = chatstore.new_chat(SCOPE, fingerprint, model_id, focus_account=initial_focus)
        st.session_state[key] = chat["chat_id"]

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
        with st.spinner("Looking that up…"):
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
            except Exception as e:
                error = f"Could not reach the model: {e}"
            else:
                error = None
        if error:
            chatstore.drop_last_turn(chat)
            chatstore.save_chat(chat)
            st.session_state["arya_flash_error"] = error
            st.rerun(scope="fragment")
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
    st.rerun(scope="fragment")


@st.fragment
def _arya_fragment(
    df, fingerprint: str, model_id: str, make_client,
    pack_for, report_for, names: dict, initial_focus: str | None,
) -> None:
    """Chat chrome reruns alone so Fullscreen / Close / New chat do not reload the app."""
    if "arya_open" not in st.session_state:
        st.session_state["arya_open"] = False
    if "arya_fullscreen" not in st.session_state:
        st.session_state["arya_fullscreen"] = False

    key = _state_key(fingerprint)
    chat_id = st.session_state.get(key)
    chat = chatstore.load_chat(SCOPE, fingerprint, chat_id) if chat_id else None
    if chat is not None and not (chat.get("turns") or []):
        chat = None
        st.session_state[key] = None
    pending = (st.session_state.get("arya_pending_question") or "").strip() or None
    idle = chat is None and not pending
    open_ = bool(st.session_state["arya_open"])
    fullscreen = bool(st.session_state["arya_fullscreen"]) and open_

    if not open_:
        with st.container(key="arya_fab"):
            if st.button("Ask Arya", key="arya-fab-open", type="primary", on_click=_open_widget):
                pass
        return

    shell_key = "arya_full" if fullscreen else "arya_dock"
    chats = [
        row for row in chatstore.list_chats(SCOPE, fingerprint)
        if row["turn_count"] > 0
    ]

    with st.container(key=shell_key):
        # st.columns — Streamlit's real side-by-side primitive. CSS forces
        # nowrap so the right pane (title + chat bar) cannot collapse away.
        left, right = st.columns([1, 2.6], gap="small")
        with left:
            with st.container(key="arya_left"):
                _render_left_rail(
                    chats,
                    key=key,
                    active_id=st.session_state.get(key),
                    fullscreen=fullscreen,
                )

        with right:
            with st.container(key="arya_right"):
                if fullscreen:
                    tcol, xcol = st.columns([12, 1], gap="small")
                    with tcol:
                        st.markdown(
                            '<p class="rl-arya-widget-title">Arya</p>',
                            unsafe_allow_html=True,
                        )
                    with xcol:
                        st.button(
                            "×",
                            key="arya-full-collapse",
                            help="Back to chat panel",
                            on_click=_collapse_fullscreen,
                            use_container_width=True,
                        )
                else:
                    st.markdown(
                        '<p class="rl-arya-widget-title">Arya</p>',
                        unsafe_allow_html=True,
                    )

                # Thread first, composer last — messages stay above the chat bar.
                with st.container(key="arya_thread"):
                    flash = st.session_state.pop("arya_flash_error", None)
                    if flash:
                        st.error(flash)
                    if idle:
                        st.markdown(
                            '<div class="rl-arya-hero">'
                            '<p class="rl-arya-hero-title">How can I help with your accounts?</p>'
                            '<p class="rl-arya-hero-sub">Ask about one account, or about the book.</p>'
                            "</div>",
                            unsafe_allow_html=True,
                        )
                    else:
                        if chat and chat.get("summary"):
                            with st.expander("What the earlier turns established"):
                                st.markdown(chat["summary"])
                        for turn in (chat.get("turns") or []) if chat else []:
                            _render_turn(turn, names)
                        if pending:
                            _answer_question(
                                pending,
                                chat=chat,
                                key=key,
                                fingerprint=fingerprint,
                                model_id=model_id,
                                make_client=make_client,
                                df=df,
                                pack_for=pack_for,
                                report_for=report_for,
                                initial_focus=initial_focus,
                            )
                        elif chat:
                            tapped = _render_chips(key, chat)
                            if tapped:
                                _queue_question(tapped)

                with st.container(key="arya_composer"):
                    turn_count = len(chat.get("turns") or []) if chat else 0
                    typed = _render_composer(key, turn_count)
                    if typed and not pending:
                        _queue_question(typed)


def render_aryachat(df, fingerprint: str, model_id: str, choice_label: str, make_client,
                    pack_for, report_for, names: dict, initial_focus: str | None = None) -> None:
    """Floating chatbot over the current page. Independent of Detect / Investigate."""
    _arya_fragment(
        df, fingerprint, model_id, make_client,
        pack_for, report_for, names, initial_focus,
    )
