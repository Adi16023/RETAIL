"""
The cross-account comparison panel.

Renders what `pipeline.compare` returned and nothing more. Like every module
in this package it presents figures; it never computes one — and here that
matters more than usual, because the comparison's whole claim is that it
only re-reads analysis the per-account stages already produced. A total or
an average invented at render time would quietly break that promise.

The panel leads with the groups rather than the per-account lines: the
reason to read eighteen accounts together is to see which of them are
living the same story, and a flat list of eighteen one-liners is just the
single-account view eighteen times over.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from .palette import active

# How a group bears on the book. Mirrors the timeline's reads_as vocabulary,
# plus the case that only exists across accounts: one whose history is too
# short to be judged alongside the rest.
_GROUP_STYLE = {
    "concern": ("🔴", "critical"),
    "reassuring": ("🟢", "good"),
    "cannot_judge_yet": ("🟡", "warning"),
    "context": ("⚪", "muted"),
}

_GROUP_ORDER = ["concern", "cannot_judge_yet", "reassuring", "context"]


def _name_lookup(digest: dict) -> dict:
    """account_id -> "ACC-107 - Ashford Retail", from the digest the model was
    given. Taken from there rather than from the model's reply so a label can
    never be something the model invented."""
    return {
        a["account_id"]: (
            f"{a['account_id']} · {a['account_name']}" if a.get("account_name") else a["account_id"]
        )
        for a in digest.get("accounts", [])
    }


def render_comparison(result: dict, digest: dict) -> None:
    """One finished comparison."""
    colors = active()
    names = _name_lookup(digest)
    label = lambda account_id: names.get(account_id, account_id)  # noqa: E731

    if result.get("headline"):
        st.markdown(f"#### {result['headline']}")

    groups = result.get("groups") or []
    if groups:
        st.markdown("**Accounts telling the same story**")
        ordered = sorted(
            groups,
            key=lambda g: _GROUP_ORDER.index(g.get("reads_as"))
            if g.get("reads_as") in _GROUP_ORDER else len(_GROUP_ORDER),
        )
        for group in ordered:
            icon, role = _GROUP_STYLE.get(group.get("reads_as"), ("⚪", "muted"))
            members = group.get("account_ids") or []
            with st.container(border=True):
                st.markdown(
                    f"{icon} **{group.get('story', '')}** "
                    f"<span style='color:{colors['muted']}'>· {len(members)} account"
                    f"{'s' if len(members) != 1 else ''}</span>",
                    unsafe_allow_html=True,
                )
                st.caption(group.get("what_they_share", ""))
                st.markdown("  \n".join(f"· {label(a)}" for a in members))

    standouts = result.get("standouts") or []
    if standouts:
        st.markdown("**The ones that break the pattern**")
        for item in standouts:
            st.info(f"**{label(item['account_id'])}** — {item['why_it_stands_out']}")

    focus = result.get("where_to_look_first") or []
    st.markdown("**Where to look first**")
    if focus:
        for rank, item in enumerate(focus, start=1):
            with st.container(border=True):
                st.markdown(f"**{rank}. {label(item['account_id'])}**")
                st.write(item.get("reason", ""))
                st.caption(f"Suggested: {item.get('suggested_action', '')}")
    else:
        st.success(
            "Nothing in this selection needs attention right now — no account here is "
            "losing value in a way that calls for a conversation."
        )

    per_account = result.get("per_account") or []
    if per_account:
        with st.expander(f"One line on each of the {len(per_account)} accounts"):
            st.dataframe(
                pd.DataFrame([
                    {"Account": label(row["account_id"]), "What's happening": row["business_read"]}
                    for row in per_account
                ]),
                width="stretch",
                hide_index=True,
            )

    st.caption(
        "Read from the same dated findings as the single-account view above — no new figures "
        "were computed for this comparison, and the accounts' own verdicts are unchanged by it."
    )
