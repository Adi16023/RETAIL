"""
The intervention panel: what this leak would be worth fixing, and what to do.

Two halves, and the order matters. The priced options render first and need
no model, no key and no network — a manager can size every option and make
the decision themselves. The written recommendation is a separate, explicit
button on top of that. If the model call fails, the numbers above it are
untouched and still correct.

The figure that does the work here is the break-even. "Recovers Rs 21,794 a
month" is useful; "you could lose 48% of this account's volume before that
stops paying" is what actually settles whether a manager is willing to have
an awkward conversation at renewal. It is deliberately labelled as a
tolerance rather than a prediction — nothing in a transaction history
supports a forecast of how a customer reacts to a price change, and implying
otherwise would be exactly the false confidence this project exists to avoid.
"""

from __future__ import annotations

from html import escape

import pandas as pd
import streamlit as st

from .palette import active
from . import theme

ACTION_LABELS = {
    "price_recovery": "Reset the price",
    "mix_recovery": "Win back the premium lines",
    "win_back": "Win back lost business",
    "consolidation": "Consolidate the ordering",
    "relationship_review": "Review the relationship",
    "investigate_cause": "Establish the cause first",
    "expansion": "Expansion opportunity",
    "no_action": "No action needed",
    "gather_data": "Gather more data first",
}


def _money(value) -> str:
    return "—" if value is None else f"₹{value:,.0f}"


LEVER_INTRO = {
    "price_recovery": (
        "The money is leaving through the price. Changing price does not change what the goods "
        "cost, so the whole movement lands in margin — which is why winding the discount back to "
        "where it started reproduces the margin rate this account used to earn."
    ),
    "win_back": (
        "A line has stopped completely. This is sized from what it was actually worth per month "
        "before it stopped, valued at the margin this account earns today."
    ),
    "mix_recovery": (
        "Spend has slid into cheaper lines. Priced as closing part of the gap between the margin "
        "rate this account used to earn and the one it earns now, at today's revenue."
    ),
    "margin_recovery": (
        "Margin has fallen without discounting moving, so the cause sits in cost or mix. Priced "
        "as closing part of the gap back to the margin rate this account used to earn."
    ),
    "revenue_recovery": (
        "Spend has fallen across the book with no single line responsible. Priced as closing part "
        "of the gap between what this account used to spend each month and what it spends now."
    ),
}


def _outcome_column(lever: str) -> tuple[str, str]:
    """(column heading, help) for the figure that decides this lever."""
    if lever == "price_recovery":
        return "Volume you could lose", (
            "How much volume could be lost before the move stops being worth making. "
            "A tolerance for being wrong, not a prediction."
        )
    if lever == "win_back":
        return "Room to concede", (
            "Discounting cuts price without cutting cost, so margin on won-back business runs "
            "out when the concession reaches the margin rate."
        )
    return "Margin rate after", "Where the margin rate lands if this is achieved."


def render_options(options: list[dict], colors: dict) -> list[dict]:
    """Every option this account has, side by side — what you'd do, and what
    it gets you.

    No slider. The reader is not here to run scenarios; they are here to be
    told what to do. Showing the options together is what lets the
    recommendation below be read as a choice among real alternatives rather
    than as an assertion, and it is what makes "we are taking part of the
    value, deliberately" legible.
    """
    lever = options[0]["lever"]
    st.markdown("**What could be done, and what each would be worth**")
    st.caption(LEVER_INTRO.get(lever, ""))

    heading, help_text = _outcome_column(lever)
    rows = []
    for option in options:
        if lever == "price_recovery":
            outcome = f"{option['volume_that_could_be_lost_before_this_stops_paying_pct']}%"
        elif lever == "win_back":
            outcome = f"{option['max_price_concession_pct']}%"
        else:
            outcome = f"{option.get('resulting_margin_rate_pct', '—')}%"
        rows.append({
            "What you'd do": option["option"][0].upper() + option["option"][1:],
            "Margin back / month": _money(option["recovers_margin_per_month"]),
            "Over 12 months": _money(option["recovers_margin_over_12_months"]),
            heading: outcome,
            "Fixes it?": "Yes" if option.get("restores_account_to_healthy") else "Partly",
        })

    st.dataframe(
        pd.DataFrame(rows), use_container_width=True, hide_index=True,
        column_config={heading: st.column_config.TextColumn(help=help_text),
                       "Fixes it?": st.column_config.TextColumn(
                           help="Whether the account still reads as leaking afterwards, on the "
                                "same threshold the analysis itself uses. Recovering part of the "
                                "value is a good outcome — it does not have to be all of it.")},
    )
    if lever == "win_back":
        first = options[0]
        st.caption(
            f"**{', '.join(first['categories'])}** was worth "
            f"**{_money(first['lost_revenue_per_month'])} a month** "
            + (f"(**{first['share_of_account_baseline_pct']}%** of this account) "
               if first.get("share_of_account_baseline_pct") else "")
            + f"and has been gone **{first['months_already_gone']} months** — "
            f"{_money(first['value_lost_so_far'])} so far."
        )
    st.caption(
        "Nothing here predicts what the customer will do. It prices what each move would be "
        "worth if it landed, so the choice below can be made on numbers rather than instinct."
    )
    return options


def render_early_warning(report: dict) -> None:
    """A leak with nothing yet to recover.

    Shrinking baskets and splitting orders mean the account is buying in a
    different shape, not buying less — so there is no loss to size. Inventing
    a recovery figure here is exactly the false precision this project
    exists to avoid, and it is still worth acting on.
    """
    st.info(
        "**Nothing has been lost yet — which is what makes this cheap to act on.** This account "
        "is buying in a different shape rather than buying less, so there is no recovery to "
        "price. The play below is a judgement about where this is heading, not a claim about "
        "money already gone."
    )


def render_decision(decision: dict, colors: dict) -> None:
    """The written recommendation."""
    action = decision.get("action_type", "")
    label = ACTION_LABELS.get(action, action.replace("_", " ").title())
    role = "good" if action in ("no_action", "expansion") else "serious"

    theme.banner(
        role,
        decision.get("headline", "") or label,
        [
            f"<span style='letter-spacing:.08em;text-transform:uppercase;font-size:0.76rem;"
            f"font-weight:650'>{escape(label)}</span>",
            f"Target: {escape(str(decision.get('recommended_option', '')))}",
        ],
    )

    if decision.get("expected_result"):
        st.markdown(f"**What this gets you** — {decision['expected_result']}")

    # The steps come FIRST and outside the tabs. This panel asks "what should
    # we do about it?", and an answer that leads with rationale and hides the
    # work behind a tab label leaves the reader still asking the question.
    steps = decision.get("what_to_do") or []
    if steps:
        st.markdown("**Do this**")
        for number, step in enumerate(steps, start=1):
            st.markdown(f"{number}. {step}")
        st.write("")

    with st.expander("Why this, and not more or less"):
        st.markdown(decision.get("rationale", ""))
        left, right = st.columns(2)
        with left:
            st.markdown("**Why not push harder**")
            st.caption(decision.get("why_not_more_aggressive", "—"))
        with right:
            st.markdown("**Why not settle for less**")
            st.caption(decision.get("why_not_less_aggressive", "—"))

    restores = decision.get("restores_account_to_healthy")
    left_over = decision.get("what_is_left_over")
    if restores is True:
        st.success("**This returns the account to healthy.**")
    elif restores is False and left_over:
        # Information, not a warning: taking part of the value back is a
        # legitimate call, and flagging it as a shortfall would push every
        # recommendation toward a maximum nobody would actually execute.
        st.info(f"**Recovers part of it:** {left_over}")

    talking, checks, follow_up = st.tabs(
        ["What to say", "Check first", "How we'll know it worked"]
    )
    with talking:
        for point in decision.get("talking_points") or []:
            st.markdown(f"- {point}")
        if decision.get("downside_if_wrong"):
            st.warning(f"**If this goes wrong:** {decision['downside_if_wrong']}")
    with checks:
        st.caption(
            "Assumptions the transaction data cannot settle. Verify these before acting, "
            "not afterwards."
        )
        for item in decision.get("check_before_acting") or []:
            st.markdown(f"- {item}")
    with follow_up:
        for item in decision.get("how_we_will_know_it_worked") or []:
            st.markdown(f"- {item}")
        months = decision.get("review_in_months")
        if months:
            st.caption(f"Review in **{months} months**.")

    owner, timing = st.columns(2)
    owner.markdown(f"**Owner**  \n{decision.get('owner', '—')}")
    timing.markdown(f"**Timing**  \n{decision.get('timing', '—')}")

    st.caption(
        "Every figure quoted above was computed by the deterministic stages; the AI chose "
        "among the priced options and made the case for one. It produced no number itself."
    )


def render_no_lever(report: dict) -> None:
    """Said plainly when the discount lever does not fit this account.

    Silence would read as "nothing to do here", which is wrong — a defection
    or a mix slide needs a real intervention, just not this one.
    """
    dimensions = report.get("leak_dimensions") or []
    if report.get("defer"):
        st.info(
            "No intervention is priced for this account — the investigation deferred, and "
            "acting on a verdict nobody was willing to make is worse than waiting. "
            "See **What to do** above for what would settle it."
        )
    elif report.get("verdict") == "healthy":
        st.success("Nothing to fix here — this account is trading normally.")
    else:
        named = ", ".join(d.replace("_", " ") for d in dimensions) or "this account"
        st.info(
            f"The value is leaving through **{named}**, which a discount change would not "
            "address. Pricing an intervention here would answer a question nobody asked — "
            "the recommended actions above are the right starting point."
        )
