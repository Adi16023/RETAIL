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

import streamlit as st

from .palette import active

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


def _extra_metric(option: dict) -> tuple[str, str, str] | None:
    """(label, value, help) for the figure that actually decides this lever."""
    lever = option["lever"]
    if lever == "price_recovery":
        return (
            "Volume you could lose",
            f"{option['volume_that_could_be_lost_before_this_stops_paying_pct']}%",
            "Before the intervention stops being worth doing. A tolerance, not a prediction.",
        )
    if lever == "win_back":
        return (
            "Room to concede on price",
            f"{option['max_price_concession_pct']}%",
            "Discounting cuts price without cutting cost, so margin on won-back business runs "
            "out exactly when the concession reaches the margin rate. Beyond this you are "
            "buying the business at a loss.",
        )
    if lever in ("mix_recovery", "margin_recovery"):
        return (
            "Margin rate after",
            f"{option['resulting_margin_rate_pct']}%",
            f"Up from {option['margin_rate_today_pct']}% today; it was "
            f"{option['margin_rate_before_pct']}% before.",
        )
    return None


def render_options(options: list[dict], colors: dict) -> dict | None:
    """The priced ladder for whichever lever fits this account.

    Deterministic throughout — this is the half that must keep working when
    the model does not.
    """
    st.markdown("**What would it be worth to fix?**")

    levers = []
    for option in options:
        if option["lever"] not in levers:
            levers.append(option["lever"])

    lever = levers[0]
    if len(levers) > 1:
        labels = {
            next(o["lever_label"] for o in options if o["lever"] == name): name for name in levers
        }
        lever = labels[st.radio("Which lever?", list(labels), horizontal=True)]

    scoped = [o for o in options if o["lever"] == lever]
    st.caption(LEVER_INTRO.get(lever, ""))

    labels = {f"{o['share_recovered_pct']}% — {o['option']}": o for o in scoped}
    # A slider rather than a dropdown: the point is to feel the trade-off
    # move, not to pick from a list.
    choice = st.select_slider(
        "How much would you aim to recover?",
        options=list(labels),
        value=list(labels)[len(labels) // 2],
    )
    option = labels[choice]

    columns = st.columns(4)
    if option["recovers_revenue_per_month"]:
        columns[0].metric("Revenue back / month", _money(option["recovers_revenue_per_month"]))
    else:
        columns[0].metric("Share of the gap closed", f"{option['share_recovered_pct']}%")
    columns[1].metric(
        "Margin back / month", _money(option["recovers_margin_per_month"]),
        help="Revenue and margin overlap — margin is a slice of revenue. Never add them.",
    )
    columns[2].metric("Margin over 12 months", _money(option["recovers_margin_over_12_months"]))
    extra = _extra_metric(option)
    if extra:
        columns[3].metric(extra[0], extra[1], help=extra[2])

    restores = option.get("restores_account_to_healthy")
    if restores is True:
        st.success(
            "**This closes the gap.** On the same threshold the analysis itself uses, this "
            "option returns the account to healthy."
        )
    elif restores is False:
        short = (
            f"{option['short_of_healthy_by_pp']}pp of margin"
            if option.get("short_of_healthy_by_pp")
            else f"{_money(option.get('still_short_per_month'))} a month"
        )
        st.warning(
            f"**This helps but does not fix it.** The account still reads as leaking — "
            f"{short} short of healthy. A partial fix can be the right call; just do not "
            "close the case on it."
        )

    if lever == "win_back":
        st.caption(
            f"**{', '.join(option['categories'])}** was worth "
            f"**{_money(option['lost_revenue_per_month'])} a month** "
            + (f"(**{option['share_of_account_baseline_pct']}%** of this account) "
               if option.get("share_of_account_baseline_pct") else "")
            + f"and has been gone **{option['months_already_gone']} months** — "
            f"{_money(option['value_lost_so_far'])} so far. Winning it back with a concession of "
            f"half the available room still returns "
            f"{_money(option['margin_if_won_back_at_half_that_concession'])} a month."
        )
    elif lever == "price_recovery":
        st.caption(
            f"Asking for **{option['reduction_from_today_pp']}pp** off today's discount. This "
            f"account could lose up to "
            f"**{option['volume_that_could_be_lost_before_this_stops_paying_pct']}% of its "
            "volume** and still leave you better off than doing nothing — a tolerance for being "
            "wrong, not a prediction. Nothing in a transaction history says how a buyer reacts "
            "to a price change."
        )
    else:
        st.caption(
            "Recovery is shown as a share of the gap, not as a forecast. Nothing here predicts "
            "whether the customer comes back — it prices what it would be worth if they did."
        )
    return option


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

    st.markdown(
        f"""<div style="border-left:5px solid {colors[role]};background:{colors['band']}55;
        padding:0.8rem 1.05rem;border-radius:6px;margin:0.5rem 0 0.9rem">
        <div style="font-size:0.76rem;color:{colors['muted']};text-transform:uppercase;
        letter-spacing:.04em">{label}</div>
        <div style="font-size:1.1rem;font-weight:640;margin-top:0.15rem">
        {decision.get('recommended_option', '')}</div>
        <div style="font-size:0.93rem;color:{colors['text_secondary']};margin-top:0.4rem">
        {decision.get('headline', '')}</div></div>""",
        unsafe_allow_html=True,
    )

    st.markdown(decision.get("rationale", ""))

    restores = decision.get("restores_account_to_healthy")
    left_over = decision.get("what_is_left_over")
    if restores is False and left_over:
        st.warning(f"**Does not fully fix it:** {left_over}")
    elif restores is True:
        st.success("**This returns the account to healthy.**")

    # The rejected alternatives sit beside the choice, not buried: a
    # recommendation that does not say what it turned down is an assertion.
    left, right = st.columns(2)
    with left:
        st.markdown("**Why not push harder**")
        st.caption(decision.get("why_not_more_aggressive", "—"))
    with right:
        st.markdown("**Why not settle for less**")
        st.caption(decision.get("why_not_less_aggressive", "—"))

    talking, checks, follow_up = st.tabs(
        ["Take this into the room", "Check first", "How we'll know it worked"]
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
