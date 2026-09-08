"""
The Lifetime value page: what an account is expected to be worth over the
next 12 and 24 months, on its baseline behaviour and on its current path,
and the gap between the two.

With an account open: four tiles (baseline path, current path, value at
risk, value per month now versus before — the number that drives the gap),
the two paths as a chart month by month, and the horizon table. The dated
reasons, the priced moves and the rank/backtest lines were shown for a day
and removed at the user's request on Sept 8; the pipeline still computes
them (`book_position`, `lifetime_effect_of_options`, `backtest`) for the
chat tool and the pass-check script.

With no account open: the whole book ranked by value at risk, as a
click-through table.

Every figure comes from `pipeline.lifetime`; nothing is computed here.
"""

from __future__ import annotations

from html import escape
from urllib.parse import quote

import altair as alt
import pandas as pd
import streamlit as st

from . import theme
from .palette import active

HORIZON_SHOWN = "24"
PRODUCTS_SHOWN = 10


def _breakdown_table(rows: list[dict], label: str, basis: str) -> None:
    frame = pd.DataFrame([{
        label: r["name"],
        f"{basis.capitalize()}/month baseline": r["value_per_month_baseline"],
        f"{basis.capitalize()}/month recent": r["value_per_month_recent"],
        "Baseline path": r["cltv_baseline"],
        "Current path": r["cltv_current"],
        "Value at risk": r["value_at_risk"],
    } for r in rows])
    st.dataframe(
        frame, width="stretch", hide_index=True,
        column_config={
            f"{basis.capitalize()}/month baseline": st.column_config.NumberColumn(format="₹%.0f"),
            f"{basis.capitalize()}/month recent": st.column_config.NumberColumn(format="₹%.0f"),
            "Baseline path": st.column_config.NumberColumn(format="₹%.0f"),
            "Current path": st.column_config.NumberColumn(format="₹%.0f"),
            "Value at risk": st.column_config.NumberColumn(format="₹%.0f"),
        },
    )


def _money(value) -> str:
    return "—" if value is None else f"₹{value:,.0f}"


def _pct(value) -> str:
    return "—" if value is None else f"{value * 100:.0f}%"


POTENTIAL, REALITY = "Potential", "Reality"


def _paths_chart(block: dict, colors: dict) -> alt.Chart:
    """Potential versus reality, cumulative, history then projection.

    Both lines are the account's real money until the deviation month, so
    they lie on top of each other; potential then keeps the baseline rate
    (blue) while reality follows what happened and, past today, the recent
    rate (red). History is solid, projection dashed, with a rule at today
    and one at the month the paths part."""
    rows = []
    for point in block["value_paths"]:
        for path, key in ((POTENTIAL, "potential"), (REALITY, "reality")):
            rows.append({"month": f"{point['month']}-01", "path": path, "phase": point["phase"],
                         "value": point[key]})
            # Today belongs to both segments, so the dashed projection starts
            # where the solid history ends instead of a month later.
            if point["offset"] == 0:
                rows.append({**rows[-1], "phase": "projection"})
    frame = pd.DataFrame(rows)
    frame["month"] = pd.to_datetime(frame["month"])
    scale = alt.Scale(domain=[POTENTIAL, REALITY], range=[colors["series_1"], colors["critical"]])
    base = alt.Chart(frame).encode(
        x=alt.X("month:T", title=None, axis=alt.Axis(format="%b %Y", labelAngle=0, tickCount="year")),
        y=alt.Y("value:Q", title="Cumulative value (₹)", axis=alt.Axis(format="~s")),
        color=alt.Color("path:N", scale=scale, legend=alt.Legend(title=None, orient="top-left")),
        strokeDash=alt.StrokeDash(
            "phase:N", scale=alt.Scale(domain=["history", "projection"], range=[[1, 0], [6, 4]]),
            legend=None,
        ),
        tooltip=[alt.Tooltip("month:T", title="Month", format="%b %Y"), alt.Tooltip("path:N", title="Path"),
                 alt.Tooltip("phase:N", title="Phase"),
                 alt.Tooltip("value:Q", title="Cumulative ₹", format=",.0f")],
    )
    lines = base.mark_line(strokeWidth=2)

    history = [p for p in block["value_paths"] if p["phase"] == "history"]
    marks = [{"month": f"{history[-1]['month']}-01", "label": "Today"}]
    if block.get("deviation_month"):
        marks.append({"month": f"{block['deviation_month']}-01", "label": "Paths part"})
    marks_frame = pd.DataFrame(marks)
    marks_frame["month"] = pd.to_datetime(marks_frame["month"])
    rules = alt.Chart(marks_frame).mark_rule(color=colors["muted"], strokeDash=[2, 2]).encode(x="month:T")
    labels = alt.Chart(marks_frame).mark_text(
        align="left", dx=4, dy=-4, baseline="top", color=colors["muted"], fontSize=11,
    ).encode(x="month:T", y=alt.value(0), text="label:N")
    return (lines + rules + labels).properties(height=280)


def render_lifetime_account(pack: dict) -> None:
    block = pack.get("_lifetime_value") or {}
    if block.get("status") != "scored":
        st.info(
            "Not enough history to project a lifetime value for this account. "
            + str(block.get("reason") or "")
        )
        return

    colors = active()
    horizons = block["horizons_months"]
    shown = horizons[HORIZON_SHOWN]
    basis = "margin" if block.get("value_basis") == "margin" else "revenue"
    risk = shown["value_at_risk"]
    change = shown["value_change"]
    before, now = block["value_per_month_baseline"], block["value_per_month_recent"]
    momentum = (now - before) / before if before else None

    theme.kpi_row([
        {"label": f"Lifetime value, {HORIZON_SHOWN} months, baseline path", "value": _money(shown["cltv_baseline"]),
         "delta": f"{basis} per month {_money(before)}", "tone": "neutral",
         "hint": f"{HORIZON_SHOWN} months at the value per month the account earned before the recent window."},
        {"label": f"Lifetime value, {HORIZON_SHOWN} months, current path", "value": _money(shown["cltv_current"]),
         "delta": f"{basis} per month {_money(now)}",
         "tone": "bad" if change < 0 else ("good" if change > 0 else "neutral"),
         "hint": f"The same {HORIZON_SHOWN} months at the value per month earned in the recent window."},
        {"label": "Value at risk", "value": _money(risk),
         "delta": "gap between the two paths" if risk else "none — current path is not below baseline",
         "tone": "bad" if risk else "good",
         "hint": "The lifetime cost of the leak. The same loss as the monthly figures, on a longer clock — never add the two."},
        {"label": "Value trend",
         "value": "—" if momentum is None else f"{momentum * 100:+.0f}%",
         "delta": f"{_money(before)} → {_money(now)}",
         "tone": "bad" if (momentum or 0) < -0.02 else ("good" if (momentum or 0) > 0.02 else "neutral"),
         "hint": "The single change that drives the gap. Still buying: "
                 f"{_pct(block['p_active_now'])} (from the account's ordering rhythm against the book's pattern)."},
    ])

    st.altair_chart(_paths_chart(block, colors), width="stretch")
    deviation = block.get("deviation_month")
    lost = block.get("lost_since_deviation") or 0
    if deviation and lost:
        when = pd.Period(deviation).strftime("%B %Y")
        story = (
            f"The lines part in {when}, the month after the account last earned its old average. "
            f"Since then it has earned {_money(lost)} less than it would have — the gap at today — "
            f"and on its current path the gap widens by another {_money(risk)} over the next {HORIZON_SHOWN} months."
        )
    else:
        story = (
            "The account is still earning at least its old average, so potential and reality are the "
            "same money all the way to today and only the projections differ."
        )
    st.caption(
        f"Blue is what the account would have earned at its baseline rate; red is what it actually earned "
        f"(solid), then where it is heading at today's rate (dashed). {story} "
        "A projection of the current path, not a forecast of what the customer will decide."
    )

    rows = []
    for horizon, values in horizons.items():
        rows.append({
            "Horizon": f"{horizon} months",
            "Active months": values["expected_active_months"],
            "Expected orders": values["expected_orders"],
            "Baseline path": values["cltv_baseline"],
            "Current path": values["cltv_current"],
            "Value at risk": values["value_at_risk"],
        })
    st.dataframe(
        pd.DataFrame(rows), width="stretch", hide_index=True,
        column_config={
            "Active months": st.column_config.NumberColumn(format="%.1f"),
            "Expected orders": st.column_config.NumberColumn(format="%.1f"),
            "Baseline path": st.column_config.NumberColumn(format="₹%.0f"),
            "Current path": st.column_config.NumberColumn(format="₹%.0f"),
            "Value at risk": st.column_config.NumberColumn(format="₹%.0f"),
        },
    )

    # Where the value sits: the same projection one level down, by category
    # and by product. Every line is value per month x the same 24 months, so
    # the parts add back to the account total.
    breakdown = block.get("breakdown") or {}
    horizon = breakdown.get("horizon_months", HORIZON_SHOWN)
    if breakdown.get("category"):
        st.markdown(f"**Category level analysis, {horizon} months**")
        _breakdown_table(breakdown["category"], "Category", basis)
    products = breakdown.get("product") or []
    if products:
        st.markdown(f"**Product level analysis, {horizon} months**")
        _breakdown_table(products[:PRODUCTS_SHOWN], "Product", basis)
        remaining = products[PRODUCTS_SHOWN:]
        if remaining:
            with st.expander(f"Remaining {len(remaining)} products"):
                _breakdown_table(remaining, "Product", basis)
    if breakdown.get("category") or products:
        st.caption(breakdown.get("note", ""))


def render_lifetime_book(rows: list[dict]) -> None:
    """`rows`: one dict per account with account_id, account_name, and the
    lifetime_value block. Ranked by value at risk over the shown horizon."""
    table = []
    for row in rows:
        block = row.get("lifetime_value") or {}
        scored = block.get("status") == "scored"
        shown = block.get("horizons_months", {}).get(HORIZON_SHOWN, {}) if scored else {}
        before, now = block.get("value_per_month_baseline"), block.get("value_per_month_recent")
        table.append({
            "account_id": row["account_id"],
            "account_name": row.get("account_name") or "",
            "scored": scored,
            "momentum": ((now - before) / before) if scored and before else None,
            "baseline": shown.get("cltv_baseline"),
            "current": shown.get("cltv_current"),
            "at_risk": shown.get("value_at_risk"),
        })
    table.sort(key=lambda r: (-(r["at_risk"] or 0), r["account_id"]))

    cells = []
    for row in table:
        # The page rides in the link: a click is a real navigation and may
        # start a fresh session, whose default page would otherwise be Detect.
        href = f"?account={quote(row['account_id'], safe='')}&page=cltv"
        if row["scored"]:
            momentum = "—" if row["momentum"] is None else f"{row['momentum'] * 100:+.0f}%"
            values = (_money(row["baseline"]), _money(row["current"]), _money(row["at_risk"]), momentum)
        else:
            values = ("—", "—", "—", "—")
        tds = "".join(
            f'<td><a class="rl-book-hit" href="{escape(href, quote=True)}">{escape(v)}</a></td>'
            for v in (row["account_id"], row["account_name"], *values)
        )
        cells.append(f"<tr>{tds}</tr>")
    headers = "".join(
        f"<th>{escape(h)}</th>" for h in
        ("Account", "Name", f"Baseline path, {HORIZON_SHOWN} mo", f"Current path, {HORIZON_SHOWN} mo",
         "Value at risk", "Value trend")
    )
    st.markdown(
        '<div class="rl-book-wrap"><table class="rl-book">'
        f"<thead><tr>{headers}</tr></thead><tbody>{''.join(cells)}</tbody></table></div>",
        unsafe_allow_html=True,
    )
    # An unscored account shows dashes in its row; no caption under the table
    # (removed at the user's request) — the account's own page says why.
