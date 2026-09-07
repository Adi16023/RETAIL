"""
The account dashboard: pick an account, see what is happening to it.

Read top to bottom it answers a manager's questions in order — is anything
wrong (status strip), how big (KPI tiles), where (tabs by dimension), show
me the numbers (table). Data-quality markers (gaps, bulk months) sit on the
Trend chart; the sufficiency label is in the status strip.

Everything on this page is DETERMINISTIC. Exploring an account costs no API
calls; the AI verdict is a separate, explicit button. That matters because a
manager clicking through eighteen accounts should not be spending money or
waiting on a model to look at a chart.
"""

from __future__ import annotations

from html import escape

import altair as alt
import pandas as pd
import streamlit as st

from pipeline.evidence import build_evidence_pack
from pipeline.timeline import PRESENCE_ONLY_DIMENSIONS

from .charts import (
    category_chart,
    order_chart,
    rate_chart,
    revenue_chart,
    tier_mix_chart,
)
from .metrics import category_comparison, headline_metrics, monthly_frame, tier_long
from .palette import STATUS_MEANING, active, status_style
from . import theme
from .verdict import noise_verdict

# Status-strip key -> the significance test that speaks to it.
SIGNIFICANCE_KEY = {
    "revenue_decline": "revenue", "margin": "margin", "discount": "discount",
    "tier_mix": "tier_mix", "order_pattern": "order_frequency",
}

# Five views, with icons instead of a tab bar. Data quality was the sixth
# until Sept 8 and was removed at the user's request: its gap / bulk-month
# markers already sit on the Trend chart, and the sufficiency label is in
# the status strip. `_render_quality_tab` stays below, unrouted.
# `pack_keys` are the detector blocks that make this the view worth opening first.
VIEWS = [
    {
        "key": "trend",
        "icon": ":material/show_chart:",
        "label": "Trend",
        "pack_keys": ("revenue_decline", "margin"),
    },
    {
        "key": "mix",
        "icon": ":material/category:",
        "label": "Value mix",
        "pack_keys": ("tier_mix",),
    },
    {
        "key": "pricing",
        "icon": ":material/sell:",
        "label": "Pricing",
        "pack_keys": ("discount",),
    },
    {
        "key": "orders",
        "icon": ":material/shopping_bag:",
        "label": "Order pattern",
        "pack_keys": ("order_pattern",),
    },
    {
        "key": "table",
        "icon": ":material/table_rows:",
        "label": "Monthly data",
        "pack_keys": (),
    },
]

# Dimension -> (evidence-pack key, label, one-line explanation of what it means).
SIGNAL_ROW = [
    ("revenue_decline", "Revenue", "Is the account spending less overall?"),
    ("margin", "Margin", "Is the same revenue earning less profit?"),
    ("discount", "Discount", "Are we discounting more deeply than we were?"),
    ("tier_mix", "Value mix", "Is spend moving between premium and cheap lines?"),
    ("order_pattern", "Order shape", "More orders, smaller baskets — multi-sourcing?"),
]


def _is_concern(status: str | None) -> bool:
    role, _, _ = STATUS_MEANING.get(status or "", ("warning", "", ""))
    return role in ("critical", "serious")


def _default_view(pack: dict) -> str:
    """Open on the first view that is actually moving — not always Trend."""
    if any(c.get("defected") for c in pack.get("category_changes") or []):
        return "mix"
    for view in VIEWS:
        for pack_key in view["pack_keys"]:
            status = (pack.get(pack_key) or {}).get("status")
            if _is_concern(status):
                return view["key"]
    return "trend"


def _set_dash_view(state_key: str, view_key: str) -> None:
    """Runs before the fragment rerenders, so the highlight moves on the first click."""
    st.session_state[state_key] = view_key


def _render_view_picker(pack: dict, account_id: str) -> str:
    """AppTab-style vertical list. Returns the selected view key."""
    state_key = f"dash_view_{account_id}"
    if state_key not in st.session_state:
        st.session_state[state_key] = _default_view(pack)
    current = st.session_state[state_key]

    for view in VIEWS:
        st.button(
            view["label"],
            icon=view["icon"],
            key=f"dashview-{account_id}-{view['key']}",
            type="primary" if current == view["key"] else "secondary",
            use_container_width=True,
            on_click=_set_dash_view,
            args=(state_key, view["key"]),
        )
    return current


def _format_value(value, kind: str) -> str:
    if value is None:
        return "—"
    if kind == "currency":
        return f"₹{value:,.0f}"
    if kind == "percent":
        return f"{value:.1f}%"
    return f"{value:,.2f}"


def _render_status_strip(pack: dict, colors: dict) -> None:
    dimensions = pack.get("analysis_dimensions") or {}
    availability = {
        "revenue_decline": dimensions.get("revenue", True),
        "margin": dimensions.get("margin", True),
        "discount": dimensions.get("discount", True),
        "tier_mix": dimensions.get("tier_mix", True),
        "order_pattern": dimensions.get("order_pattern", True),
    }

    significance = pack.get("significance") or {}
    items = []
    for key, label, explanation in SIGNAL_ROW:
        block = pack.get(key)
        status = "unavailable" if not availability.get(key, True) or block is None else block.get("status")
        color, icon, text = status_style(status, colors)
        # The hover carries the noise test, so a manager can tell "over the
        # threshold" from "over the threshold AND bigger than this account's
        # usual wobble" without leaving the strip.
        noise = noise_verdict(significance.get(SIGNIFICANCE_KEY.get(key)))
        hint = explanation
        if noise != "—":
            hint = f"{explanation} Real, or noise? {noise}."
        items.append({"label": label, "color": color, "icon": icon, "text": text, "hint": hint})

    defected = [c["category"] for c in pack["category_changes"] if c["defected"]]
    color, icon, text = status_style("critical" if defected else "stable", colors)
    if defected:
        color, icon, text = colors["critical"], "▼", "Line lost"
    items.append({
        "label": "Categories",
        "color": color,
        "icon": icon,
        "text": text,
        "hint": "Has a product line stopped completely and stayed stopped?",
    })
    theme.status_strip(items)
    if defected:
        st.caption(f"Stopped completely: **{', '.join(defected)}**")


def _kpi_tone(difference: float, good_direction: str) -> str:
    if good_direction == "neutral" or difference == 0:
        return "neutral"
    improved = difference > 0 if good_direction == "up" else difference < 0
    return "good" if improved else "bad"


def _render_kpi_tiles(pack: dict, frame: pd.DataFrame) -> None:
    tiles = headline_metrics(pack, frame)
    for row_start in (0, 3):
        cards = []
        for tile in tiles[row_start:row_start + 3]:
            baseline, recent = tile["baseline"], tile["recent"]
            delta, tone = None, "neutral"
            if baseline is not None and recent is not None:
                difference = recent - baseline
                if tile["format"] == "percent":
                    delta = f"{difference:+.1f}pp vs baseline"
                elif tile["format"] == "currency":
                    delta = f"{difference:+,.0f} vs baseline"
                else:
                    delta = f"{difference:+,.2f} vs baseline"
                # Green is good in that metric's own direction — for discount,
                # up is bad, so the tone is set per metric rather than assumed.
                tone = _kpi_tone(difference, tile["good_direction"])
            cards.append({
                "label": tile["label"],
                "value": _format_value(recent, tile["format"]),
                "delta": delta,
                "tone": tone,
                "hint": f"Baseline: {_format_value(baseline, tile['format'])}",
            })
        theme.kpi_row(cards)


def _render_trend_tab(pack: dict, frame: pd.DataFrame, colors: dict) -> None:
    quality = pack.get("data_quality") or {}
    change_point = (pack["overall_revenue"].get("change_point") or {}).get("change_point_month")

    st.markdown("**Revenue by month**")
    st.altair_chart(
        revenue_chart(
            frame, colors,
            change_point=change_point,
            gap_months=(quality.get("gaps") or {}).get("months_with_no_orders"),
            outlier_months=[o["month"] for o in (quality.get("outlier_months") or [])],
        ),
        width="stretch",
    )
    legend = ["Shaded band = the recent window every comparison is made against."]
    if (quality.get("outlier_months") or []):
        legend.append("▲ = unusually large month (a stock-up, not a trend).")
    if (quality.get("gaps") or {}).get("months_with_no_orders"):
        legend.append("✚ = no orders placed that month (a gap, not zero trading).")
    if change_point:
        legend.append("Red rule = detected change-point.")
    st.caption(" ".join(legend))

    margin = rate_chart(frame, colors, "margin_pct", "Margin rate")
    if margin is not None:
        st.markdown("**Margin rate by month**")
        st.altair_chart(margin, width="stretch")
        # The pairing that makes the hidden leak visible: flat line above,
        # falling line below. Deliberately two charts, never a dual axis.
        st.caption(
            "Read this against the revenue chart above. Flat revenue with a falling "
            "margin rate is value leaving the account without the topline showing it."
        )
    else:
        st.info("No margin data in this file — margin cannot be analysed.")


def _render_mix_tab(pack: dict, df: pd.DataFrame, account_id: str,
                    frame: pd.DataFrame, colors: dict) -> None:
    long = tier_long(frame)
    if long.empty:
        st.info("No product value-tier column in this file — mix cannot be analysed.")
    else:
        st.markdown("**Where the money sits, by value tier**")
        as_share = st.toggle(
            "Show as share of revenue", value=True,
            help="Share exposes a mix shift that a flat total would hide.",
        )
        st.altair_chart(tier_mix_chart(long, colors, normalize=as_share), width="stretch")

        tier = pack.get("tier_mix") or {}
        change = tier.get("high_tier_share_change_pp")
        if change is not None:
            direction = "into" if change > 0 else "out of"
            st.caption(
                f"High-tier share moved {change:+.1f} percentage points — value is moving "
                f"{direction} premium lines. Direction matters more than size here: the same "
                "shift upward is a healthy account trading up."
            )

    st.markdown("**Category revenue: baseline vs recent**")
    comparison = category_comparison(df, account_id)
    chart = category_chart(comparison, colors)
    if chart is None:
        st.info("Not enough history to compare a baseline window against a recent one.")
    else:
        st.altair_chart(chart, width="stretch")
        defected = [c for c in pack["category_changes"] if c["defected"]]
        if defected:
            for entry in defected:
                st.caption(
                    f"**{entry['category']}** has been at zero for "
                    f"{entry['consecutive_months_at_zero']} consecutive months."
                )


def _render_pricing_tab(pack: dict, frame: pd.DataFrame, colors: dict) -> None:
    chart = rate_chart(frame, colors, "discount_pct", "Average discount", color_key="series_2")
    if chart is None:
        st.info("No discount or list-price data in this file — discounting cannot be analysed.")
        return

    st.markdown("**Average discount by month**")
    st.altair_chart(chart, width="stretch")
    st.caption(
        "Weighted by list value, so a discount on a large high-value line counts for more "
        "than the same discount on one cheap line."
    )

    discount = pack.get("discount") or {}
    if discount.get("status") == "creep_detected":
        st.warning(
            f"Discount has crept {discount['discount_pct_change_pp']:+.1f} percentage points "
            f"({discount['baseline_avg_discount_pct'] * 100:.1f}% → "
            f"{discount['recent_avg_discount_pct'] * 100:.1f}%) while volume and mix held steady. "
            "Revenue-only reporting cannot see this."
        )


def _render_orders_tab(pack: dict, frame: pd.DataFrame, colors: dict) -> None:
    left, right = st.columns(2)
    with left:
        st.markdown("**Orders per month**")
        st.altair_chart(order_chart(frame, colors, "orders", "Orders"), width="stretch")
    with right:
        st.markdown("**Lines per order**")
        st.altair_chart(
            order_chart(frame, colors, "lines_per_order", "Lines per order"),
            width="stretch",
        )

    pattern = pack.get("order_pattern") or {}
    if pattern.get("status") == "fragmentation_detected":
        st.warning(
            f"Order frequency is up {pattern['order_frequency_pct_change'] * 100:+.0f}% while "
            f"basket width is down {pattern['basket_width_pct_change'] * 100:+.0f}%. More orders, "
            "each smaller — the shape of an account that has started buying part of its basket "
            "elsewhere."
        )
    if pack.get("account_manager_changed"):
        st.info(
            f"Account manager changed during this history ({', '.join(pack['account_manager'])}). "
            "Worth noting as context — it correlates with everything and causes nothing by itself."
        )


def _render_quality_tab(pack: dict, frame: pd.DataFrame) -> None:
    quality = pack.get("data_quality") or {}
    sufficiency = pack.get("data_sufficiency") or {}

    label = sufficiency.get("label", "unknown")
    message = (
        f"**Data sufficiency: {label}** — {sufficiency.get('history_months')} months, "
        f"{sufficiency.get('order_count')} orders, {sufficiency.get('category_count')} categories."
    )
    (st.error if label == "insufficient" else st.success if label == "sufficient" else st.warning)(message)
    if sufficiency.get("flags"):
        for flag in sufficiency["flags"]:
            st.caption(f"• {flag.replace('_', ' ')}")

    gaps = (quality.get("gaps") or {}).get("months_with_no_orders") or []
    if gaps:
        st.warning(
            f"**No orders in:** {', '.join(gaps)}. A missing month drags any trailing average "
            "down without any change in customer behaviour — it is a gap, not a stop."
        )

    outliers = quality.get("outlier_months") or []
    for outlier in outliers:
        st.info(
            f"**Bulk month {outlier['month']}:** ₹{outlier['revenue']:,.0f}, "
            f"{outlier['multiple_of_median_month']}× a typical month. The ordinary months after "
            "it are a return to normal, not a decline."
        )

    returns = quality.get("returns") or {}
    if returns.get("return_lines"):
        st.info(
            f"**{returns['return_lines']} return/credit line(s)** worth "
            f"₹{returns['net_return_value']:,.0f}, already netted into every figure above. "
            "A credit note is not leakage."
        )

    seasonality = pack.get("seasonality") or {}
    if seasonality.get("status") == "confirmed":
        st.success(
            f"**Seasonal pattern confirmed.** The recent dip ({', '.join(seasonality['dip_months'])}) "
            f"repeats the same calendar window a year earlier "
            f"({', '.join(seasonality['echo_months'])}). Strong evidence this is seasonal, not structural."
        )

    for episode in pack.get("dip_episodes") or []:
        if episode["recovered"]:
            st.success(
                f"**Dip {episode['start_month']} → {episode['end_month']} recovered** "
                f"from {episode['recovered_from_month']}. A resolved incident, not a current leak."
            )
        elif episode["is_ongoing"]:
            st.error(
                f"**Ongoing dip since {episode['start_month']}** "
                f"({episode['months']} months, no recovery yet)."
            )

    # "returns" is excluded: it flags whether return lines are PRESENT, not
    # whether returns could be analysed. An account that never sent a credit
    # note has no gap in coverage, and saying otherwise reads as alarming
    # (see timeline.PRESENCE_ONLY_DIMENSIONS).
    unavailable = [
        name for name, ok in (pack.get("analysis_dimensions") or {}).items()
        if not ok and name not in PRESENCE_ONLY_DIMENSIONS
    ]
    if unavailable:
        st.warning(
            "**Not analysed — columns absent from this file:** "
            + ", ".join(n.replace("_", " ") for n in unavailable)
            + ". These were not measured; that is not the same as measured and fine."
        )


def _render_table_tab(frame: pd.DataFrame) -> None:
    """The table view every chart has a twin in — same rows, so a value on a
    chart is always reachable without hovering."""
    display = frame.copy()
    display["Month"] = display["month"].dt.strftime("%b %Y")
    columns = {
        "Month": "Month", "window": "Window", "revenue": "Revenue", "margin": "Margin",
        "margin_pct": "Margin %", "discount_pct": "Discount %", "high_tier_share": "High-tier %",
        "orders": "Orders", "lines": "Lines", "lines_per_order": "Lines/order",
    }
    present = {k: v for k, v in columns.items() if k in display.columns}
    table = display[list(present)].rename(columns=present)
    for percent_column in ("Margin %", "Discount %", "High-tier %"):
        if percent_column in table.columns:
            table[percent_column] = table[percent_column] * 100

    st.dataframe(
        table, width="stretch", hide_index=True,
        column_config={
            "Revenue": st.column_config.NumberColumn(format="%.0f"),
            "Margin": st.column_config.NumberColumn(format="%.0f"),
            "Margin %": st.column_config.NumberColumn(format="%.1f%%"),
            "Discount %": st.column_config.NumberColumn(format="%.1f%%"),
            "High-tier %": st.column_config.NumberColumn(format="%.1f%%"),
            "Lines/order": st.column_config.NumberColumn(format="%.2f"),
        },
    )

def render_account_dashboard(df: pd.DataFrame, account_id: str,
                             pack: dict | None = None) -> dict:
    """Render the whole dashboard for one account. Returns the evidence pack so
    the caller can reuse it for a live investigation without recomputing.

    `pack` lets the caller pass a cached pack in: every widget interaction
    re-runs the whole Streamlit script, and rebuilding the pack on each one is
    wasted work that shows up as lag when switching models.
    """
    colors = active()
    alt.theme.enable("default")

    if pack is None:
        pack = build_evidence_pack(df, account_id)
    frame = monthly_frame(df, account_id)

    identity_bits = [account_id]
    if pack.get("account_name"):
        identity_bits.append(str(pack["account_name"]))

    context = []
    if pack.get("region"):
        context.append(f"Region: <strong>{escape(str(pack['region']))}</strong>")
    manager = pack.get("account_manager")
    if manager:
        manager_text = ", ".join(manager) if isinstance(manager, list) else manager
        context.append(f"Manager: <strong>{escape(str(manager_text))}</strong>")
    history = pack.get("history") or {}
    context.append(
        f"History: <strong>{escape(str(history.get('months_of_history')))} months</strong> "
        f"({escape(str(history.get('start_date')))} → {escape(str(history.get('end_date')))})"
    )
    context.append(f"Orders: <strong>{escape(str(history.get('order_count')))}</strong>")
    theme.identity(" · ".join(identity_bits), "  ·  ".join(context))

    _render_status_strip(pack, colors)
    st.divider()
    _render_kpi_tiles(pack, frame)
    st.caption(
        "Each tile compares the trailing 6 months against everything before it. "
        "Green is good in that metric's own direction — for discount, down is good."
    )
    st.divider()
    _render_chart_views(df, account_id, pack, frame, colors)

    return pack


@st.fragment
def _render_chart_views(df: pd.DataFrame, account_id: str, pack: dict,
                        frame: pd.DataFrame, colors: dict) -> None:
    """Only this block reruns when a view is clicked. Tabs sit on the left;
    the chart on the right fades in on each switch."""
    nav, panel = st.columns([1, 4], gap="large")
    with nav:
        view = _render_view_picker(pack, account_id)
    with panel:
        if view == "trend":
            _render_trend_tab(pack, frame, colors)
        elif view == "mix":
            _render_mix_tab(pack, df, account_id, frame, colors)
        elif view == "pricing":
            _render_pricing_tab(pack, frame, colors)
        elif view == "orders":
            _render_orders_tab(pack, frame, colors)
        else:
            _render_table_tab(frame)
