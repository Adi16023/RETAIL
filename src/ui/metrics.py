"""
The one monthly frame every chart and the table view are built from.

Charts and the table read the SAME rows, so a number on a chart and the number
in the table can never disagree. Values are raw monthly figures — not the
rolling-window smoothed series the detectors work on — because a manager
reading a chart wants the month that actually happened, and the smoothing
exists to stop noise creating verdicts, not to hide months.

Nothing here is a detector. It reshapes what ingestion already produced; the
verdict-relevant statuses all come from the evidence pack.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from pipeline.changepoint import RECENT_MONTHS, full_month_index


# Cached because every widget interaction re-runs the whole Streamlit script.
# These reshape thousands of rows and depend only on (data, account), never on
# the widget that changed — recomputing them per keystroke is what makes a
# dashboard feel sluggish.
@st.cache_data(show_spinner=False)
def monthly_frame(df: pd.DataFrame, account_id: str) -> pd.DataFrame:
    """One row per calendar month of this account's history.

    Months with no orders are present with zeros rather than absent, so a gap
    is visible as a gap instead of the line quietly jumping over it.
    """
    account = df[df["account_id"] == account_id].copy()
    if account.empty:
        return pd.DataFrame()

    months = full_month_index(account)
    account["month"] = account["date"].dt.to_period("M")
    sales = account[account["is_return"] != 1] if "is_return" in account.columns else account

    frame = pd.DataFrame({"period": months})
    frame["month"] = frame["period"].dt.to_timestamp()

    def summed(source: pd.DataFrame, column: str) -> pd.Series:
        if column not in source.columns:
            return pd.Series(0.0, index=months)
        return source.groupby("month")[column].sum().reindex(months, fill_value=0.0)

    frame["revenue"] = summed(account, "revenue").to_numpy()

    # A file with no margin column must leave these EMPTY, not zero. Summing an
    # all-NaN column with fill_value=0 would draw a confident flat 0% margin
    # line — asserting the account earns nothing, when the truth is we cannot
    # see it. Absent and zero are different answers.
    has_margin = "margin" in account.columns and account["margin"].notna().any()
    if has_margin:
        frame["margin"] = summed(account, "margin").to_numpy()
        frame["margin_pct"] = [
            (m / r) if r else None for m, r in zip(frame["margin"], frame["revenue"])
        ]
    else:
        frame["margin"] = None
        frame["margin_pct"] = None

    orders = account.drop_duplicates("order_id")
    sales_orders = sales.drop_duplicates("order_id")
    frame["orders"] = (
        sales_orders.groupby("month")["order_id"].nunique().reindex(months, fill_value=0).to_numpy()
    )
    frame["lines"] = account.groupby("month").size().reindex(months, fill_value=0).to_numpy()
    frame["lines_per_order"] = [
        (lines / orders_) if orders_ else None
        for lines, orders_ in zip(frame["lines"], frame["orders"])
    ]

    # Discount weighted by gross list value, matching Stage 2 — an unweighted
    # mean would let a handful of small orders mask creep on the lines that
    # actually carry the revenue.
    if "discount_pct" in sales.columns and sales["discount_pct"].notna().any():
        priced = sales[sales["discount_pct"].notna()].copy()
        if "list_price" in priced.columns and priced["list_price"].notna().any():
            priced["weight"] = (priced["list_price"] * priced["quantity"]).abs()
        else:
            priced["weight"] = priced["revenue"].abs()
        priced.loc[priced["weight"].isna() | (priced["weight"] <= 0), "weight"] = 1.0
        priced["weighted"] = priced["discount_pct"] * priced["weight"]
        grouped = priced.groupby("month")[["weight", "weighted"]].sum().reindex(months)
        frame["discount_pct"] = [
            (w / t) if (t and t > 0) else None
            for w, t in zip(grouped["weighted"], grouped["weight"])
        ]
    else:
        frame["discount_pct"] = None

    if "tier" in account.columns and account["tier"].notna().any():
        for tier in ("High", "Mid", "Low"):
            frame[f"tier_{tier}"] = (
                account[account["tier"] == tier].groupby("month")["revenue"].sum()
                .reindex(months, fill_value=0.0).to_numpy()
            )
        total = frame[["tier_High", "tier_Mid", "tier_Low"]].sum(axis=1)
        frame["high_tier_share"] = [
            (h / t) if t else None for h, t in zip(frame["tier_High"], total)
        ]

    frame["window"] = ["Recent" if i >= len(months) - RECENT_MONTHS else "Baseline"
                       for i in range(len(months))]
    frame["has_orders"] = frame["orders"] > 0
    return frame


def tier_long(frame: pd.DataFrame) -> pd.DataFrame:
    """Tier revenue in long form for a stacked chart, ordered High -> Low."""
    columns = [c for c in ("tier_High", "tier_Mid", "tier_Low") if c in frame.columns]
    if not columns:
        return pd.DataFrame()
    # Melt off a narrowed copy: pandas refuses a value_name that collides with
    # an existing column, and the source frame has its own "revenue".
    long = frame[["month", *columns]].melt(
        id_vars=["month"], value_vars=columns, var_name="tier", value_name="revenue"
    )
    long["tier"] = long["tier"].str.replace("tier_", "", regex=False)
    return long


@st.cache_data(show_spinner=False)
def category_comparison(df: pd.DataFrame, account_id: str) -> pd.DataFrame:
    """Average monthly revenue per category, baseline window vs recent window.

    A before/after pair per category is what makes a defection legible: the
    line that went to zero is the bar with nothing on its recent end.
    """
    account = df[df["account_id"] == account_id].copy()
    if account.empty:
        return pd.DataFrame()

    months = full_month_index(account)
    if len(months) <= RECENT_MONTHS:
        return pd.DataFrame()

    account["month"] = account["date"].dt.to_period("M")
    baseline_months, recent_months = months[:-RECENT_MONTHS], months[-RECENT_MONTHS:]

    rows = []
    for category in sorted(account["category"].unique()):
        subset = account[account["category"] == category]
        baseline = subset[subset["month"].isin(baseline_months)]["revenue"].sum() / len(baseline_months)
        recent = subset[subset["month"].isin(recent_months)]["revenue"].sum() / len(recent_months)
        rows.append({"category": category, "window": "Baseline", "revenue": round(float(baseline), 2)})
        rows.append({"category": category, "window": "Recent", "revenue": round(float(recent), 2)})

    comparison = pd.DataFrame(rows)
    order = (
        comparison[comparison["window"] == "Baseline"]
        .sort_values("revenue", ascending=False)["category"].tolist()
    )
    comparison["category"] = pd.Categorical(comparison["category"], categories=order, ordered=True)
    return comparison.sort_values("category")


def headline_metrics(pack: dict, frame: pd.DataFrame) -> list[dict]:
    """The tiles a manager reads first: baseline vs recent for each dimension.

    Every value comes from the evidence pack, so the tiles, the charts and the
    agent's verdict are all quoting the same arithmetic. `good_direction` says
    which way is good, because for discount down is good and for margin up is —
    a single "green means up" rule would mislead on half of these.
    """
    revenue = pack.get("overall_revenue") or {}
    margin = pack.get("margin_profile") or {}
    discount = pack.get("discount") or {}
    tier = pack.get("tier_mix") or {}
    orders = pack.get("order_behavior") or {}

    def pct(value):
        return None if value is None else value * 100

    tiles = [
        {
            "label": "Revenue / month",
            "baseline": revenue.get("baseline_monthly_median"),
            "recent": revenue.get("recent_monthly_rate"),
            "format": "currency",
            "good_direction": "up",
        },
        {
            "label": "Margin rate",
            "baseline": pct(margin.get("baseline_margin_pct")),
            "recent": pct(margin.get("recent_margin_pct")),
            "format": "percent",
            "good_direction": "up",
        },
        {
            "label": "High-tier share",
            "baseline": pct((tier.get("baseline_share_by_tier") or {}).get("High")),
            "recent": pct((tier.get("recent_share_by_tier") or {}).get("High")),
            "format": "percent",
            "good_direction": "up",
        },
        {
            "label": "Avg discount",
            "baseline": pct(discount.get("baseline_avg_discount_pct")),
            "recent": pct(discount.get("recent_avg_discount_pct")),
            "format": "percent",
            "good_direction": "down",
        },
        {
            "label": "Orders / month",
            "baseline": orders.get("order_frequency_baseline_per_month"),
            "recent": orders.get("order_frequency_recent_per_month"),
            "format": "number",
            "good_direction": "neutral",
        },
        {
            "label": "Lines / order",
            "baseline": orders.get("basket_width_baseline"),
            "recent": orders.get("basket_width_recent"),
            "format": "number",
            "good_direction": "up",
        },
    ]
    return tiles
