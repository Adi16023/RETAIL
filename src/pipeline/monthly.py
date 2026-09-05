"""
One account's history as one row per calendar month — the raw monthly table
that both the significance tests and the dashboard read from.

Raw, not smoothed: these are the months that actually happened. The rolling
window the detectors use exists to stop noise creating verdicts, and the
permutation tests need the un-smoothed months precisely because they are
measuring that noise.

Months with no orders are present as zero-revenue rows, not absent. A gap has
to be visible as a gap; a series that quietly skips it would understate the
account's true volatility and overstate every trend fitted through it.

Returns (credit notes) are NETTED into revenue and margin — that is what a
credit note means financially — but excluded from the behavioural columns
(orders, lines, basket width, discount), where a standalone credit is not an
order and counting it as one fakes a frequency rise and a basket collapse at
once. This mirrors Stage 2's `_sales_lines`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .changepoint import full_month_index

TIERS = ("High", "Mid", "Low")


def monthly_table(acc_df: pd.DataFrame) -> pd.DataFrame:
    """Raw monthly rows for ONE account's transactions.

    Columns always present: period, revenue, margin, margin_pct, orders,
    lines, lines_per_order, discount_pct. Tier columns (tier_High/Mid/Low,
    high_tier_share) appear only when the file carries a tier column.
    Rate columns are NaN for months with no orders, and entirely NaN when the
    underlying column is absent from the file — never zero. Absent and zero
    are different answers.
    """
    if acc_df.empty:
        return pd.DataFrame()

    account = acc_df.copy()
    months = full_month_index(account)
    account["month"] = account["date"].dt.to_period("M")
    sales = account[account["is_return"] != 1] if "is_return" in account.columns else account

    frame = pd.DataFrame({"period": months})

    def summed(source: pd.DataFrame, column: str) -> np.ndarray:
        return source.groupby("month")[column].sum().reindex(months, fill_value=0.0).to_numpy()

    frame["revenue"] = summed(account, "revenue")

    has_margin = "margin" in account.columns and account["margin"].notna().any()
    if has_margin:
        frame["margin"] = summed(account, "margin")
        with np.errstate(divide="ignore", invalid="ignore"):
            frame["margin_pct"] = np.where(frame["revenue"] != 0,
                                           frame["margin"] / frame["revenue"], np.nan)
    else:
        frame["margin"] = np.nan
        frame["margin_pct"] = np.nan

    frame["orders"] = (
        sales.drop_duplicates("order_id").groupby("month")["order_id"].nunique()
        .reindex(months, fill_value=0).to_numpy()
    )
    frame["lines"] = sales.groupby("month").size().reindex(months, fill_value=0).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        frame["lines_per_order"] = np.where(frame["orders"] > 0,
                                            frame["lines"] / frame["orders"], np.nan)

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
        with np.errstate(divide="ignore", invalid="ignore"):
            frame["discount_pct"] = np.where(grouped["weight"] > 0,
                                             grouped["weighted"] / grouped["weight"], np.nan)
        # Numerator and denominator kept separately so a rate can be tested as
        # a ratio of sums over a window — the same quantity the detector
        # reports — rather than as a mean of monthly ratios, which lets a
        # month with one small order count as much as a month with twenty.
        frame["discount_weight"] = grouped["weight"].fillna(0.0).to_numpy()
        frame["discount_weighted"] = grouped["weighted"].fillna(0.0).to_numpy()
    else:
        frame["discount_pct"] = np.nan
        frame["discount_weight"] = np.nan
        frame["discount_weighted"] = np.nan

    if "tier" in account.columns and account["tier"].notna().any():
        for tier in TIERS:
            frame[f"tier_{tier}"] = (
                account[account["tier"] == tier].groupby("month")["revenue"].sum()
                .reindex(months, fill_value=0.0).to_numpy()
            )
        total = frame[[f"tier_{t}" for t in TIERS]].sum(axis=1)
        frame["tier_total"] = total.to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            frame["high_tier_share"] = np.where(total != 0, frame["tier_High"] / total, np.nan)

    return frame
