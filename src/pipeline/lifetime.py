"""
Customer lifetime value (CLTV) — what an account is expected to be worth
over the next 12 and 24 months, on its baseline behaviour and on its current
path, and the gap between the two.

Two ideas, both standard in customer analytics, one figure that is ours:

* **How many more orders, and are they still active?** A BG/NBD model
  (Fader, Hardie & Lee 2005), the textbook "buy till you die" model for
  non-contractual buyers. From each account's repeat-order count, its age
  and the time since its last order it gives the probability the account is
  still active and the expected number of orders over a horizon. The four
  parameters are fitted ONCE on the whole book, pooled: with 18 accounts no
  account has enough history to fit on its own, so each is scored by its own
  history against the shared pattern. Fitted with SciPy; no new dependency.
* **What each order is worth.** Margin per order, not revenue — the flat-
  revenue leaks (ACC-101, ACC-107) are the whole point of this project, and
  a revenue CLTV would call them healthy. Where the file carries no margin
  the value is measured in revenue and the block says so.
* **The figure that leads: value at risk.** CLTV is computed twice — at the
  baseline margin per order and at the recent margin per order — with the
  same expected orders. The difference is the lifetime cost of the leak: the
  same loss the dashboard shows per month, on a longer clock. The two are
  never added.

Every number is a function of the order file. The LLM may quote the block;
it never computes any of it. An account with too few repeat orders or too
short a history is reported as insufficient_history, not scored — ACC-109
gets no lifetime value, the same way it gets no verdict.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln, hyp2f1

from .changepoint import RECENT_MONTHS

DAYS_PER_MONTH = 30.4375
HORIZONS_MONTHS = (12, 24)
# Every rupee figure is value per month x the horizon months, flat: 24 for
# the 24-month view, 12 for the 12-month view. A 10% time discount and the
# BG/NBD expected-months adjustment were applied until Sept 8 and removed at
# the user's request: they turned 24 into 22.2 on every account of this book
# (nobody has gone dark, so the model added nothing the discount did not
# take away) and a reader could not reproduce the tiles by hand. The BG/NBD
# fit still reports expected orders, expected active months and the
# probability of still buying, for information; it no longer scales money.
# Below either floor the BG/NBD conditioning has almost nothing to condition
# on, and the margin-per-order split has no baseline. Six orders means five
# repeat orders; eight months is the same floor the trend test uses.
MIN_ORDERS = 6
MIN_HISTORY_MONTHS = 8
# Products listed in the breakdown; a long tail of one-off lines adds
# nothing a manager can act on.
MAX_PRODUCTS_LISTED = 40
# Ridge penalty on the four parameters, as the lifetimes library does for
# small samples: without it the dropout parameters wander on a book where
# almost nobody has actually gone dark.
PENALIZER = 0.01


# --- Per-account summary ------------------------------------------------------------------

def _sales_orders(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (account, order): the order's date. Return-only lines are
    not orders; a credit note is not a purchase."""
    d = df if "is_return" not in df.columns else df[df["is_return"] != 1]
    return d.groupby(["account_id", "order_id"], as_index=False)["date"].min()


def account_summary(df: pd.DataFrame) -> pd.DataFrame:
    """The three numbers BG/NBD needs per account, in months: `frequency`
    (repeat orders), `recency` (first order -> last order) and `age` (first
    order -> end of the file). The observation end is the file's last date
    for every account, so an account that went quiet is seen as quiet."""
    orders = _sales_orders(df)
    end = df["date"].max()
    rows = []
    for account_id, group in orders.groupby("account_id"):
        first, last = group["date"].min(), group["date"].max()
        rows.append({
            "account_id": account_id,
            "orders": int(len(group)),
            "frequency": int(len(group)) - 1,
            "recency": (last - first).days / DAYS_PER_MONTH,
            "age": (end - first).days / DAYS_PER_MONTH,
        })
    return pd.DataFrame(rows).set_index("account_id")


# --- BG/NBD ---------------------------------------------------------------------------------

def _negative_log_likelihood(log_params, x, t_x, T, penalizer: float) -> float:
    r, alpha, a, b = np.exp(log_params)
    a_1 = gammaln(r + x) - gammaln(r) + r * np.log(alpha)
    a_2 = gammaln(a + b) + gammaln(b + x) - gammaln(b) - gammaln(a + b + x)
    a_3 = -(r + x) * np.log(alpha + T)
    a_4 = np.log(a) - np.log(b + np.maximum(x, 1) - 1) - (r + x) * np.log(alpha + t_x)
    with_repeat = np.logaddexp(a_3, a_4)
    ll = a_1 + a_2 + np.where(x > 0, with_repeat, a_3)
    return -float(ll.sum()) + penalizer * float(np.sum(np.exp(log_params) ** 2))


def fit_bg_nbd(summary: pd.DataFrame, penalizer: float = PENALIZER) -> dict:
    """Maximum-likelihood BG/NBD on the pooled book. Returns the four
    parameters plus what they were fitted on."""
    x = summary["frequency"].to_numpy(dtype=float)
    t_x = summary["recency"].to_numpy(dtype=float)
    T = summary["age"].to_numpy(dtype=float)
    result = minimize(
        _negative_log_likelihood, x0=np.zeros(4), args=(x, t_x, T, penalizer),
        method="Nelder-Mead", options={"maxiter": 4000, "xatol": 1e-6, "fatol": 1e-8},
    )
    r, alpha, a, b = (float(v) for v in np.exp(result.x))
    return {
        "r": r, "alpha": alpha, "a": a, "b": b,
        "accounts_fitted": int(len(summary)),
        "converged": bool(result.success),
        "penalizer": penalizer,
        "method": "BG/NBD, pooled maximum likelihood",
    }


def p_alive(params: dict, x: float, t_x: float, T: float) -> float:
    """Probability the account is still active, given its history."""
    r, alpha, a, b = params["r"], params["alpha"], params["a"], params["b"]
    if x <= 0:
        return 1.0
    ratio = (a / (b + x - 1)) * ((alpha + T) / (alpha + t_x)) ** (r + x)
    return float(1.0 / (1.0 + ratio))


def expected_orders(params: dict, x: float, t_x: float, T: float, t: float) -> float:
    """Expected number of orders in the next `t` months, given the history
    (Fader, Hardie & Lee's conditional expectation). Falls back to the plain
    observed rate when the closed form is not finite, which happens when the
    fitted `a` sits at 1."""
    r, alpha, a, b = params["r"], params["alpha"], params["a"], params["b"]
    if t <= 0:
        return 0.0
    try:
        hyp = hyp2f1(r + x, b + x, a + b + x - 1, t / (alpha + T + t))
        first = (a + b + x - 1) / (a - 1)
        second = 1 - ((alpha + T) / (alpha + T + t)) ** (r + x) * hyp
        denominator = 1.0
        if x > 0:
            denominator += (a / (b + x - 1)) * ((alpha + T) / (alpha + t_x)) ** (r + x)
        value = float(first * second / denominator)
    except (ZeroDivisionError, FloatingPointError, ValueError):
        value = float("nan")
    if not math.isfinite(value) or value < 0:
        rate = x / T if T > 0 else 0.0
        return float(rate * t * p_alive(params, x, t_x, T))
    return value


# --- Value per active month ---------------------------------------------------------

def _value_per_month(acc_df: pd.DataFrame) -> dict:
    """Margin per month in the baseline window and in the recent window, the
    same split every tile uses. Revenue when the file has no margin.

    Per MONTH, not per order, on purpose. A customer who splits the same
    spend into more, smaller orders (ACC-108, and the healthy ACC-102 in its
    recent months) keeps the same value per month while value per order
    falls by a fifth; measuring per order would report a leak that is not
    there. The BG/NBD side is converted to active months to match.
    """
    basis = "margin" if "margin" in acc_df.columns and acc_df["margin"].notna().any() else "revenue"
    d = acc_df.copy()
    d["month"] = d["date"].dt.to_period("M")
    months = pd.period_range(d["month"].min(), d["month"].max(), freq="M")
    if len(months) <= RECENT_MONTHS:
        return {"basis": basis, "baseline": None, "recent": None}
    cutoff = months[-RECENT_MONTHS]
    baseline_months = len(months) - RECENT_MONTHS
    return {
        "basis": basis,
        "baseline": round(float(d[d["month"] < cutoff][basis].sum()) / baseline_months, 2),
        "recent": round(float(d[d["month"] >= cutoff][basis].sum()) / RECENT_MONTHS, 2),
    }


def _value_breakdown(acc_df: pd.DataFrame, basis: str, by: str, active_months: float,
                     limit: int | None = None) -> list[dict]:
    """The account's lifetime value split across `by` (category or product)
    over the longest horizon. Every line shares the account's expected
    months of continued buying, so the split is by value per month and the
    parts add back to the account total. Sorted by value at risk, then by
    baseline value, so both the lines losing and the lines backfilling them
    are visible."""
    if by not in acc_df.columns:
        return []
    d = acc_df.copy()
    d["month"] = d["date"].dt.to_period("M")
    months = pd.period_range(d["month"].min(), d["month"].max(), freq="M")
    if len(months) <= RECENT_MONTHS:
        return []
    cutoff = months[-RECENT_MONTHS]
    baseline_months = len(months) - RECENT_MONTHS
    rows = []
    for name, group in d.groupby(by):
        before = float(group[group["month"] < cutoff][basis].sum()) / baseline_months
        recent = float(group[group["month"] >= cutoff][basis].sum()) / RECENT_MONTHS
        baseline_value, current_value = round(before * active_months), round(recent * active_months)
        rows.append({
            "name": str(name),
            "value_per_month_baseline": round(before, 2),
            "value_per_month_recent": round(recent, 2),
            "cltv_baseline": baseline_value,
            "cltv_current": current_value,
            "value_change": current_value - baseline_value,
            "value_at_risk": max(0, baseline_value - current_value),
        })
    rows.sort(key=lambda r: (-r["value_at_risk"], -r["cltv_baseline"], r["name"]))
    return rows[:limit] if limit else rows


def _monthly_value(acc_df: pd.DataFrame, basis: str) -> pd.Series:
    """Value per calendar month over the account's whole history, gap months
    as zero — the series the paths chart accumulates."""
    d = acc_df.copy()
    d["month"] = d["date"].dt.to_period("M")
    months = pd.period_range(d["month"].min(), d["month"].max(), freq="M")
    return d.groupby("month")[basis].sum().reindex(months, fill_value=0.0).astype(float)


def _value_paths(acc_df: pd.DataFrame, value: dict, horizon: int) -> tuple[list[dict], str | None, int]:
    """Potential versus reality, cumulative, over the history and `horizon`
    months beyond it. Returns (rows, deviation_month, lost_since_deviation).

    Reality is what the account actually earned, month by month, then
    carried on at the recent rate. Potential is the same actual money up to
    the DEVIATION MONTH — the month after the last one in which the account
    still earned at least its baseline value per month — and the baseline
    rate from there on. So the two lines coincide while the account was
    itself, part company the month it stopped being, and the gap at today is
    what the slide has already cost; over the horizon it widens by exactly
    the value at risk. A projection from today alone (the first version)
    drew two straight lines opening from month 1, which read as "fine today,
    worse later" on accounts whose decline was already a year old.

    The break is defined against the baseline average rather than found by
    a change-point search because the leaks here take three shapes — a step
    (ACC-106), a bend (ACC-101) and a slide from the first month (ACC-107)
    — and the largest-drop split landed a year late on the bend. "The last
    month they earned their old average" is one rule that dates all three
    and is a sentence a manager can check against the monthly table.
    """
    monthly = _monthly_value(acc_df, value["basis"])
    baseline, recent = value["baseline"], value["recent"]
    above = [i for i, v in enumerate(monthly.values) if v >= baseline]
    # No deviation when the account is still earning its average in its
    # final month (healthy, or trading up): the lines split only at today.
    dev_index = above[-1] + 1 if above and above[-1] + 1 < len(monthly) else None

    rows, actual, potential = [], 0.0, 0.0
    last = len(monthly) - 1
    for i, (period, v) in enumerate(monthly.items()):
        actual += v
        potential += baseline if dev_index is not None and i >= dev_index else v
        rows.append({"month": str(period), "offset": i - last, "phase": "history",
                     "potential": round(potential), "reality": round(actual)})
    # From the rounded rows, so the caption's figure is exactly the gap the
    # chart draws at today.
    lost = rows[-1]["potential"] - rows[-1]["reality"]
    today = monthly.index[-1]
    for k in range(1, horizon + 1):
        rows.append({"month": str(today + k), "offset": k, "phase": "projection",
                     "potential": round(potential + k * baseline),
                     "reality": round(actual + k * recent)})
    deviation_month = str(monthly.index[dev_index]) if dev_index is not None else None
    return rows, deviation_month, max(0, lost)


def expected_active_months(params: dict, x: float, t_x: float, T: float, horizon: int) -> float:
    """Months of continued buying expected over the horizon: the expected
    orders divided by the account's long-run order rate, capped at the
    horizon, so a customer expected to keep ordering at their usual pace over
    12 months counts as 12 active months. Reported for information; the
    money uses the flat horizon (see HORIZONS_MONTHS)."""
    rate = x / T if T > 0 else 0.0
    if rate <= 0:
        return 0.0
    return min(float(horizon), expected_orders(params, x, t_x, T, float(horizon)) / rate)


# --- The block --------------------------------------------------------------------------------

@lru_cache(maxsize=8)
def _fit_cached(key: tuple) -> dict:
    frame = pd.DataFrame(list(key), columns=["account_id", "orders", "frequency", "recency", "age"]).set_index("account_id")
    return fit_bg_nbd(frame)


def fit_book(df: pd.DataFrame) -> tuple[pd.DataFrame, dict | None]:
    """The per-account summary and the pooled fit, cached on the summary
    itself so the same file never fits twice. A book with fewer than two
    scorable accounts gets no fit."""
    summary = account_summary(df)
    usable = summary[(summary["orders"] >= MIN_ORDERS) & (summary["age"] >= MIN_HISTORY_MONTHS)]
    if len(usable) < 2:
        return summary, None
    key = tuple(
        (str(a), int(row["orders"]), int(row["frequency"]), round(float(row["recency"]), 4), round(float(row["age"]), 4))
        for a, row in usable.iterrows()
    )
    return summary, _fit_cached(key)


def account_lifetime_value(df: pd.DataFrame, account_id: str,
                           fitted: tuple[pd.DataFrame, dict | None] | None = None) -> dict:
    """The `lifetime_value` block for one account."""
    summary, params = fitted if fitted is not None else fit_book(df)
    if account_id not in summary.index:
        return {"status": "insufficient_history", "reason": "no orders"}
    row = summary.loc[account_id]
    orders, x, t_x, T = int(row["orders"]), float(row["frequency"]), float(row["recency"]), float(row["age"])

    if orders < MIN_ORDERS or T < MIN_HISTORY_MONTHS:
        return {
            "status": "insufficient_history",
            "orders": orders,
            "months_of_history": round(T, 1),
            "reason": (f"needs at least {MIN_ORDERS} orders and {MIN_HISTORY_MONTHS} months of history "
                       f"to project; has {orders} orders over {T:.1f} months"),
        }
    if params is None:
        return {"status": "insufficient_history", "orders": orders, "months_of_history": round(T, 1),
                "reason": "too few scorable accounts in the file to fit the book"}

    acc_df = df[df["account_id"] == account_id]
    value = _value_per_month(acc_df)
    if value["baseline"] is None or value["recent"] is None:
        return {"status": "insufficient_history", "orders": orders, "months_of_history": round(T, 1),
                "reason": "no baseline window to compare the recent value per month against"}

    horizons = {}
    for horizon in HORIZONS_MONTHS:
        exp_orders = expected_orders(params, x, t_x, T, float(horizon))
        active = expected_active_months(params, x, t_x, T, horizon)
        # Value per month x the horizon, flat. The model's expected months
        # sit alongside for information and do not scale the money.
        # Rounded first, then differenced, so the gap on screen is exactly
        # the difference of the two tiles beside it.
        baseline_value = round(horizon * value["baseline"])
        current_value = round(horizon * value["recent"])
        horizons[str(horizon)] = {
            "months": horizon,
            "expected_orders": round(exp_orders, 2),
            "expected_active_months": round(active, 2),
            "cltv_baseline": baseline_value,
            "cltv_current": current_value,
            "value_change": current_value - baseline_value,
            "value_at_risk": max(0, baseline_value - current_value),
        }

    longest = max(HORIZONS_MONTHS)
    paths, deviation_month, lost_since_deviation = _value_paths(acc_df, value, longest)

    breakdown = {
        "horizon_months": longest,
        "category": _value_breakdown(acc_df, value["basis"], "category", float(longest)),
        "product": _value_breakdown(acc_df, value["basis"], "product_id", float(longest), limit=MAX_PRODUCTS_LISTED),
        "note": (
            "Lines share the account's expected months of continued buying, so their values add "
            "back to the account total. Line losses can exceed the account's value at risk because "
            "lines that grew offset part of the loss."
        ),
    }

    return {
        "value_paths": paths,
        "deviation_month": deviation_month,
        "lost_since_deviation": lost_since_deviation,
        "breakdown": breakdown,
        "status": "scored",
        "value_basis": value["basis"],
        "orders": orders,
        "months_of_history": round(T, 1),
        "months_since_last_order": round(T - t_x, 1),
        "p_active_now": round(p_alive(params, x, t_x, T), 3),
        "value_per_month_baseline": value["baseline"],
        "value_per_month_recent": value["recent"],
        "horizons_months": horizons,
        "book_fit": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in params.items()},
        "note": (
            "Lifetime value = value per month x the horizon months (12 or 24), flat. Expected orders "
            "and expected active months (BG/NBD, fitted on the whole book) are reported alongside "
            "for information and do not scale the money. Baseline uses the value per month "
            "before the recent window; current uses the recent window. Value at risk is the gap — "
            "the same loss the monthly figures show, on a longer clock. Never add it to monthly "
            "revenue at risk."
        ),
    }


# --- Around the block: position, options, accuracy ----------------------------------

def book_position(blocks: dict[str, dict], account_id: str, horizon: str = "24") -> dict | None:
    """Where one account sits in the book by value at risk: rank, how many
    were scored, and its share of the book's total at risk."""
    at_risk = {
        a: b["horizons_months"][horizon]["value_at_risk"]
        for a, b in blocks.items() if (b or {}).get("status") == "scored"
    }
    if account_id not in at_risk:
        return None
    ordered = sorted(at_risk, key=lambda a: (-at_risk[a], a))
    total = sum(at_risk.values())
    return {
        "rank": ordered.index(account_id) + 1,
        "of": len(ordered),
        "share_of_book_at_risk": round(at_risk[account_id] / total, 4) if total > 0 else 0.0,
        "book_total_at_risk": round(total),
    }


def lifetime_effect_of_options(block: dict, options: list[dict], horizon: str = "24") -> list[dict]:
    """What each priced intervention recovers over the horizon: its monthly
    margin recovery times the same flat horizon months the projection
    uses, and the share of the value at risk that closes."""
    if (block or {}).get("status") != "scored" or not options:
        return []
    months = block["horizons_months"][horizon]["months"]
    at_risk = block["horizons_months"][horizon]["value_at_risk"]
    rows = []
    for option in options:
        per_month = float(option.get("recovers_margin_per_month") or option.get("recovers_revenue_per_month") or 0)
        if per_month <= 0:
            continue
        lifetime = per_month * months
        rows.append({
            "lever_label": option.get("lever_label"),
            "option": option.get("option"),
            "recovers_per_month": round(per_month),
            "recovers_over_horizon": round(lifetime),
            "share_of_value_at_risk": round(min(1.0, lifetime / at_risk), 4) if at_risk > 0 else None,
            "restores_account_to_healthy": bool(option.get("restores_account_to_healthy")),
        })
    return rows


def backtest(df: pd.DataFrame, holdout_months: int = 6) -> dict:
    """Fit on history minus the last `holdout_months`, predict them, compare.

    The pass check for the projection: a model that cannot reproduce the
    last six months on the book it was fitted to has no business projecting
    the next twenty-four. Returns the per-account rows and the book-level
    errors; the script prints them and the page quotes the order error.
    """
    end = df["date"].max()
    cutoff = end - pd.Timedelta(days=holdout_months * DAYS_PER_MONTH)
    train = df[df["date"] <= cutoff]
    summary, params = fit_book(train)
    if params is None:
        return {"status": "insufficient_book", "holdout_months": holdout_months}

    basis = "margin" if "margin" in df.columns and df["margin"].notna().any() else "revenue"
    held = df[df["date"] > cutoff]
    held_orders = _sales_orders(held).groupby("account_id").size()
    held_value = held.groupby("account_id")[basis].sum()
    train_value_per_month = train.groupby("account_id")[basis].sum() / summary["age"].clip(lower=1.0)

    rows = []
    for account_id, row in summary.iterrows():
        if row["orders"] < MIN_ORDERS or row["age"] < MIN_HISTORY_MONTHS:
            continue
        x, t_x, T = float(row["frequency"]), float(row["recency"]), float(row["age"])
        predicted = expected_orders(params, x, t_x, T, float(holdout_months))
        active = expected_active_months(params, x, t_x, T, holdout_months)
        rows.append({
            "account_id": str(account_id),
            "predicted_orders": round(predicted, 1),
            "actual_orders": int(held_orders.get(account_id, 0)),
            "predicted_value": round(active * float(train_value_per_month.get(account_id, 0.0))),
            "actual_value": round(float(held_value.get(account_id, 0.0))),
        })
    total_pred = sum(r["predicted_orders"] for r in rows)
    total_actual = sum(r["actual_orders"] for r in rows)
    value_pred = sum(r["predicted_value"] for r in rows)
    value_actual = sum(r["actual_value"] for r in rows)
    return {
        "status": "ok",
        "holdout_months": holdout_months,
        "fitted_to": str(cutoff.date()),
        "accounts": len(rows),
        "value_basis": basis,
        "params": params,
        "rows": rows,
        "order_error_pct": round(abs(total_pred - total_actual) / total_actual * 100, 1) if total_actual else None,
        "value_error_pct": round(abs(value_pred - value_actual) / value_actual * 100, 1) if value_actual else None,
        "mean_abs_order_error": round(sum(abs(r["predicted_orders"] - r["actual_orders"]) for r in rows) / len(rows), 2) if rows else None,
        "predicted_orders": round(total_pred, 1),
        "actual_orders": total_actual,
    }
