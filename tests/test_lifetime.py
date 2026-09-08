"""
Customer lifetime value (src/pipeline/lifetime.py).

What must hold:
- The BG/NBD fit on the workbook converges to finite, positive parameters.
- A thin account (ACC-109, five months) is insufficient_history, never a number.
- Value is measured in margin, so the flat-revenue leak (ACC-107) shows a
  large value at risk while a healthy control (ACC-102) shows little.
- Value at risk is the gap between the two paths and is never negative;
  a growing account shows a positive value_change and zero at risk.
- Expected orders grow with the horizon and are finite.
- The block rides on the evidence pack and the chat tool returns it.
- A file without a margin column falls back to revenue and says so.
"""

import functools
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pipeline.aryachat import _execute_chat_tool
from pipeline.evidence import build_evidence_pack
from pipeline.ingest import ingest
from pipeline.lifetime import (
    HORIZONS_MONTHS,
    MIN_HISTORY_MONTHS,
    MIN_ORDERS,
    account_lifetime_value,
    account_summary,
    expected_orders,
    fit_book,
    p_alive,
)

from datasets import MERIDIAN_CSV, thin_transactions


@pytest.fixture(scope="module")
def df():
    clean, _ = ingest(str(MERIDIAN_CSV))
    return clean


@pytest.fixture(scope="module")
def fitted(df):
    return fit_book(df)


def test_summary_has_one_row_per_account_in_months(df):
    summary = account_summary(df)
    assert len(summary) == 18
    row = summary.loc["ACC-101"]
    assert row["orders"] == 30 and row["frequency"] == 29
    assert 22 < row["age"] < 24 and row["recency"] <= row["age"]


def test_book_fit_converges_to_finite_positive_parameters(fitted):
    _, params = fitted
    assert params is not None and params["accounts_fitted"] >= 15
    for key in ("r", "alpha", "a", "b"):
        assert math.isfinite(params[key]) and params[key] > 0, key


def test_thin_account_is_not_scored(df, fitted):
    block = account_lifetime_value(df, "ACC-109", fitted)
    assert block["status"] == "insufficient_history"
    assert block["orders"] < MIN_ORDERS or block["months_of_history"] < MIN_HISTORY_MONTHS
    assert "cltv" not in str(block)


def test_flat_revenue_leak_shows_lifetime_value_at_risk(df, fitted):
    """ACC-107's revenue is flat and its margin rate halved: a revenue CLTV
    would call it healthy; a margin CLTV shows the leak."""
    block = account_lifetime_value(df, "ACC-107", fitted)
    assert block["status"] == "scored" and block["value_basis"] == "margin"
    assert block["value_per_month_recent"] < block["value_per_month_baseline"] * 0.6
    for horizon in HORIZONS_MONTHS:
        h = block["horizons_months"][str(horizon)]
        assert h["value_at_risk"] > 0
        assert h["cltv_current"] < h["cltv_baseline"]
        assert h["value_at_risk"] == h["cltv_baseline"] - h["cltv_current"]
    # Over 24 months the leak costs more than over 12.
    assert block["horizons_months"]["24"]["value_at_risk"] > block["horizons_months"]["12"]["value_at_risk"]


def test_healthy_control_has_little_at_risk_and_a_grower_shows_none(df, fitted):
    """Value is per MONTH: ACC-102 now splits the same spend into more,
    smaller orders, so a per-order value would show a fifth of it at risk;
    per month it is flat. ACC-108 (fragmentation) is the same trap."""
    control = account_lifetime_value(df, "ACC-102", fitted)
    leak = account_lifetime_value(df, "ACC-101", fitted)
    assert control["status"] == "scored" and leak["status"] == "scored"
    assert control["horizons_months"]["24"]["value_at_risk"] < leak["horizons_months"]["24"]["value_at_risk"] * 0.1
    fragmenter = account_lifetime_value(df, "ACC-108", fitted)
    assert fragmenter["horizons_months"]["24"]["value_at_risk"] < leak["horizons_months"]["24"]["value_at_risk"] * 0.3
    grower = account_lifetime_value(df, "ACC-105", fitted)
    assert grower["horizons_months"]["24"]["value_change"] > 0
    assert grower["horizons_months"]["24"]["value_at_risk"] == 0


def test_expected_orders_grow_with_the_horizon_and_p_alive_is_a_probability(df, fitted):
    summary, params = fitted
    row = summary.loc["ACC-101"]
    x, t_x, T = float(row["frequency"]), float(row["recency"]), float(row["age"])
    e6, e12, e24 = (expected_orders(params, x, t_x, T, t) for t in (6.0, 12.0, 24.0))
    assert 0 < e6 < e12 < e24 and math.isfinite(e24)
    # A monthly buyer should be expected to keep ordering roughly monthly.
    assert 6 < e12 < 20
    assert 0 <= p_alive(params, x, t_x, T) <= 1
    assert p_alive(params, 0.0, 0.0, T) == 1.0


def test_block_rides_on_the_pack_off_the_prompt_and_the_chat_tool_returns_it(df):
    """The projection is a display key: the Lifetime value page and the chat
    read it, the investigator never sees it, so the tuned verdict prompt is
    unchanged and a projection of the leak's cost cannot feed the verdict."""
    from pipeline.agent import pack_for_prompt
    pack = build_evidence_pack(df, "ACC-101")
    assert pack["_lifetime_value"]["status"] == "scored"
    assert "_lifetime_value" not in pack_for_prompt(pack)
    assert not any("lifetime" in k for k in pack_for_prompt(pack))
    pack_for = functools.lru_cache(maxsize=None)(lambda a: build_evidence_pack(df, a))
    block, touched = _execute_chat_tool(df, pack_for, None, "get_lifetime_value", {"account_id": "ACC-101"})
    assert touched == ["ACC-101"] and block["horizons_months"]["24"]["cltv_baseline"] > 0


def test_value_paths_coincide_until_the_deviation_month_then_widen_by_the_value_at_risk(df, fitted):
    """Potential and reality are the same money while the account was
    itself; they part the month after it last earned its baseline average
    (ACC-101: October 2025, when margin fell from ~₹96k to ₹79k a month and
    never came back); the gap at today is what has been lost so far, and
    over the next 24 months it grows by exactly the value at risk."""
    block = account_lifetime_value(df, "ACC-101", fitted)
    rows = block["value_paths"]
    assert block["deviation_month"] == "2025-10"
    history = [r for r in rows if r["phase"] == "history"]
    projection = [r for r in rows if r["phase"] == "projection"]
    assert [r["offset"] for r in projection] == list(range(1, 25))
    assert history[0]["month"] == "2024-09" and history[-1]["offset"] == 0
    before = [r for r in history if r["month"] < "2025-10"]
    assert before and all(r["potential"] == r["reality"] for r in before)
    after = [r for r in history if r["month"] >= "2025-10"]
    assert all(r["potential"] > r["reality"] for r in after)
    today = history[-1]
    assert today["potential"] - today["reality"] == block["lost_since_deviation"] > 0
    end = projection[-1]
    h24 = block["horizons_months"]["24"]
    assert abs((end["potential"] - today["potential"]) - h24["cltv_baseline"]) <= 1
    assert abs((end["reality"] - today["reality"]) - h24["cltv_current"]) <= 1
    assert abs((end["potential"] - end["reality"]) - (block["lost_since_deviation"] + h24["value_at_risk"])) <= 2
    # Reality is the account's real money: at today it is the sum of every margin line.
    actual = df[df["account_id"] == "ACC-101"]["margin"].sum()
    assert abs(today["reality"] - actual) <= 1


def test_value_paths_have_no_deviation_on_an_account_still_earning_its_average(df, fitted):
    """ACC-111 trades up: its last month is at or above its baseline, so
    potential and reality are identical through history and split only at
    today, with reality above."""
    block = account_lifetime_value(df, "ACC-111", fitted)
    assert block["deviation_month"] is None and block["lost_since_deviation"] == 0
    history = [r for r in block["value_paths"] if r["phase"] == "history"]
    assert all(r["potential"] == r["reality"] for r in history)
    assert block["value_paths"][-1]["reality"] >= block["value_paths"][-1]["potential"]


def test_book_position_ranks_by_value_at_risk(df, fitted):
    from pipeline.lifetime import book_position
    blocks = {a: account_lifetime_value(df, a, fitted) for a in ("ACC-101", "ACC-102", "ACC-107", "ACC-109")}
    top = book_position(blocks, "ACC-101")
    assert top["rank"] == 1 and top["of"] == 3, "ACC-109 is unscored and not counted"
    assert 0 < top["share_of_book_at_risk"] <= 1
    assert book_position(blocks, "ACC-109") is None


def test_options_carry_over_the_horizon_as_a_share_of_the_gap(df, fitted):
    from pipeline.decide import price_options
    from pipeline.lifetime import lifetime_effect_of_options
    pack = build_evidence_pack(df, "ACC-101")
    block = pack["_lifetime_value"]
    effects = lifetime_effect_of_options(block, price_options(pack, None))
    assert effects and all(e["recovers_over_horizon"] > 0 for e in effects)
    months = block["horizons_months"]["24"]["months"]
    assert months == 24, "money is value per month x the flat horizon"
    assert effects[0]["recovers_over_horizon"] == round(effects[0]["recovers_per_month"] * months)
    assert all(0 < e["share_of_value_at_risk"] <= 1 for e in effects)


def test_backtest_reproduces_the_held_out_months_within_tolerance(df):
    from pipeline.lifetime import backtest
    result = backtest(df, holdout_months=6)
    assert result["status"] == "ok" and result["accounts"] >= 15
    assert result["order_error_pct"] is not None and result["order_error_pct"] <= 25
    assert {"account_id", "predicted_orders", "actual_orders"} <= set(result["rows"][0])


def test_breakdown_parts_add_back_to_the_account_and_name_the_leaking_lines(df, fitted):
    block = account_lifetime_value(df, "ACC-101", fitted)
    breakdown = block["breakdown"]
    total = block["horizons_months"][str(breakdown["horizon_months"])]
    for level in ("category", "product"):
        rows = breakdown[level]
        assert rows, level
        # Each line shares the account's expected months, so the parts sum to
        # the account total (to rounding across ~20 lines).
        assert abs(sum(r["cltv_baseline"] for r in rows) - total["cltv_baseline"]) < 100
        assert abs(sum(r["cltv_current"] for r in rows) - total["cltv_current"]) < 100
        # Losses on the lines exceed the account's net at-risk: the cheap lines
        # that grew offset part of the loss — the rotation made visible.
        assert sum(r["value_at_risk"] for r in rows) >= total["value_at_risk"]
        assert all(r["value_at_risk"] >= 0 for r in rows)
        assert rows == sorted(rows, key=lambda r: (-r["value_at_risk"], -r["cltv_baseline"], r["name"]))
    assert breakdown["category"][0]["name"] == "Power Tools"
    assert breakdown["product"][0]["name"] == "PT-Cordless Drill 18V"
    grown = next(r for r in breakdown["category"] if r["name"] == "Consumables")
    assert grown["value_change"] > 0 and grown["value_at_risk"] == 0


def test_a_file_without_margin_measures_value_in_revenue():
    thin, _ = ingest(thin_transactions(with_margin=False))
    fitted = fit_book(thin)
    block = account_lifetime_value(thin, "ACC-101", fitted)
    assert block["status"] == "scored" and block["value_basis"] == "revenue"
    assert block["horizons_months"]["12"]["cltv_baseline"] > 0
