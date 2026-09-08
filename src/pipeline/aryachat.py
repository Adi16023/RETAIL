"""
AryaChat — the analyst a manager talks to about the whole book of accounts.

This is the fourth LLM call site and it is justified the same way the other
two optional agents are: answering a free-form question is a judgement about
which facts matter and how to say them, not a threshold. It sits on top of
everything the pipeline has already computed and can never change a verdict.

The shape is the ARYA MCP chat loop (youkti-arya `core/mcp_chat/loop.py`):
the model is given a small amount of standing context and a CATALOGUE OF
TOOLS, decides which tool answers the question, reads the result, and
answers from it — LLM → tools → results → LLM until it answers without a
tool call. Progress is emitted as EVENTS (text deltas, a tool_start /
tool_end pair per call, next-question chips) so the screen can show the
answer being written instead of a spinner. Nothing here knows about
Streamlit.

What the model is handed up front is deliberately small: an ACCOUNT INDEX
(id, name, region, verdict if one exists — enough to resolve "Northgate" to
ACC-101) and, when the chat has an ACCOUNT IN FOCUS, that account's brief
(verdict, money, six signals in plain words, KPI tiles, dated events).
Everything else is a tool: the book as a table, any account's brief, a
comparison digest for several accounts, and for one account its orders,
its products, its category mix, its month-by-month table, the harmless
explanations that were ruled out and the priced interventions, plus the
five drill-downs the investigator itself uses. Every tool returns figures
computed by deterministic code from the order file; the model quotes, it
never computes.

Focus is sticky. Whichever account the last answer's tools touched becomes
the account in focus, so "and the margin?" on the next turn is about it.
The caller persists it with the chat.

Three things are ours rather than inherited from that loop:

* **The figure check.** Instead of trusting the model to list what it
  quoted, every number in a finished answer is checked against the index,
  the focus brief, this turn's tool results and the earlier exchange;
  anything found nowhere is returned as `unsourced_figures` and the screen
  flags it. A figure from nowhere is the one failure that makes this
  system untrustworthy, so it is measured, not declared.
* **Required tools.** A product / seasonal / pricing question about one
  account answered without the tool that holds the answer is sent back
  once (measured live: "which products dropped" answered from headlines
  named two and missed the biggest three). The rejected draft was already
  streamed, so a `retry` event tells the screen to clear it.
* **No confirm gate, no hardening.** Every tool reads the order file
  through deterministic code; nothing reaches the outside world and no
  result is third-party text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Optional

import pandas as pd

from .agent import DRILLDOWN_TOOLS, DEFAULT_MODEL, _execute_tool, compact_json
from .changepoint import RECENT_MONTHS
from .compare import build_comparison_digest
from .decide import price_options
from .monthly import monthly_table
from .timeline import build_timeline

MAX_ITERATIONS = 8
# What an Anthropic model needs; the Groq shim clamps this to its own ceiling.
# The open-source model is a reasoning model whose thinking is charged to the
# same budget — at 2,048 it ran out of room after one tool result and
# returned nothing. Same figure the investigator asks for.
OUTPUT_TOKENS = 16000
# Turns kept verbatim in the prompt (a turn is one user OR one assistant
# message, so 6 turns is the last three exchanges).
KEEP_LAST_TURNS = 6
# Once this many un-summarised turns have built up, the UI compacts before the
# next question. The opening context is ~1,500 tokens; a dozen short turns is
# another ~1,500.
COMPACT_AFTER_TURNS = 12
MAX_TIMELINE_EVENTS = 12
# One tool result may not exceed this many characters of prompt. The cap
# exists so no tool can quietly blow the free tier's 8,000 tokens-per-minute
# budget. The cut is marked, so the model knows it saw a prefix.
MAX_TOOL_RESULT_CHARS = 12000
# What the screen keeps of a tool result, per call, so "how did it know
# that?" can be answered after the fact without storing the whole payload.
RESULT_PREVIEW_CHARS = 500
MAX_FOLLOW_UPS = 3
# The account index is sent on every turn, so it is one line per account and
# capped; a bigger book is reached through find_account and get_account_book.
MAX_INDEX_ACCOUNTS = 60
MAX_COMPARE_ACCOUNTS = 8
MAX_ORDERS_LISTED = 60
MAX_PRODUCTS_LISTED = 40
MAX_FIND_MATCHES = 10


class ChatError(Exception):
    """The model never produced an answer (or, for compaction, a summary)."""


# --- What the model is told ------------------------------------------------------

SYSTEM_PROMPT = """You are AryaChat, the analyst a B2B account manager talks to about their book of customer \
accounts. You answer from exactly two sources: the ACCOUNT INDEX and ACCOUNT IN FOCUS in the first \
message, and the tools, which return figures computed by deterministic code from the accounts' own \
order history. You have no other knowledge of these customers.

WORK LIKE AN ANALYST WITH A TOOLKIT. Read the question, decide which tool holds the answer, call \
it, and answer from what came back. Work in whole steps: if the answer needs an account resolved \
and then its products, do both, then answer. Prefer one tool that answers the question over several \
that circle it; never call a tool whose result you will not use.

WHICH ACCOUNT. A question that names an account (by id like ACC-104, by name like "Northgate", or by \
a fragment) is about that account: resolve it from the index, or with find_account when unsure, and \
pass its account_id to every tool. A question that names no account is about the ACCOUNT IN FOCUS. \
If there is none and the question needs one, ask which account — naming two or three likely \
candidates from the index — instead of guessing. A question about several accounts or the whole book \
("which accounts…", "who is worst", "compare A and B") uses get_account_book or compare_accounts.

EVERY NUMBER YOU STATE MUST APPEAR VERBATIM IN THE OPENING MESSAGE OR IN A TOOL RESULT. Never \
calculate, total, average, subtract, annualise, round, convert or project a figure — not even to \
tidy it. If the figure the question needs is not in front of you, call the tool that has it. If no \
tool has it, say so. Every answer is checked afterwards: a figure that appears nowhere is flagged \
to the manager as unsupported.

CALL THE TOOL BEFORE ANSWERING A DETAIL QUESTION. A brief carries the headline picture only. WHICH \
products -> get_product_changes (what moved) or get_product_revenue (what they buy); WHAT happened \
in a month or the trend -> get_monthly_table; HOW they order (cadence, basket, individual orders) \
-> get_orders; WHAT they buy by category -> get_category_mix; WHETHER a dip is seasonal, recovered, \
a data gap or a bulk month -> get_ruled_out_checks; WHAT could be done and what it is worth -> \
get_priced_options; WHAT the account is worth over the next 12 or 24 months, or what the leak \
costs over its lifetime -> get_lifetime_value; WHY the verdict -> get_account_brief. When asked which products "dropped", \
report BOTH the products that stopped and the ones that declined, each with what it was worth per \
month before and after.

NEVER PREDICT THE CUSTOMER. Order history cannot say how a buyer will react, whether a win-back \
will succeed, or why they changed. Break-even figures in the priced options are tolerances, not \
forecasts.

SAY PLAINLY WHAT THE DATA CANNOT ANSWER. Contract terms, competitors, the buyer's reasons, anything \
outside the order file: answer the part the data supports, then say in one plain sentence what the \
order data cannot settle. Do not fill the gap with a guess.

WRITE FOR A MANAGER, NOT FOR A PAYLOAD READER. Lead with the answer, then the detail, in at most \
120 words. Business words and formatted figures: "Rs 88,810 a month", "32.2%", "April 2026", "down \
12 points". A short table (at most 6 rows, the columns that answer the question, never an id \
column except the account id) when listing accounts, products, orders or months; otherwise plain \
sentences. Never a p-value, a raw ratio like 0.3215, "pp", a field name, a tool name, or this \
system's internal status names (erosion_detected, creep_detected, material_decline, \
downgrade_detected, premiumisation_detected, fragmentation_detected, baskets_shrinking, mild_drift, \
defected, change-point) or reworded versions of them. Never narrate your own mechanics ("I called \
the tool", "the brief says"). Say what the customer did.

THE VERDICT IS NOT YOURS TO CHANGE. If asked whether an investigation is right, lay out the \
evidence for and against from the tools. Do not issue a different verdict. An account marked "not \
analysed" has no verdict yet; say so and answer from the evidence.

EARLIER IN THIS CHAT. The opening message may carry a summary of earlier turns and the messages \
after it are the recent exchange. Treat both as established; do not re-answer what was already \
settled unless asked again.

NEXT-QUESTION CHIPS
End every finished answer with 2-3 chips — what the manager would actually ask next — each on its \
own line, in exactly this form, with nothing after them:
<follow_up>Which Power Tools products fell at Northgate Traders?</follow_up>
The chip text IS the message sent when it is tapped, and the manager reads it as-is: one short \
question (roughly 4-10 words), self-contained, naming the specific account, categories, products or \
months from THIS answer so it works verbatim as the next message. Each chip is something your tools \
can answer, one step FORWARD:
  an account named -> why it is flagged, what changed and when, how it compares to a peer
  a category or product moved -> which products, since when, is it seasonal
  a dip -> has it recovered, was it a one-off bulk month or a data gap
  a margin or discount change -> the month-by-month picture, what a win-back is worth
  several accounts -> compare them, which to call first
Never: generic ("tell me more"), a repeat of what was just answered, anything the order data cannot \
answer. Omit chips when nothing useful follows."""


SUBMIT_SUMMARY_TOOL = {
    "name": "submit_summary",
    "description": "Submit the compacted summary of the earlier conversation. Call this exactly once.",
    "input_schema": {
        "type": "object",
        "properties": {
            "summary": {
                "type": "string",
                "description": "At most 150 words. What was asked and what was established, which accounts were discussed, keeping every figure exactly as it was stated. Business words only.",
            },
        },
        "required": ["summary"],
        "additionalProperties": False,
    },
    "strict": True,
}

COMPACT_PROMPT = """You compact the earlier part of a conversation between an account manager and an \
analyst about the manager's book of customer accounts, so the analyst can keep answering with the \
context intact. Write at most 150 words: which accounts were discussed, what was asked, what was \
established, and every figure that was stated, copied exactly as written — never recomputed, never \
rounded. Business words only; no internal status names. Call submit_summary exactly once."""


# --- Plain words for detector statuses ---------------------------------------------
#
# Briefs are model-facing, so they could carry the codes — but the prompt
# forbids the codes in the answer, and handing the model the words it is
# allowed to use is what makes that rule easy to keep.

_PLAIN_STATUS = {
    "revenue": {
        "material_decline": "spending significantly less than before",
        "mild_drift": "drifting slightly lower",
        "stable": "steady",
        "growth": "growing",
    },
    "margin": {
        "erosion_detected": "margin rate has fallen",
        "stable": "steady",
        "improvement_detected": "margin rate has improved",
    },
    "discount": {
        "creep_detected": "discount has crept up",
        "stable": "steady",
        "discipline_improved": "discount has tightened",
    },
    "tier_mix": {
        "downgrade_detected": "spend is moving into cheaper lines",
        "stable": "steady",
        "premiumisation_detected": "spend is moving into premium lines",
    },
    "order_pattern": {
        "fragmentation_detected": "more orders, each one smaller",
        "baskets_shrinking": "each order is smaller than before",
        "stable": "steady",
    },
}

_PLAIN_VERDICT = {
    "leakage_detected": "leakage detected",
    "healthy": "healthy",
    "insufficient_data": "not enough evidence to call",
}


def _plain(dimension: str, block: dict | None, available: bool) -> str:
    if not available or block is None:
        return "not measured (column absent from the file)"
    status = block.get("status")
    if status == "insufficient_history":
        return "not enough history to judge"
    return _PLAIN_STATUS.get(dimension, {}).get(status, "steady")


def _signals(pack: dict) -> dict:
    dimensions = pack.get("analysis_dimensions") or {}
    return {
        "revenue": _plain("revenue", pack.get("revenue_decline"), dimensions.get("revenue", True)),
        "margin": _plain("margin", pack.get("margin"), dimensions.get("margin", True)),
        "discount": _plain("discount", pack.get("discount"), dimensions.get("discount", True)),
        "tier_mix": _plain("tier_mix", pack.get("tier_mix"), dimensions.get("tier_mix", True)),
        "order_pattern": _plain("order_pattern", pack.get("order_pattern"), dimensions.get("order_pattern", True)),
    }


def _verdict_line(report: dict | None) -> str:
    if not report:
        return "not analysed"
    if report.get("defer"):
        return "deferred — not enough evidence to call"
    verdict = _PLAIN_VERDICT.get(report.get("verdict"), str(report.get("verdict")))
    priority = (report.get("prioritization") or {}).get("priority")
    shape = (report.get("temporary_or_structural") or "").replace("_", " ")
    parts = [verdict]
    if shape and shape != "not applicable":
        parts.append(shape)
    if priority:
        parts.append(f"priority {priority}")
    return ", ".join(parts)


def _pct(value) -> float | None:
    return None if value is None else round(float(value) * 100, 1)


def _money(value) -> float | None:
    return None if value is None else round(float(value))


# --- Briefs ------------------------------------------------------------------------------

def kpis_from_pack(pack: dict) -> dict:
    """The six dashboard tiles, baseline against recent, from the pack alone.

    The same six figures the KPI row shows, so an answer and the tile above
    it can never disagree. Computed nowhere else — this reads the profiles
    Stage 2 already built.
    """
    revenue = pack.get("overall_revenue") or {}
    margin = pack.get("margin_profile") or {}
    discount = pack.get("discount") or {}
    tier = pack.get("tier_mix") or {}
    orders = pack.get("order_behavior") or {}
    high_base = (tier.get("baseline_share_by_tier") or {}).get("High")
    high_recent = (tier.get("recent_share_by_tier") or {}).get("High")

    def tile(label, baseline, recent, unit, scale=1.0, places=1):
        """Round for display AFTER taking the difference on the raw values —
        the same order the KPI row uses — so the change here is the change
        on the tile (32.15 -> 19.89 is "down 12.3 points" on both, not 12.2)."""
        if baseline is None or recent is None:
            change = None
        else:
            change = round((float(recent) - float(baseline)) * scale, places)
        shown = lambda v: None if v is None else round(float(v) * scale, places)  # noqa: E731
        return {"label": label, "baseline": shown(baseline), "recent": shown(recent),
                "change": change, "unit": unit}

    return {
        "revenue_per_month": tile("Revenue per month (Rs)", revenue.get("baseline_monthly_median"),
                                  revenue.get("recent_monthly_rate"), "Rs", places=0),
        "margin_rate": tile("Margin rate (%)", margin.get("baseline_margin_pct"),
                            margin.get("recent_margin_pct"), "% (change in points)", scale=100),
        "high_tier_share": tile("High-tier share of spend (%)", high_base, high_recent,
                                "% (change in points)", scale=100),
        "average_discount": tile("Average discount (%)", discount.get("baseline_avg_discount_pct"),
                                 discount.get("recent_avg_discount_pct"), "% (change in points)", scale=100),
        "orders_per_month": tile("Orders per month", orders.get("order_frequency_baseline_per_month"),
                                 orders.get("order_frequency_recent_per_month"), "orders", places=2),
        "lines_per_order": tile("Lines per order", orders.get("basket_width_baseline"),
                                orders.get("basket_width_recent"), "lines", places=2),
    }


def build_account_brief(pack: dict, report: dict | None = None) -> dict:
    """What the model is shown about one account before any question about it.

    A digest, not the pack: the verdict in business terms, the money, the
    six signals in plain words, the KPI tiles, the categories that stopped,
    and the dated events. Detail lives behind the tools. Nothing under an
    underscore key and no transaction row can reach this.
    """
    history = pack.get("history") or {}
    sufficiency = pack.get("data_sufficiency") or {}

    brief = {
        "account": {
            "account_id": pack.get("account_id"),
            "account_name": pack.get("account_name"),
            "region": pack.get("region"),
            "account_manager": pack.get("account_manager"),
            "account_manager_changed_mid_history": bool(pack.get("account_manager_changed")),
            "months_of_history": history.get("months_of_history"),
            "orders_in_history": history.get("order_count"),
            "data_sufficiency": sufficiency.get("label"),
            "history_window": f"{history.get('start_date')} to {history.get('end_date')}",
        },
        "signals": _signals(pack),
        "kpis_baseline_vs_recent": kpis_from_pack(pack),
        "categories_that_stopped": [
            {"category": c["category"], "months_at_zero": c.get("consecutive_months_at_zero")}
            for c in (pack.get("category_changes") or []) if c.get("defected")
        ],
        "what_changed": [
            {"when": e["when"], "headline": e["headline"], "detail": e["detail"]}
            for e in build_timeline(pack) if e.get("notable")
        ][:MAX_TIMELINE_EVENTS],
    }

    if report:
        impact = report.get("financial_impact") or {}
        margin_impact = impact.get("margin_impact") or {}
        priority = report.get("prioritization") or {}
        brief["verdict"] = {
            "verdict": report.get("verdict"),
            "temporary_or_structural": report.get("temporary_or_structural"),
            "confidence": report.get("confidence"),
            "deferred": bool(report.get("defer")),
            "value_is_leaving_through": report.get("leak_dimensions") or [],
            "attributed_categories": report.get("attributed_categories") or [],
            "summary": report.get("narrative"),
            "recommended_actions": report.get("recommended_actions") or [],
        }
        brief["money"] = {
            "revenue_at_risk_per_month_rs": _money(impact.get("total_monthly_revenue_at_risk")),
            "margin_at_risk_per_month_rs": _money(margin_impact.get("monthly_margin_at_risk")),
            "severity_pct_of_baseline": _pct(impact.get("overall_severity_pct_of_baseline")),
            "priority": priority.get("priority"),
            "exposure_over_12_months_rs": _money(max(
                (priority.get("churn_risk_projection") or {}).get("projected_12_month_loss_if_unaddressed") or 0,
                (priority.get("churn_risk_projection") or {}).get("projected_12_month_margin_loss_if_unaddressed") or 0,
            )) or None,
        }
    else:
        brief["verdict"] = "not yet investigated — the AI verdict has not been run for this account"

    brief["note"] = (
        "Every figure here and in every tool result was computed by code from this account's "
        "orders. Quote figures exactly; never compute a new one."
    )
    return brief


def _identity_column(df: pd.DataFrame, column: str) -> dict[str, str]:
    if column not in df.columns:
        return {}
    return (df.drop_duplicates("account_id").set_index("account_id")[column]
            .fillna("").astype(str).to_dict())


def build_account_index(df: pd.DataFrame, report_for: Callable[[str], dict | None] | None = None) -> dict:
    """The standing context for the whole book: one short line per account,
    enough to resolve a name to an id and to know who has a verdict. Sent on
    every turn, so it carries no figures — the book as a table is a tool.
    Capped; a bigger book is reached through find_account."""
    names, regions = _identity_column(df, "account_name"), _identity_column(df, "region")
    ids = sorted(df["account_id"].unique())
    rows = []
    for account_id in ids[:MAX_INDEX_ACCOUNTS]:
        row = {"account_id": account_id, "name": names.get(account_id) or None,
               "region": regions.get(account_id) or None}
        if report_for is not None:
            row["verdict"] = _verdict_line(report_for(account_id))
        rows.append(row)
    index = {"accounts_in_book": len(ids), "accounts": rows}
    if len(ids) > MAX_INDEX_ACCOUNTS:
        index["note"] = (f"{len(ids) - MAX_INDEX_ACCOUNTS} more accounts are not listed; use "
                         "find_account to resolve one, get_account_book for all.")
    return index


# --- Tools --------------------------------------------------------------------------------

_ACCOUNT_PROPERTY = {
    "account_id": {"type": "string", "description": "The account's id from the index, e.g. ACC-104."}
}


def _account_scoped(tool: dict) -> dict:
    """An investigator drill-down, widened to take the account it is about.
    Stage 4 runs on one account and needs no id; the chat spans the book."""
    schema = tool["input_schema"]
    return {
        **tool,
        "description": tool["description"] + " Pass the account_id.",
        "input_schema": {
            **schema,
            "properties": {**_ACCOUNT_PROPERTY, **schema.get("properties", {})},
            "required": ["account_id", *schema.get("required", [])],
        },
    }


def _account_tool(name: str, description: str) -> dict:
    return {
        "name": name,
        "description": description,
        "input_schema": {"type": "object", "properties": dict(_ACCOUNT_PROPERTY),
                         "required": ["account_id"], "additionalProperties": False},
        "strict": True,
    }


BOOK_TOOLS = [
    {
        "name": "find_account",
        "description": "Resolve a name, partial name, id fragment, region or account manager to account ids. Use when a question names an account you cannot match in the index with certainty. Returns up to 10 matches.",
        "input_schema": {"type": "object", "properties": {"query": {"type": "string"}},
                         "required": ["query"], "additionalProperties": False},
        "strict": True,
    },
    {
        "name": "get_account_book",
        "description": "Every account as one row: identity, months of history, orders, revenue per month (baseline and recent, with the change), margin rate, discount, high-tier share, the five signals in plain words, categories that stopped, and — where an investigation exists — the verdict, priority (High/Medium/Low), revenue and margin at risk per month and the 12-month exposure. Use for 'which accounts…', 'who should I call first', 'rank by money at risk', 'how many accounts are…' and any question across the book.",
        "input_schema": {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
        "strict": True,
    },
    _account_tool("get_account_brief",
                  "One account's brief: the verdict and money (if investigated), the six signals in plain words, the six KPI tiles baseline vs recent, categories that stopped, and the dated events. Use for 'what is happening on X', 'why is X flagged', and to bring an account into focus."),
    {
        "name": "compare_accounts",
        "description": "Two to eight accounts side by side: for each, identity, history, and the dated events that changed (what, when, how it reads), so the manager can see who to call first and which accounts are living the same story. Use for 'compare A and B', 'is X worse than Y', 'which of these…'.",
        # No minItems/maxItems: the Anthropic API rejects array bounds other
        # than 0 or 1 in a tool schema. The executor enforces 2..MAX_COMPARE_ACCOUNTS.
        "input_schema": {
            "type": "object",
            "properties": {"account_ids": {
                "type": "array", "items": {"type": "string"},
                "description": f"Two to {MAX_COMPARE_ACCOUNTS} account ids from the index.",
            }},
            "required": ["account_ids"], "additionalProperties": False,
        },
        "strict": True,
    },
]

PATTERN_TOOLS = [
    _account_tool("get_orders",
                  "The account's orders, one row each, newest last: date, number of lines, revenue, margin, average discount, categories covered, whether it was a return. Use for cadence and pattern questions — how often they order, how big an order is, when they last ordered, what a particular order contained. Capped at the most recent 60 orders."),
    _account_tool("get_product_revenue",
                  "Every product the account buys: category, tier, monthly revenue in the baseline period and in the last six months, how many orders it appeared in, first and last month bought. Use for 'what do they buy', 'their biggest products', 'what have they never bought recently'. Capped at the 40 largest."),
    _account_tool("get_category_mix",
                  "Revenue by category: baseline vs recent share of spend, and baseline vs recent monthly revenue per category. Use for 'what do they buy by category', 'which categories grew or shrank'."),
    _account_tool("get_monthly_table",
                  "The account's month-by-month table, full history: revenue, margin, margin rate, orders, lines, lines per order, average discount and High-tier share. Use for 'what happened in <month>' and 'show me the trend' questions."),
    _account_tool("get_kpis",
                  "The six headline tiles — revenue per month, margin rate, High-tier share, average discount, orders per month, lines per order — each as baseline, recent (last 6 months) and change."),
    _account_tool("get_ruled_out_checks",
                  "The harmless-explanation checks the analysis ran for this account: was the dip seasonal (same months last year), has it already recovered, months with no orders, one-off bulk months, returns, which way the product mix moved, whether the account manager changed. Use for 'is this seasonal', 'could this be a data problem', 'is it just a big order'."),
    _account_tool("get_lifetime_value",
                  "The account's customer lifetime value: expected orders and months of continued buying over the next 12 and 24 months (a BG/NBD model fitted on the whole book), margin per month in the baseline period and recently, lifetime value on the baseline path and on the current path, the value at risk between them, and the SAME split by category and by product (each line's value per month before and now, baseline path, current path, value at risk). Use for 'what is this account worth', 'how much will the leak cost over time', 'which category or product is costing the most lifetime value'. Reports insufficient_history for thin accounts."),
    _account_tool("get_priced_options",
                  "Every intervention the pipeline could price for this account — what each would recover per month and over 12 months, the resulting margin rate or break-even tolerance, and whether it restores the account to healthy. Empty when nothing has been lost yet or the account is healthy. Use for 'what could we do', 'what is a win-back worth'."),
]

CHAT_TOOLS = BOOK_TOOLS + PATTERN_TOOLS + [_account_scoped(t) for t in DRILLDOWN_TOOLS]


class UnknownAccount(ValueError):
    """The model passed an account_id that is not in the file."""


def find_accounts(df: pd.DataFrame, query: str) -> list[dict]:
    """Accounts whose id, name, region or manager contains `query`
    (case-insensitive), up to MAX_FIND_MATCHES."""
    needle = " ".join((query or "").split()).casefold()
    columns = [c for c in ("account_id", "account_name", "region", "account_manager") if c in df.columns]
    identity = df.drop_duplicates("account_id")[columns].fillna("").astype(str)
    if needle:
        haystack = identity.apply(lambda col: col.str.casefold().str.contains(needle, regex=False), axis=0)
        identity = identity[haystack.any(axis=1)]
    return identity.sort_values("account_id").head(MAX_FIND_MATCHES).to_dict("records")


def _resolve_account(df: pd.DataFrame, account_id) -> str:
    ids = set(df["account_id"].unique())
    candidate = str(account_id or "").strip()
    if candidate in ids:
        return candidate
    if candidate.upper() in ids:
        return candidate.upper()
    matches = find_accounts(df, candidate) if candidate else []
    hint = (" Did you mean: " + ", ".join(
        f"{m['account_id']} ({m.get('account_name', '')})" for m in matches) + "?") if matches else \
        " Call find_account with the customer's name to resolve it."
    raise UnknownAccount(f"No account with id {candidate!r} in this file.{hint}")


def _account_book_row(pack: dict, report: dict | None) -> dict:
    revenue = pack.get("overall_revenue") or {}
    margin = pack.get("margin_profile") or {}
    discount = pack.get("discount") or {}
    tier = pack.get("tier_mix") or {}
    history = pack.get("history") or {}
    return {
        "account_id": pack.get("account_id"),
        "account_name": pack.get("account_name"),
        "region": pack.get("region"),
        "account_manager": pack.get("account_manager"),
        "months_of_history": history.get("months_of_history"),
        "orders": history.get("order_count"),
        "data_sufficiency": (pack.get("data_sufficiency") or {}).get("label"),
        "revenue_per_month_baseline_rs": _money(revenue.get("baseline_monthly_median")),
        "revenue_per_month_recent_rs": _money(revenue.get("recent_monthly_rate")),
        "revenue_change_pct": _pct(revenue.get("pct_change")),
        "margin_rate_recent_pct": _pct(margin.get("recent_margin_pct")),
        "average_discount_recent_pct": _pct(discount.get("recent_avg_discount_pct")),
        "high_tier_share_recent_pct": _pct((tier.get("recent_share_by_tier") or {}).get("High")),
        "signals": _signals(pack),
        "categories_that_stopped": [c["category"] for c in (pack.get("category_changes") or []) if c.get("defected")],
        "verdict": _verdict_line(report),
        # The prioritisation figures, when an investigation exists: what the
        # Prioritise page ranks on, so "who do I call first" can be answered
        # in rupees for investigated accounts and from signals for the rest.
        **_money_at_risk(report),
    }


def _money_at_risk(report: dict | None) -> dict:
    if not report or report.get("defer"):
        return {}
    impact = report.get("financial_impact") or {}
    projection = (report.get("prioritization") or {}).get("churn_risk_projection") or {}
    return {
        "priority": (report.get("prioritization") or {}).get("priority"),
        "revenue_at_risk_per_month_rs": _money(impact.get("total_monthly_revenue_at_risk")),
        "margin_at_risk_per_month_rs": _money((impact.get("margin_impact") or {}).get("monthly_margin_at_risk")),
        "exposure_over_12_months_rs": _money(max(
            projection.get("projected_12_month_loss_if_unaddressed") or 0,
            projection.get("projected_12_month_margin_loss_if_unaddressed") or 0,
        )) or None,
    }


def _tool_get_account_book(df: pd.DataFrame, pack_for, report_for) -> dict:
    ids = sorted(df["account_id"].unique())
    rows = [_account_book_row(pack_for(a), report_for(a)) for a in ids]
    return {"accounts": rows,
            "note": "Recent = the last six months of history; baseline = everything before that."}


def _recent_cutoff(acc_df: pd.DataFrame) -> pd.Period:
    """First month of the recent window — the same split every KPI tile uses."""
    return acc_df["date"].max().to_period("M") - (RECENT_MONTHS - 1)


def _window_months(acc_df: pd.DataFrame) -> tuple[int, int]:
    """(baseline months, recent months) for turning window totals into monthly rates."""
    first = acc_df["date"].min().to_period("M")
    last = acc_df["date"].max().to_period("M")
    cutoff = _recent_cutoff(acc_df)
    baseline = max(0, (cutoff - first).n)
    recent = min(RECENT_MONTHS, (last - first).n + 1)
    return baseline, recent


def _tool_get_orders(df: pd.DataFrame, account_id: str) -> dict:
    acc_df = df[df["account_id"] == account_id]
    rows = []
    for order_id, lines in acc_df.groupby("order_id"):
        row = {
            "order_id": order_id,
            "date": str(lines["date"].min().date()),
            "lines": int(len(lines)),
            "revenue": round(float(lines["revenue"].sum()), 2),
            "categories": sorted(lines["category"].dropna().unique().tolist()) if "category" in lines else [],
        }
        if "margin" in lines and lines["margin"].notna().any():
            row["margin"] = round(float(lines["margin"].sum()), 2)
        if "discount_pct" in lines and lines["discount_pct"].notna().any():
            row["average_discount_percent"] = round(float(lines["discount_pct"].mean()) * 100, 1)
        if "is_return" in lines and int(lines["is_return"].sum()) > 0:
            row["contains_return"] = True
        rows.append(row)
    rows.sort(key=lambda r: (r["date"], r["order_id"]))
    total = len(rows)
    listed = rows[-MAX_ORDERS_LISTED:]
    out = {"orders_in_history": total, "orders": listed}
    if total > len(listed):
        out["note"] = f"Showing the most recent {len(listed)} of {total} orders."
    return out


def _tool_get_product_revenue(df: pd.DataFrame, account_id: str) -> dict:
    acc_df = df[df["account_id"] == account_id].copy()
    acc_df["month"] = acc_df["date"].dt.to_period("M")
    cutoff = _recent_cutoff(acc_df)
    baseline_months, recent_months = _window_months(acc_df)
    rows = []
    for product_id, lines in acc_df.groupby("product_id"):
        baseline = lines[lines["month"] < cutoff]["revenue"].sum()
        recent = lines[lines["month"] >= cutoff]["revenue"].sum()
        row = {
            "product_id": product_id,
            "category": lines["category"].iloc[0] if "category" in lines else None,
            "baseline_monthly_avg": round(float(baseline) / baseline_months, 2) if baseline_months else None,
            "recent_monthly_avg": round(float(recent) / recent_months, 2) if recent_months else None,
            "orders_bought_in": int(lines["order_id"].nunique()),
            "first_month": str(lines["month"].min()),
            "last_month": str(lines["month"].max()),
        }
        if "tier" in lines and lines["tier"].notna().any():
            row["tier"] = str(lines["tier"].dropna().iloc[0])
        rows.append(row)
    rows.sort(key=lambda r: -((r["baseline_monthly_avg"] or 0) + (r["recent_monthly_avg"] or 0)))
    total = len(rows)
    listed = rows[:MAX_PRODUCTS_LISTED]
    out = {"products_in_history": total, "recent_window_starts": str(cutoff), "products": listed}
    if total > len(listed):
        out["note"] = f"Showing the {len(listed)} largest of {total} products."
    return out


def _tool_get_category_mix(df: pd.DataFrame, account_id: str, pack: dict) -> dict:
    acc_df = df[df["account_id"] == account_id].copy()
    acc_df["month"] = acc_df["date"].dt.to_period("M")
    cutoff = _recent_cutoff(acc_df)
    baseline_months, recent_months = _window_months(acc_df)
    mix = pack.get("category_mix") or {}
    baseline_share, recent_share = mix.get("baseline_share") or {}, mix.get("recent_share") or {}
    rows = []
    for category, lines in acc_df.groupby("category"):
        baseline = lines[lines["month"] < cutoff]["revenue"].sum()
        recent = lines[lines["month"] >= cutoff]["revenue"].sum()
        rows.append({
            "category": category,
            "baseline_share_percent": _pct(baseline_share.get(category)),
            "recent_share_percent": _pct(recent_share.get(category)),
            "baseline_monthly_avg": round(float(baseline) / baseline_months, 2) if baseline_months else None,
            "recent_monthly_avg": round(float(recent) / recent_months, 2) if recent_months else None,
        })
    rows.sort(key=lambda r: -(r["baseline_share_percent"] or 0))
    return {"recent_window_starts": str(cutoff), "categories": rows,
            "high_value_categories": mix.get("high_value_categories") or []}


def _tool_get_monthly_table(df: pd.DataFrame, account_id: str) -> list[dict]:
    acc_df = df[df["account_id"] == account_id]
    frame = monthly_table(acc_df)
    keep = [c for c in ("revenue", "margin", "margin_pct", "orders", "lines", "lines_per_order",
                        "discount_pct", "high_tier_share") if c in frame.columns]
    rows = []
    for _, row in frame.iterrows():
        entry = {"month": str(row["period"])}
        for column in keep:
            value = row[column]
            if pd.isna(value):
                entry[column] = None
            elif column in ("margin_pct", "discount_pct", "high_tier_share"):
                entry[column + "_percent"] = round(float(value) * 100, 1)
            else:
                entry[column] = round(float(value), 2)
        rows.append(entry)
    return rows


def _tool_get_ruled_out_checks(pack: dict) -> dict:
    quality = pack.get("data_quality") or {}
    seasonality = pack.get("seasonality") or {}
    episodes = pack.get("dip_episodes") or []
    tier = pack.get("tier_mix") or {}
    gaps = (quality.get("gaps") or {}).get("months_with_no_orders") or []
    outliers = quality.get("outlier_months") or []
    returns = quality.get("returns") or {}
    direction = {
        "premiumisation_detected": "favourable — spend moved INTO premium lines",
        "downgrade_detected": "unfavourable — spend moved OUT of premium lines into cheaper ones",
    }.get(tier.get("status"), "no material mix shift")
    return {
        "seasonal_repeat": {
            "confirmed": seasonality.get("status") == "confirmed",
            "reading": {
                "confirmed": "the recent dip repeats the same calendar window a year earlier",
                "not_present": "checked — the same months a year earlier were normal, so not seasonal",
                "no_recent_dip": "not applicable — there is no recent dip to explain",
            }.get(seasonality.get("status"), "could not be checked — under a year of history"),
            "dip_months": seasonality.get("dip_months") or [],
            "echo_months_last_year": seasonality.get("echo_months") or [],
        },
        "dip_episodes": [
            {"start": e.get("start_month"), "end": e.get("end_month"), "months": e.get("months"),
             "recovered": bool(e.get("recovered")), "still_ongoing": bool(e.get("is_ongoing"))}
            for e in episodes
        ],
        "months_with_no_orders": gaps,
        "one_off_bulk_months": [
            {"month": o.get("month"), "revenue": _money(o.get("revenue")),
             "times_a_typical_month": o.get("multiple_of_median_month")} for o in outliers
        ],
        "returns": {
            "credit_lines": returns.get("return_lines") or 0,
            "net_value": _money(returns.get("net_return_value")) if returns.get("return_lines") else 0,
            "note": "credit notes are already netted into every figure; a return is not leakage",
        },
        "product_mix_direction": direction,
        "account_manager_changed_mid_history": bool(pack.get("account_manager_changed")),
        "note": "A manager change, a region or a data gap correlates with everything and causes nothing by itself.",
    }


def _execute_chat_tool(df: pd.DataFrame, pack_for, report_for, name: str, tool_input: dict):
    """Run one tool. Returns (result, accounts the call was about)."""
    tool_input = dict(tool_input or {})
    report_for = report_for or (lambda _a: None)

    if name == "find_account":
        return find_accounts(df, tool_input.get("query", "")), []
    if name == "get_account_book":
        return _tool_get_account_book(df, pack_for, report_for), []
    if name == "compare_accounts":
        ids = list(dict.fromkeys(_resolve_account(df, a) for a in (tool_input.get("account_ids") or [])))
        if len(ids) < 2:
            raise ValueError("compare_accounts needs at least two distinct account_ids.")
        if len(ids) > MAX_COMPARE_ACCOUNTS:
            raise ValueError(f"compare_accounts takes at most {MAX_COMPARE_ACCOUNTS} accounts at once.")
        return build_comparison_digest(df, ids), ids

    account_id = _resolve_account(df, tool_input.pop("account_id", None))
    pack = pack_for(account_id)
    if name == "get_account_brief":
        return build_account_brief(pack, report_for(account_id)), [account_id]
    if name == "get_orders":
        return _tool_get_orders(df, account_id), [account_id]
    if name == "get_product_revenue":
        return _tool_get_product_revenue(df, account_id), [account_id]
    if name == "get_category_mix":
        return _tool_get_category_mix(df, account_id, pack), [account_id]
    if name == "get_monthly_table":
        return _tool_get_monthly_table(df, account_id), [account_id]
    if name == "get_kpis":
        return kpis_from_pack(pack), [account_id]
    if name == "get_ruled_out_checks":
        return _tool_get_ruled_out_checks(pack), [account_id]
    if name == "get_lifetime_value":
        block = pack.get("_lifetime_value") or {"status": "insufficient_history", "reason": "not computed"}
        # The chart series and the fit parameters are for the page; the chat
        # quotes figures, so they only cost tokens here.
        return {k: v for k, v in block.items() if k not in ("value_paths", "book_fit")}, [account_id]
    if name == "get_priced_options":
        impact = (report_for(account_id) or {}).get("financial_impact")
        options = price_options(pack, impact)
        return (options or {"priced_options": [], "note": "Nothing could be priced — either the account is healthy, the verdict deferred, or nothing has been lost yet."}), [account_id]
    # The five investigation drill-downs, unchanged apart from the account they run on.
    return _execute_tool(df, account_id, name, tool_input), [account_id]


def _cap_result(text: str, limit: int = MAX_TOOL_RESULT_CHARS) -> str:
    """Bound one tool result, marking the cut so the model knows it saw a
    prefix. Done here, before the result is stored or sent, so the logged
    result and the one the model read are the same string."""
    if len(text) <= limit:
        return text
    dropped = len(text) - limit
    return (
        text[:limit]
        + f"\n\n[truncated: {dropped:,} of {len(text):,} characters omitted. "
        "Narrow the question if you need the rest.]"
    )


# --- Questions that must go through a tool ---------------------------------------
#
# A brief's what_changed lines name the products that STOPPED, which is
# enough of a hint that a model answers "which products dropped?" from them
# and never fetches the ones that merely collapsed — measured live: two
# products named, the biggest three missed. A prompt rule alone did not hold
# across models, so the loop checks: if the question is plainly about one of
# these for one account and the model answers without having called a tool
# that has the answer, it is sent back once with the reason. Book-level
# questions ("which accounts lost products") are exempt — there the right
# tool is the book, and the model is trusted to pick it.

REQUIRED_TOOL_HINTS = [
    (re.compile(r"\b(products?|skus?|items?|product lines?)\b", re.I),
     ("get_product_changes", "get_product_revenue"),
     "A brief names only the lines that stopped entirely. Call get_product_changes (what moved) or "
     "get_product_revenue (what they buy) first, then answer."),
    (re.compile(r"season", re.I), ("get_ruled_out_checks",),
     "Call get_ruled_out_checks first — it holds the prior-year comparison — then answer."),
    (re.compile(r"\b(what (could|can|should) we do|worth|recover|win[- ]?back|options?|intervention)\b", re.I),
     ("get_priced_options",),
     "Call get_priced_options first — it holds every move that could be priced — then answer."),
]

_BOOK_LEVEL = re.compile(r"\b(accounts|which account|who|whose|every|all|book|portfolio|compare|versus|vs)\b", re.I)


def required_tools_for(question: str) -> list[tuple[tuple[str, ...], str]]:
    """(acceptable tools, instruction) pairs a single-account question plainly
    needs. Empty for a question across the book."""
    if _BOOK_LEVEL.search(question or ""):
        return []
    return [(tools, why) for pattern, tools, why in REQUIRED_TOOL_HINTS if pattern.search(question)]


# A lone tag-shaped string ("</something>") is serialiser junk, not text.
_TAG_JUNK = re.compile(r"^\s*</?[^\s<>]{0,60}>\s*$")


def clean_text(value) -> str:
    text = "" if value is None else str(value).strip()
    return "" if _TAG_JUNK.match(text) else text


# --- Next-question chips ------------------------------------------------------------
#
# The model closes a finished answer with 2-3 `<follow_up>…</follow_up>` lines
# (see SYSTEM_PROMPT). They are parsed out here — ONE parser, applied on every
# path a turn can end on — and never reach the screen as text: the streaming
# filter withholds them from text deltas, and the clean text is what gets
# stored and replayed. Ported from the ARYA MCP chat, where leaving them in
# the text was the original bug (raw tags mid-stream, and a stored turn
# replaying them to the model).

_FOLLOW_UP_RE = re.compile(
    r'<follow_up(?:\s+label\s*=\s*"([^"]*)")?\s*>(.*?)</follow_up\s*>',
    re.DOTALL | re.IGNORECASE,
)


def split_follow_ups(text: str) -> tuple[str, list[str]]:
    """Separate a model reply into (clean text, next-question chips).

    Every tag is stripped from the text even when more than the cap were
    offered — the cap bounds what the UI shows, not what leaks through. Chips
    are flat strings, deduplicated and whitespace-normalised: each is both
    what the chip shows AND the message tapping it sends.
    """
    chips: list[str] = []
    seen: set[str] = set()
    for match in _FOLLOW_UP_RE.finditer(text or ""):
        prompt = " ".join((match.group(2) or "").split())
        if not prompt or prompt.lower() in seen:
            continue
        seen.add(prompt.lower())
        chips.append(prompt)
    clean = _FOLLOW_UP_RE.sub("", text or "").rstrip()
    return clean, chips[:MAX_FOLLOW_UPS]


class FollowUpStreamFilter:
    """Forward text deltas with `<follow_up>` spans removed.

    Deltas are arbitrary slices — a tag can arrive as `<fol` + `low_up>` + …
    — so the filter holds back only what it cannot yet classify: from a `<`
    that could still be the start of a tag until either the closing tag
    arrives (span dropped) or the next character proves it is ordinary text
    (a `<` in "a < b") and it is released. Everything else is forwarded at
    once, so the stream stays token-by-token for the whole answer and simply
    ends where the chips begin.
    """

    OPEN = "<follow_up"
    CLOSE = "</follow_up>"

    def __init__(self, emit: Callable[[str], None]):
        self._emit = emit
        self._buf = ""

    def push(self, delta: str) -> None:
        self._buf += delta
        self._drain(final=False)

    def flush(self) -> None:
        """End of one LLM call: release anything held, drop an unterminated tag."""
        self._drain(final=True)

    def _drain(self, final: bool) -> None:
        out: list[str] = []
        while self._buf:
            i = self._buf.find("<")
            if i == -1:
                out.append(self._buf)
                self._buf = ""
                break
            out.append(self._buf[:i])
            self._buf = self._buf[i:]
            head = self._buf[: len(self.OPEN)].lower()
            if head == self.OPEN:
                j = self._buf.lower().find(self.CLOSE)
                if j == -1:
                    if final:
                        self._buf = ""  # a tag the model never closed: never show it
                    break  # hold until the closing tag arrives
                self._buf = self._buf[j + len(self.CLOSE):]
                continue
            if not final and len(head) < len(self.OPEN) and self.OPEN.startswith(head):
                break  # could still become a tag — wait for more
            out.append("<")  # an ordinary '<' — release it
            self._buf = self._buf[1:]
        text = "".join(out)
        if text:
            self._emit(text)


# --- The figure check -----------------------------------------------------------------
#
# The rule is "every number you state must appear verbatim in the opening
# message or a tool result". Rather than ask the model to list what it
# quoted, the finished answer is checked: every figure in it is looked for in
# the sources this turn could legitimately have read. Dates are not figures;
# a small count without a unit ("3 products", "6 months") is the model
# describing a list, not quoting a number, and is left alone. A stated figure
# matches a source value when the source rounds to it at the precision the
# answer used, so "Rs 88,810" matches 88810.4 and "down 12 points" matches
# -12.3; a source ratio also matches its percentage form, so 0.3215 in a tool
# result supports "32.2%" — the tolerance is what "verbatim" means for a
# person, not for a string compare.

_DATE_LIKE = re.compile(r"\b\d{4}-\d{2}(?:-\d{2})?\b|\b(?:19|20)\d{2}\b")
_NUMBER = re.compile(r"(?<![\w.,])(\d{1,3}(?:,\d{2,3})+|\d+)(\.\d+)?(?!\w)")
_UNIT_BEFORE = re.compile(r"(?:rs\.?|₹|inr|\$|usd)\s*$", re.I)
_UNIT_AFTER = re.compile(r"^\s*(?:%|percent|points?|pts|pp\b|x\b|×|times)", re.I)
_SMALL_COUNT_LIMIT = 13


def _numbers_in_text(text: str) -> list[tuple[str, float, int, bool]]:
    """(as written, value, decimals, has_unit) for every figure in `text`."""
    found = []
    stripped = _DATE_LIKE.sub(" ", text or "")
    for match in _NUMBER.finditer(stripped):
        whole, fraction = match.group(1), match.group(2) or ""
        value = float(whole.replace(",", "") + fraction)
        decimals = len(fraction) - 1 if fraction else 0
        has_unit = bool(_UNIT_BEFORE.search(stripped[max(0, match.start() - 6):match.start()])
                        or _UNIT_AFTER.match(stripped[match.end():match.end() + 10]))
        found.append((match.group(0), value, decimals, has_unit))
    return found


def _source_values(sources: list) -> list[float]:
    """Every number a turn could have quoted, from JSON-able sources and text."""
    values: set[float] = set()
    for source in sources:
        if source is None:
            continue
        text = source if isinstance(source, str) else compact_json(source)
        for _, value, _, _ in _numbers_in_text(text):
            values.add(value)
            if 0 < abs(value) <= 1:
                values.add(round(value * 100, 6))  # a ratio, quoted as a percentage
    return sorted(values)


def unsourced_figures(answer: str, sources: list) -> list[str]:
    """Figures in `answer` that appear in none of `sources`, as written."""
    available = _source_values(sources)
    missing: list[str] = []
    for written, value, decimals, has_unit in _numbers_in_text(answer):
        if not has_unit and decimals == 0 and abs(value) < _SMALL_COUNT_LIMIT:
            continue
        tolerance = 0.5 * 10 ** (-decimals) + 1e-9
        if any(abs(candidate - value) <= tolerance for candidate in available):
            continue
        if written not in missing:
            missing.append(written)
    return missing


# --- Events -------------------------------------------------------------------------------

@dataclass
class AgentEvent:
    """A progress event emitted mid-turn, for the screen.

    A turn can spend seconds in tool calls before producing text, so without
    these the manager stares at a spinner. `text` deltas arrive as the model
    writes; `tool_start` / `tool_end` bracket each call; `retry` says the
    draft just streamed was refused and a fresh one is coming; `follow_ups`
    carries the next-question chips once, after the text, on a finished turn.
    """

    type: str  # "text" | "tool_start" | "tool_end" | "retry" | "follow_ups"
    text: str = ""
    tool_id: str = ""
    tool_name: str = ""
    arguments: dict = field(default_factory=dict)
    ok: bool = True
    result_chars: int = 0
    result_preview: str = ""
    reason: str = ""
    follow_ups: list[str] = field(default_factory=list)


EventCallback = Callable[[AgentEvent], None]


# --- Messages and the loop ---------------------------------------------------------

def _alternating(turns: list[dict]) -> list[dict]:
    """Prior turns as user/assistant messages, text only, strictly alternating.

    A user turn whose answer never arrived (a failed call) is dropped rather
    than sent, so the history never carries an unanswered question the model
    might try to answer instead of the current one.
    """
    messages: list[dict] = []
    pending_user = None
    for turn in turns:
        if turn.get("role") == "user":
            pending_user = turn
        elif turn.get("role") == "assistant" and pending_user is not None:
            messages.append({"role": "user", "content": pending_user["text"]})
            messages.append({"role": "assistant", "content": turn["text"]})
            pending_user = None
    return messages


def build_messages(index: dict, focus_brief: dict | None, summary: str | None,
                   turns: list[dict], question: str) -> list[dict]:
    """The prompt for one question: the account index, the account in focus
    (if any) and the compacted summary first, then the recent exchange
    verbatim, then the question."""
    opening = "ACCOUNT INDEX:\n" + compact_json(index)
    if focus_brief:
        opening += "\n\nACCOUNT IN FOCUS:\n" + compact_json(focus_brief)
    else:
        opening += "\n\nACCOUNT IN FOCUS: none yet — resolve one from the question, or ask."
    if summary:
        opening += "\n\nEARLIER IN THIS CHAT (compacted):\n" + summary
    messages = [
        {"role": "user", "content": opening},
        {"role": "assistant", "content": "Understood. I will answer only from this context and the tools. What would you like to know?"},
    ]
    messages.extend(_alternating(turns[-KEEP_LAST_TURNS:]))
    messages.append({"role": "user", "content": question})
    return messages


def _text_of(response) -> str:
    return "\n".join(b.text for b in response.content if getattr(b, "type", None) == "text" and b.text).strip()


def _usage_of(response) -> dict:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    return {key: int(getattr(usage, key, 0) or 0) for key in ("input_tokens", "output_tokens")}


def _complete(client, model: str, messages: list, tools: list, on_text: Optional[Callable[[str], None]]):
    """One LLM call, streamed when the caller wants deltas and the client can
    stream. A client with only `create` still works — the whole text arrives
    as one delta after the call — so the loop never depends on streaming for
    correctness, only for feel."""
    kwargs = dict(model=model, max_tokens=OUTPUT_TOKENS, system=SYSTEM_PROMPT, tools=tools, messages=messages)
    stream = getattr(client.messages, "stream", None) if on_text is not None else None
    if stream is None:
        response = client.messages.create(**kwargs)
        if on_text is not None:
            text = _text_of(response)
            if text:
                on_text(text)
        return response
    with stream(**kwargs) as live:
        for delta in live.text_stream:
            if delta:
                on_text(delta)
        return live.get_final_message()


def ask(
    client,
    df: pd.DataFrame,
    question: str,
    *,
    pack_for: Callable[[str], dict],
    report_for: Callable[[str], dict | None] | None = None,
    focus_account: str | None = None,
    turns: list[dict] | None = None,
    summary: str | None = None,
    model: str = DEFAULT_MODEL,
    max_iterations: int = MAX_ITERATIONS,
    on_event: Optional[EventCallback] = None,
) -> dict:
    """Answer one question about the book, running the loop to completion.

    `pack_for(account_id)` and `report_for(account_id)` are how the chat
    reaches what the pipeline has already computed (the caller usually hands
    in cached versions). `focus_account` is the account the chat is currently
    about, if any; the returned `focus_account` is what it should be after
    this answer — the last account a tool was called on, else unchanged.

    Returns the answer (clean prose), the next-question chips, the tools the
    loop actually ran (with a log the screen can open), the accounts those
    tools touched, the figures the answer states that no source supports,
    summed token usage, and whether the step ceiling cut it short. `on_event`
    receives progress as it happens; omit it and the turn runs silently.

    Raises ChatError if the model never produces any answer text.
    """
    report_for = report_for or (lambda _a: None)
    index = build_account_index(df, report_for)
    focus_brief = None
    if focus_account and focus_account in set(df["account_id"].unique()):
        focus_brief = build_account_brief(pack_for(focus_account), report_for(focus_account))
    else:
        focus_account = None
    messages = build_messages(index, focus_brief, summary, turns or [], question)

    tool_log: list[dict] = []
    touched: list[str] = []
    # Only a tool about ONE account moves the focus. "Compare it with Harbor"
    # touches two accounts but is still about the one already in focus.
    focus_moves: list[str] = []
    # What the answer may legitimately quote: the opening context, this turn's
    # tool results (the exact strings the model read), and the earlier exchange.
    sources: list = [index, focus_brief, summary, question, *[t.get("text") for t in (turns or [])]]
    usage: dict[str, int] = {}
    must_call = required_tools_for(question)
    pushed_back = False
    nudged = False
    last_text = ""

    def emit(event: AgentEvent) -> None:
        if on_event is not None:
            on_event(event)

    stream_filter = FollowUpStreamFilter(lambda text: emit(AgentEvent(type="text", text=text)))
    on_text = stream_filter.push if on_event is not None else None

    def finished(answer_text: str, iteration: int, truncated: bool) -> dict:
        answer, chips = split_follow_ups(answer_text)
        emit(AgentEvent(type="follow_ups", follow_ups=chips))
        return {
            "answer": answer,
            "follow_ups": chips,
            "tools_used": [entry["name"] for entry in tool_log],
            "tool_log": tool_log,
            "accounts_touched": list(dict.fromkeys(touched)),
            "focus_account": focus_moves[-1] if focus_moves else focus_account,
            "unsourced_figures": unsourced_figures(answer, sources),
            "usage": usage,
            "iterations": iteration,
            "truncated": truncated,
        }

    for iteration in range(1, max_iterations + 1):
        response = _complete(client, model, messages, CHAT_TOOLS, on_text)
        stream_filter.flush()  # tags never span LLM calls
        for key, value in _usage_of(response).items():
            usage[key] = usage.get(key, 0) + value
        text = _text_of(response)
        if text:
            last_text = text
        tool_use_blocks = [b for b in response.content if b.type == "tool_use"]

        if not tool_use_blocks:
            if not text:
                # Out of room with nothing visible (a reasoning model that
                # spent the whole budget thinking). One retry, told to be brief.
                if response.stop_reason in ("length", "max_tokens") and not nudged:
                    nudged = True
                    messages.append({"role": "user", "content": (
                        "You ran out of room. Answer now, briefly — at most 60 words — from what "
                        "you already have."
                    )})
                    continue
                raise ChatError(
                    f"Model stopped (stop_reason={response.stop_reason!r}) without answering."
                )
            called = {entry["name"] for entry in tool_log}
            skipped = [(tools, why) for tools, why in must_call if not called.intersection(tools)]
            # The hints only bind once the question is about ONE account —
            # either the chat has one in focus or a tool already resolved one.
            if skipped and not pushed_back and (focus_account or touched):
                pushed_back = True
                reason = " ".join(why for _, why in skipped)
                messages.append({"role": "assistant", "content": response.content})
                messages.append({"role": "user", "content": "Not accepted yet. " + reason})
                emit(AgentEvent(type="retry", reason=reason))
                continue
            return finished(text, iteration, truncated=False)

        messages.append({"role": "assistant", "content": response.content})
        results = []
        for block in tool_use_blocks:
            arguments = dict(block.input) if block.input else {}
            emit(AgentEvent(type="tool_start", tool_id=block.id, tool_name=block.name, arguments=arguments))
            try:
                result, accounts = _execute_chat_tool(df, pack_for, report_for, block.name, block.input)
                content = _cap_result(compact_json(result))
                entry = {"name": block.name, "input": arguments, "ok": True, "accounts": accounts,
                         "result_chars": len(content), "result_preview": content[:RESULT_PREVIEW_CHARS]}
                touched.extend(accounts)
                if len(accounts) == 1:
                    focus_moves.append(accounts[0])
                sources.append(content)
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": content})
            except Exception as e:  # a bad tool call is reported to the model, never raised
                entry = {"name": block.name, "input": arguments, "ok": False, "error": str(e),
                         "accounts": [], "result_chars": 0, "result_preview": ""}
                results.append({"type": "tool_result", "tool_use_id": block.id,
                                "content": f"Error: {e}", "is_error": True})
            tool_log.append(entry)
            emit(AgentEvent(type="tool_end", tool_id=block.id, tool_name=block.name, ok=entry["ok"],
                            result_chars=entry["result_chars"], result_preview=entry["result_preview"]))
        # Every result for one assistant turn goes back in ONE message.
        messages.append({"role": "user", "content": results})

    # Hit the step ceiling with the model still calling tools. What it had
    # written stands, marked as cut short, rather than erroring a turn that
    # may already hold useful work.
    if not split_follow_ups(last_text)[0]:
        raise ChatError(f"Exceeded max_iterations={max_iterations} without an answer.")
    return finished(last_text, max_iterations, truncated=True)


def compact_conversation(client, turns: list[dict], summary: str | None = None,
                         model: str = DEFAULT_MODEL) -> str:
    """One call that folds the given turns (and any existing summary) into a
    new summary. The caller decides which turns to fold — normally everything
    except the last KEEP_LAST_TURNS."""
    transcript = "\n".join(
        f"{'Manager' if t.get('role') == 'user' else 'Analyst'}: {t.get('text', '')}" for t in turns
    )
    body = ""
    if summary:
        body += "EXISTING SUMMARY OF EVEN EARLIER TURNS:\n" + summary + "\n\n"
    body += "TURNS TO COMPACT:\n" + transcript
    response = client.messages.create(
        model=model, max_tokens=OUTPUT_TOKENS, system=COMPACT_PROMPT,
        tools=[SUBMIT_SUMMARY_TOOL], messages=[{"role": "user", "content": body}],
    )
    block = next((b for b in response.content if b.type == "tool_use" and b.name == "submit_summary"), None)
    if block is None or "summary" not in block.input:
        raise ChatError(
            f"Model stopped (stop_reason={response.stop_reason!r}) without calling submit_summary."
        )
    return str(block.input["summary"]).strip()
