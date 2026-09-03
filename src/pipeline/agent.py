"""
Stage 4: Investigation & Attribution Agent — the single LLM touchpoint in
the pipeline (see plan.md "Design principle: LLM only where judgement is
genuinely required"). Everything upstream (stages 1-3) is deterministic
code; this stage reasons over the evidence pack those stages produced,
optionally drills down via read-only tools back into the same deterministic
functions, and emits one structured verdict. It never computes a number —
every figure it can cite comes from the evidence pack or a drill-down tool
result, never from its own arithmetic.

The client is injected (not constructed here) so tests can pass a fake
that mimics the same `.messages.create(...)` shape as `anthropic.Anthropic`
without hitting the real API — see tests/test_agent.py.
"""

from __future__ import annotations

import json

import pandas as pd

from .detect import _monthly_category_series, product_changes
from .changepoint import full_month_index

DEFAULT_MODEL = "claude-opus-5"
MAX_ITERATIONS = 6

SYSTEM_PROMPT = """You are the investigation and attribution agent in a revenue-leakage detection \
system for B2B retail accounts. You are the ONLY reasoning step in this pipeline — every stage \
before you (data cleaning, baseline computation, change-point detection) is deterministic code, \
and every number you see was computed there, not by you. You must never invent, adjust, or compute \
a number yourself; only cite facts that appear in the evidence pack or in a tool result.

Your job: given one account's evidence pack, decide whether it shows healthy behaviour, temporary \
variation, or structural revenue leakage — and which categories are responsible if so.

TEMPORARY VS STRUCTURAL — apply these criteria explicitly, not vibes:
- Structural signals: a sustained (multi-period) category decline past a change-point with no \
seasonal precedent; replacement by lower-value substitutes; shrinking order sizes in the affected \
category; a change-point that has not recovered (recovered: false) and has persisted several months.
- Temporary signals: seasonal precedent in prior history for the same calendar period; a single \
anomalous period; a change that has already recovered (recovered: true); an isolated one-off event \
(e.g. one unusually large historical order) rather than a sustained shift.
- The evidence pack's `seasonal_precedent` field is deliberately conservative: it only ever says \
"confirmed" when at least 2 prior occurrences of the same calendar month back it up, so it will \
often say "insufficient_history" rather than guess when history is short — that is not the same as \
"no seasonality." If seasonality could plausibly explain a decline and the field says \
insufficient_history or not_present, call get_category_seasonal_breakdown yourself and judge the raw \
calendar-month history directly — with under ~2 years of data you may see a plausible seasonal \
pattern the deterministic check couldn't yet confirm on its own stricter standard.
- CATEGORY MATERIALITY: a change-point in a category that was only a small share of this account's \
baseline revenue (see category_mix.baseline_share) is much weaker evidence of meaningful leakage than \
one in a major category, even if the percentage decline looks large — a low-revenue category is also \
where random noise most easily crosses a decline threshold by chance. Do not build a confident \
structural verdict primarily around a single low-share category's change-point; either look for \
corroborating evidence (does total revenue or a high-value category also move, does the pattern \
repeat across categories), or reflect the weak materiality in lower confidence.
- ORDER BEHAVIOUR IS ITS OWN SIGNAL, not just a modifier of category-level findings: check \
order_behavior (basket_width_pct_change, order_frequency_pct_change, aov_pct_change) directly. A \
sustained drop in basket width while frequency holds roughly steady is itself structural leakage — \
smaller baskets, not fewer orders — even when no single category shows a strong change-point. Don't \
only look at category_changes and conclude there's nothing there; a real shift can show up in how \
orders are shaped rather than in any one category's revenue.

DEFER WHEN THE EVIDENCE DOESN'T SUPPORT A CLEAN ANSWER. This is the most important instruction in \
this prompt. If history is too short (see data_sufficiency), the shift could plausibly be seasonal \
but you can't confirm it, or the signals conflict, set defer=true, use low confidence, and name in \
data_needed_if_deferring exactly what data would resolve it. A deferred, low-confidence, well- \
reasoned answer is scored HIGHER than a forced confident one — do not manufacture a verdict to \
sound decisive.

You may call the drill-down tools as many times as you need before answering. When you have enough \
evidence, call submit_verdict exactly once with your final structured answer. Every entry in \
cited_facts must reference a specific fact from the evidence pack or a tool result (e.g. \
"Industrial Equipment: change-point 2025-05, 100% decline, recovered=false, sustained 7 months") — \
not a vague restatement."""


def _tool_get_category_monthly_series(df: pd.DataFrame, account_id: str, category: str) -> dict:
    acc_df = df[df["account_id"] == account_id]
    months = full_month_index(acc_df)
    series = _monthly_category_series(acc_df, category, months)
    return {str(m): round(float(v), 2) for m, v in series.items()}


def _tool_get_category_seasonal_breakdown(df: pd.DataFrame, account_id: str, category: str) -> dict:
    acc_df = df[df["account_id"] == account_id]
    months = full_month_index(acc_df)
    series = _monthly_category_series(acc_df, category, months)
    by_calendar_month: dict[int, list] = {}
    for month, value in series.items():
        by_calendar_month.setdefault(month.month, []).append({"period": str(month), "revenue": round(float(value), 2)})
    return {str(k): v for k, v in sorted(by_calendar_month.items())}


def _tool_get_product_changes(df: pd.DataFrame, account_id: str, category: str | None = None) -> list:
    acc_df = df[df["account_id"] == account_id]
    changes = product_changes(acc_df, top_n=50)
    if category:
        changes = [c for c in changes if c["category"] == category]
    return changes


DRILLDOWN_TOOLS = [
    {
        "name": "get_category_monthly_series",
        "description": "Raw monthly revenue for one category for this account, full history, no smoothing — for inspecting a change-point or trend in finer detail than the evidence pack's summary.",
        "input_schema": {
            "type": "object",
            "properties": {"category": {"type": "string"}},
            "required": ["category"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "name": "get_category_seasonal_breakdown",
        "description": "This category's monthly revenue grouped by calendar month across all years in history — use this to judge seasonality yourself when the evidence pack's seasonal_precedent field says insufficient_history or not_present but seasonality still seems plausible; that field requires 2+ prior years of the same calendar month before it will confirm anything.",
        "input_schema": {
            "type": "object",
            "properties": {"category": {"type": "string"}},
            "required": ["category"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "name": "get_product_changes",
        "description": "Product-level revenue changes (disappeared/declined/new) for this account, optionally filtered to one category — for locating which specific products, not just which category, are responsible.",
        "input_schema": {
            "type": "object",
            "properties": {"category": {"type": ["string", "null"]}},
            "required": ["category"],
            "additionalProperties": False,
        },
        "strict": True,
    },
]

SUBMIT_VERDICT_TOOL = {
    "name": "submit_verdict",
    "description": "Submit your final, structured investigation result. Call this exactly once, when you have enough evidence to answer (including a deferred answer).",
    "input_schema": {
        "type": "object",
        "properties": {
            "verdict": {
                "type": "string",
                "enum": ["healthy", "leakage_detected", "insufficient_data"],
            },
            "temporary_or_structural": {
                "type": "string",
                "enum": ["temporary", "structural", "not_applicable"],
            },
            "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
            "defer": {"type": "boolean"},
            "attributed_categories": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Categories responsible for the leakage, empty if healthy or deferring.",
            },
            "cited_facts": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Specific facts from the evidence pack or tool results that this verdict rests on.",
            },
            "narrative": {
                "type": "string",
                "description": "A short explanation a business stakeholder could act on directly.",
            },
            "recommended_actions": {"type": "array", "items": {"type": "string"}},
            "data_needed_if_deferring": {
                "type": "array",
                "items": {"type": "string"},
                "description": "What additional data would resolve the ambiguity. Empty if not deferring.",
            },
        },
        "required": [
            "verdict", "temporary_or_structural", "confidence", "defer",
            "attributed_categories", "cited_facts", "narrative",
            "recommended_actions", "data_needed_if_deferring",
        ],
        "additionalProperties": False,
    },
    "strict": True,
}


class AgentError(Exception):
    """Raised when the agent loop fails to reach a submit_verdict call
    (max iterations exhausted, or the model stopped for an unexpected
    reason) — surfaced as a clear error rather than a silent bad verdict."""


def _execute_tool(df: pd.DataFrame, account_id: str, name: str, tool_input: dict):
    if name == "get_category_monthly_series":
        return _tool_get_category_monthly_series(df, account_id, tool_input["category"])
    if name == "get_category_seasonal_breakdown":
        return _tool_get_category_seasonal_breakdown(df, account_id, tool_input["category"])
    if name == "get_product_changes":
        return _tool_get_product_changes(df, account_id, tool_input.get("category"))
    raise AgentError(f"Unknown tool requested by model: {name}")


def investigate(
    client,
    df: pd.DataFrame,
    account_id: str,
    evidence_pack: dict,
    model: str = DEFAULT_MODEL,
    max_iterations: int = MAX_ITERATIONS,
) -> dict:
    """Run the Stage 4 agent loop for one account. Returns the parsed
    submit_verdict input dict. Raises AgentError if the model never
    submits a verdict within max_iterations."""
    tools = DRILLDOWN_TOOLS + [SUBMIT_VERDICT_TOOL]
    messages = [{"role": "user", "content": json.dumps(evidence_pack)}]

    for _ in range(max_iterations):
        response = client.messages.create(
            model=model,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=tools,
            messages=messages,
        )

        tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
        submit_block = next((b for b in tool_use_blocks if b.name == "submit_verdict"), None)
        if submit_block is not None:
            return submit_block.input

        if not tool_use_blocks:
            raise AgentError(
                f"Model stopped (stop_reason={response.stop_reason!r}) without calling submit_verdict."
            )

        messages.append({"role": "assistant", "content": response.content})
        tool_results = []
        for block in tool_use_blocks:
            try:
                result = _execute_tool(df, account_id, block.name, block.input)
                tool_results.append({
                    "type": "tool_result", "tool_use_id": block.id, "content": json.dumps(result),
                })
            except Exception as e:
                tool_results.append({
                    "type": "tool_result", "tool_use_id": block.id,
                    "content": f"Error: {e}", "is_error": True,
                })
        messages.append({"role": "user", "content": tool_results})

    raise AgentError(f"Exceeded max_iterations={max_iterations} without a submit_verdict call.")
