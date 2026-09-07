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
import re

import pandas as pd

from .detect import _monthly_category_series, product_changes
from .changepoint import full_month_index

DEFAULT_MODEL = "claude-sonnet-5"  # the one model the app and the scorer run on (Sept 7)
MAX_ITERATIONS = 6

# Output ceiling per model call. Claude 5-family models think before they
# answer and that thinking is billed against this same ceiling — measured
# on ACC-106, 1,647 of 2,557 output tokens were thinking — so 4,096 cut the
# submit call off part-way on a long think. The Groq shim clamps this to
# its own ceiling (see groq_client.GROQ_MAX_OUTPUT_TOKENS); nothing here
# is provider-aware.
MAX_OUTPUT_TOKENS = 16000

# Sent when a reply hits the output ceiling before the submit call.
TRUNCATION_NUDGE = (
    "Your reply was cut off by the output limit before you called submit_verdict. Do not write "
    "any more analysis. Call submit_verdict now with your answer, keeping every field within "
    "the OUTPUT FORMAT limits."
)

SYSTEM_PROMPT = """You are the investigation agent in a revenue-leakage detection system for B2B \
retail accounts. Everything before you was deterministic code; every number you see was computed \
there. Never invent, adjust or compute a number. Cite only facts that appear in the evidence pack \
or a tool result.

Your job: read one account's evidence pack and decide whether it shows healthy behaviour, a \
temporary dip, or structural revenue leakage — and where the value is leaving if it is. Your \
answer goes straight onto a sales manager's screen. It must be short, specific and actionable.

MOST ACCOUNTS THAT LOOK LIKE LEAKS ARE NOT. A scary recent number is usually seasonal, already \
recovered, inflated by an earlier bulk month, a missing month of data, or normal drift. Flagging \
everything is the most common failure. Equally, a real leak can look healthy at the topline. Your \
value is the discrimination, not the alarm.

CHECK EVERY DIMENSION, NOT JUST REVENUE. `analysis_dimensions` says which the file could support; \
a dimension marked false was not measured — say so, never imply it was fine.
- revenue_decline: material_decline / mild_drift / stable / growth. mild_drift is the normal band, \
not a leak.
- margin: erosion_detected at flat revenue is a leak revenue-only reasoning cannot see. Always \
check it before calling an account healthy.
- discount: creep_detected means the same goods sold at a steadily deeper discount; volume and \
mix can look untouched while the money leaves through the price.
- tier_mix: downgrade_detected at flat revenue is the classic hidden leak. premiumisation_detected \
is the opposite and is GOOD NEWS.
- order_pattern: fragmentation_detected (more, smaller orders) is an early multi-sourcing signal \
on its own, even with flat revenue.
- category_changes: `defected: true` means a category stopped and stayed stopped — name it.

DIRECTION IS NOT MAGNITUDE. Rising margin, rising high-tier share and rising revenue are good. An \
account buying fewer units but trading UP into premium lines is healthy; flagging it is a serious \
error. Read which way a mix moved before judging it.

RULE OUT THE INNOCENT EXPLANATIONS FIRST:
- seasonality.status == "confirmed": the same dip happened in the same calendar window a year \
earlier. Strong evidence it is seasonal. Name the echo months.
- dip_episodes: a dip that `recovered` is a closed incident. Only an `is_ongoing` episode or an \
unrecovered change-point is live.
- data_quality.gaps.months_with_no_orders: a hole in the data, not zero trading. It drags trailing \
averages down by itself.
- data_quality.outlier_months: a one-off stock-up inflates its window; the months after it are a \
return to normal, not a decline.
- data_quality.returns: credit notes are already netted into every figure. Not leakage.
- account_manager_changed: a correlation, never a cause. Context at most.

ATTRIBUTION HONESTY. Name a category only when it actually explains the loss. A broad-based \
decline with no category defected and none dominating is whole-account disengagement: say so and \
leave attributed_categories EMPTY. A margin or discount leak goes in leak_dimensions with the \
category list empty unless one category is genuinely responsible.

MATERIALITY. A change-point in a category with a small baseline_share is weak evidence however \
large the percentage — that is where noise crosses thresholds. Do not build a confident verdict on \
one low-share category.

SEASONALITY WITH SHORT HISTORY. `seasonal_precedent` needs 2+ prior years and will usually say \
insufficient_history — that is not "no seasonality". `prior_year_echo` and the account-level \
`seasonality` field are what two years can support. If seasonality is plausible and unresolved, \
call get_category_seasonal_breakdown or get_tier_monthly_series and judge the calendar yourself.

DEFER WHEN THE EVIDENCE DOES NOT SUPPORT A CLEAN ANSWER. This is the most important rule. If \
history is too short (data_sufficiency flags such as history_too_short_for_baseline or \
cannot_check_prior_year_seasonality), the shift could be seasonal but cannot be confirmed, or the \
signals conflict: set defer=true, confidence low, and name in data_needed_if_deferring exactly what \
would resolve it. A few months of history cannot be classified temporary vs structural, and \
guessing is the failure. A deferred, well-reasoned answer scores higher than a forced one.

A SECOND OPINION, NOT A VERDICT. `model_opinion` (when `available`) is a statistical classifier's \
read of the same pack: P(leakage), P(healthy), P(defer) and the facts they rest on. It cannot read \
context and is sometimes wrong. Treat it as one more witness.
- It is never, by itself, a reason to flag. A leakage verdict still needs a named dimension and \
cited facts.
- If you disagree with its leaning, model_opinion_response must name the fact that overrides it \
(a prior-year echo, a favourable mix direction, a recovered dip), and confidence is at most medium \
unless that fact is unambiguous.
- If it agrees and is `decisive`, that corroborates you. If it is not decisive, the numbers alone \
are ambiguous — a reason for caution.
When `available` is false, ignore it and leave model_opinion_response empty.

OUTPUT FORMAT. A sales manager reads narrative and cited_facts on screen as plain text, so:
- Plain business language throughout. Never use internal status names (erosion_detected, \
creep_detected, material_decline, downgrade_detected, fragmentation_detected, mild_drift, \
defected) or the word "change-point" — not even in cited_facts. Say what the customer is doing: \
"spending a third less", "same goods at a deeper discount", "stopped buying Diagnostic \
Equipment", "no category has stopped", "drifting into cheaper lines", "splitting spend across \
more, smaller orders". Never quote pack keys or key:value pairs ("defected: false", "status: \
stable") — translate them.
- NUMBERS, WRITTEN FOR A MANAGER. Every figure must come from the pack or a tool result; you may \
only change how it is written, never its value. Rates arrive as fractions (0.3215): write 32.2%. \
Rupees: ₹88,810 with separators, no decimals. Months arrive as 2026-04: write April 2026. A move \
in a rate: "down 12 points, from 32.2% to 19.9%" — never "-12.26pp". A move in an amount: "down \
68.9%, from ₹88,810 to ₹27,661 a month". Never write a p-value, a raw ratio, a slope, a field \
name, or "significant"; say "well outside this account's normal swing" if the point matters. At \
most two figures per sentence, and only where a figure carries the point.
- narrative: HARD LIMIT 60 words and 3 sentences — count them. Sentence 1: the call and where \
value is leaving (or why the account is fine, or why you cannot call it). Sentence 2: the one or \
two figures that prove it. Sentence 3: temporary or structural, and why.
- cited_facts: 3 to 5 complete sentences, each under 25 words, each carrying one or two figures \
and a month or period, e.g. "Margin fell from 32.2% to 19.9% after November 2025 while revenue \
held within 2.5% of baseline." Written to be read aloud, not scanned.
- recommended_actions: 1 to 3 items, each one concrete action under 20 words, starting with a \
verb. Empty when the account is healthy. A separate decision step will plan the intervention, so \
keep these to the immediate next step.
- data_needed_if_deferring: 1 to 3 concrete items. Empty unless deferring.
- model_opinion_response: one sentence, under 25 words. Empty if the opinion was unavailable.

Call the drill-down tools as often as you need, then call submit_verdict exactly once."""


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


def _tool_get_monthly_economics(df: pd.DataFrame, account_id: str) -> dict:
    """Month-by-month revenue, margin, margin rate and average discount side
    by side — the view that separates a volume story from a price story."""
    acc_df = df[df["account_id"] == account_id].copy()
    months = full_month_index(acc_df)
    acc_df["month"] = acc_df["date"].dt.to_period("M")

    revenue = acc_df.groupby("month")["revenue"].sum().reindex(months, fill_value=0.0)
    has_margin = "margin" in acc_df.columns and acc_df["margin"].notna().any()
    margin = (
        acc_df.groupby("month")["margin"].sum().reindex(months, fill_value=0.0)
        if has_margin else None
    )

    discount = None
    if "discount_pct" in acc_df.columns and acc_df["discount_pct"].notna().any():
        sales = acc_df[acc_df.get("is_return", 0) != 1]
        weight = (
            (sales["list_price"] * sales["quantity"]).abs()
            if "list_price" in sales.columns and sales["list_price"].notna().any()
            else sales["revenue"].abs()
        )
        weighted = sales.assign(_w=weight, _wd=sales["discount_pct"] * weight)
        grouped = weighted.groupby("month")[["_w", "_wd"]].sum()
        discount = (grouped["_wd"] / grouped["_w"]).reindex(months)

    out = {}
    for month in months:
        row = {"revenue": round(float(revenue.loc[month]), 2)}
        if margin is not None:
            month_margin = float(margin.loc[month])
            row["margin"] = round(month_margin, 2)
            row["margin_pct"] = (
                round(month_margin / float(revenue.loc[month]), 4)
                if float(revenue.loc[month]) else None
            )
        if discount is not None:
            value = discount.loc[month]
            row["avg_discount_pct"] = None if pd.isna(value) else round(float(value), 4)
        out[str(month)] = row
    return out


def _tool_get_tier_monthly_series(df: pd.DataFrame, account_id: str) -> dict:
    """Monthly revenue split by product value tier, plus the High-tier share
    of each month — for judging a mix shift's direction directly."""
    acc_df = df[df["account_id"] == account_id].copy()
    if "tier" not in acc_df.columns or not acc_df["tier"].notna().any():
        return {"error": "This dataset has no product value-tier column; tier mix cannot be analysed."}

    months = full_month_index(acc_df)
    acc_df["month"] = acc_df["date"].dt.to_period("M")
    pivot = (
        acc_df.pivot_table(index="month", columns="tier", values="revenue", aggfunc="sum")
        .reindex(months)
        .fillna(0.0)
    )
    out = {}
    for month in months:
        row = {tier: round(float(pivot.loc[month, tier]), 2) for tier in pivot.columns}
        total = sum(row.values())
        row["high_tier_share"] = round(row.get("High", 0.0) / total, 4) if total else None
        out[str(month)] = row
    return out


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
    {
        "name": "get_monthly_economics",
        "description": "Month-by-month revenue, margin, margin rate and average discount for this account, side by side. Use this to separate a volume story from a price story — e.g. to see whether flat revenue is being held up while margin rate falls, or to trace when a discount started creeping.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "name": "get_tier_monthly_series",
        "description": "Monthly revenue split by product value tier (High/Mid/Low) with each month's High-tier share. Use this to judge the DIRECTION of a mix shift yourself — value moving out of High tier is a downgrade, value moving into it is premiumisation.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
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
            "leak_dimensions": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": ["revenue", "margin", "discount", "category_mix", "tier_mix",
                             "order_pattern"],
                },
                "description": "Which dimension(s) the value is actually leaking through. Empty if healthy or deferring. A margin or discount leak at flat revenue must say so here.",
            },
            "attributed_categories": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Categories responsible for the leakage. Empty if healthy, deferring, or if the decline is broad-based with no single category responsible — do not name a scapegoat category for a diffuse decline.",
            },
            "cited_facts": {
                "type": "array",
                "items": {"type": "string"},
                "description": "3-5 complete plain-English sentences the verdict rests on, each under 25 words with one or two figures written for a manager (₹88,810, 32.2%, April 2026) and no p-values, field names or status names.",
            },
            "narrative": {
                "type": "string",
                "description": "At most 3 sentences, under 60 words: the call and where value is leaving; the figure(s) that prove it; temporary or structural and why. Plain business language.",
            },
            "recommended_actions": {
                "type": "array",
                "items": {"type": "string"},
                "description": "1-3 immediate next steps, each under 20 words, starting with a verb. Empty if healthy. The intervention plan is made in a separate step.",
            },
            "data_needed_if_deferring": {
                "type": "array",
                "items": {"type": "string"},
                "description": "1-3 concrete items of data that would resolve the ambiguity. Empty if not deferring.",
            },
            "model_opinion_response": {
                "type": "string",
                "description": "One sentence, under 25 words: if you disagree with model_opinion's leaning, the fact that overrides it; if you agree, what corroborates. Empty if model_opinion was not available.",
            },
        },
        "required": [
            "verdict", "temporary_or_structural", "confidence", "defer",
            "leak_dimensions", "attributed_categories", "cited_facts", "narrative",
            "recommended_actions", "data_needed_if_deferring", "model_opinion_response",
        ],
        "additionalProperties": False,
    },
    "strict": True,
}


def _round_floats(value, places: int = 4):
    if isinstance(value, float):
        return round(value, places)
    if isinstance(value, dict):
        return {k: _round_floats(v, places) for k, v in value.items()}
    if isinstance(value, list):
        return [_round_floats(v, places) for v in value]
    return value


def compact_json(payload) -> str:
    """The evidence pack and tool results as the model sees them.

    Floats are rounded to four decimals and separators carry no spaces. This
    is not cosmetic: `0.24930000000000002` costs the same as a sentence, and
    the free Groq tier caps a request at 8,000 tokens per minute — the pack
    crossed that line by ~1% the day the model opinion and the third
    significance window were added, and six of eighteen accounts errored.
    Rounding changes no figure a person would read; None stays None, because
    "not measured" must remain visible."""
    return json.dumps(_round_floats(payload), separators=(",", ":"))


def pack_for_prompt(evidence_pack: dict) -> dict:
    """The evidence pack as the model sees it: every underscore-prefixed key
    removed.

    `build_evidence_pack` carries a `_presentation` block of chart series and
    per-category monthly revenue for the PDF report. That data is far more
    verbose than anything Stage 4 needs, and the prompt is tuned against all
    18 reference accounts, so quietly growing the model's input is a
    regression the scripted-client tests cannot detect. Stripping it here
    makes the report and the prompt independently extensible.
    """
    return {k: v for k, v in evidence_pack.items() if not k.startswith("_")}


class AgentError(Exception):
    """Raised when the agent loop fails to reach a submit_verdict call
    (max iterations exhausted, or the model stopped for an unexpected
    reason) — surfaced as a clear error rather than a silent bad verdict."""


# The fields report.assemble_report indexes directly. leak_dimensions and
# model_opinion_response are read with .get and may be absent — a provider
# without schema enforcement can drop an empty string, and that is not a
# reason to ask the model again.
ESSENTIAL_VERDICT_FIELDS = (
    "verdict", "temporary_or_structural", "confidence", "defer", "narrative",
    "attributed_categories", "cited_facts", "recommended_actions", "data_needed_if_deferring",
)


# A long reply sometimes loses its tool syntax part-way: the model closes a
# field in its own markup ("</narrative>") and writes the next parameter
# after it, and all of that lands inside the string. Seen on 2 of 18
# Sonnet verdicts, both healthy accounts with an empty action list. Cut at
# the first such tag; nothing before it is changed.
_TOOL_MARKUP = re.compile(r"\s*<(?:/[a-z_]+|parameter\b)[^>]*>.*$", re.S | re.I)


def _strip_tool_markup(value):
    if isinstance(value, str):
        return _TOOL_MARKUP.sub("", value)
    if isinstance(value, list):
        return [_strip_tool_markup(v) for v in value]
    if isinstance(value, dict):
        return {k: _strip_tool_markup(v) for k, v in value.items()}
    return value


def _missing_verdict_fields(tool_input) -> list[str]:
    if not isinstance(tool_input, dict):
        return list(ESSENTIAL_VERDICT_FIELDS)
    return [field for field in ESSENTIAL_VERDICT_FIELDS if field not in tool_input]


def _execute_tool(df: pd.DataFrame, account_id: str, name: str, tool_input: dict):
    if name == "get_category_monthly_series":
        return _tool_get_category_monthly_series(df, account_id, tool_input["category"])
    if name == "get_category_seasonal_breakdown":
        return _tool_get_category_seasonal_breakdown(df, account_id, tool_input["category"])
    if name == "get_product_changes":
        return _tool_get_product_changes(df, account_id, tool_input.get("category"))
    if name == "get_monthly_economics":
        return _tool_get_monthly_economics(df, account_id)
    if name == "get_tier_monthly_series":
        return _tool_get_tier_monthly_series(df, account_id)
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
    # Two filters, both load-bearing for the token budget: strip the
    # report-only `_presentation` block, then serialise compactly.
    messages = [{"role": "user", "content": compact_json(pack_for_prompt(evidence_pack))}]

    for _ in range(max_iterations):
        response = client.messages.create(
            model=model,
            max_tokens=MAX_OUTPUT_TOKENS,
            system=SYSTEM_PROMPT,
            tools=tools,
            messages=messages,
        )

        tool_use_blocks = [b for b in response.content if b.type == "tool_use"]

        if response.stop_reason == "max_tokens":
            # The reply ran out of room. Whatever came back is not a
            # verdict: prose with no call, or a tool call whose input was
            # cut off part-way (the API returns what it had, and a partial
            # submit_verdict read as complete is a wrong answer, not a
            # missing one). Hand the reply back and ask for the call. A
            # cut-off tool_use cannot be replayed without a tool_result, so
            # only text is kept. One iteration spent; the verdict is unchanged.
            if not tool_use_blocks:
                messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": TRUNCATION_NUDGE})
            continue

        submit_block = next((b for b in tool_use_blocks if b.name == "submit_verdict"), None)
        if submit_block is not None:
            missing = _missing_verdict_fields(submit_block.input)
            if not missing:
                return _strip_tool_markup(submit_block.input)
            # A call with fields absent (seen once when a long reply lost
            # its structure) is answered like a tool error: name the gap
            # and ask again, rather than crash downstream on a KeyError.
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": [{
                "type": "tool_result", "tool_use_id": submit_block.id, "is_error": True,
                "content": f"submit_verdict was missing required fields: {', '.join(missing)}. "
                           "Call it again with every field.",
            }]})
            continue

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
                    "type": "tool_result", "tool_use_id": block.id, "content": compact_json(result),
                })
            except Exception as e:
                tool_results.append({
                    "type": "tool_result", "tool_use_id": block.id,
                    "content": f"Error: {e}", "is_error": True,
                })
        messages.append({"role": "user", "content": tool_results})

    raise AgentError(f"Exceeded max_iterations={max_iterations} without a submit_verdict call.")
