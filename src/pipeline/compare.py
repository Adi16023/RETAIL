"""
Cross-account comparison — a second, optional LLM touchpoint that reads
several accounts at once and answers a portfolio question the per-account
investigation cannot: which of these accounts are telling the same story,
which one is the odd one out, and where should the account team look first.

It computes NOTHING. Stage 3 already produced every figure, and
`timeline.build_timeline` already turned those figures into business
statements — what moved, by how much, and what it means for the account.
This module selects those statements for the chosen accounts and hands them
to the model. That is the whole design: the comparison is a reading of
existing analysis, not a new analysis, so it can never disagree with the
per-account verdict about what the numbers say.

Two things are deliberately withheld from the prompt:

  * `method` — the thresholds each call was judged against. That is reviewer
    material (see reporting/pdf.py's brief/full split); a business reader
    comparing accounts wants the finding, not the rule behind it.
  * every underscore-prefixed key — the presentation-only series, stripped
    the same way `agent.pack_for_prompt` strips them.

Sending the raw evidence packs instead would be roughly ten times the tokens
for the same 18 accounts, in detector vocabulary the model would have to
re-translate into business language on every run.

The client is injected, exactly as in `agent.py`, so tests can drive the
whole path with a scripted fake and no API key — see tests/test_compare.py.
"""

from __future__ import annotations

import json

import pandas as pd

from .evidence import build_evidence_pack
from .timeline import build_timeline

DEFAULT_MODEL = "claude-opus-5"

# A comparison is one prompt, so the account list is bounded by what fits in
# it usefully rather than by what the model will technically accept. Around
# thirty accounts the reader stops being able to act on the answer anyway —
# past that the right feature is a filter, not a longer list.
MAX_ACCOUNTS = 30

# Comparing one account with itself is not a comparison.
MIN_ACCOUNTS = 2

# The reply length is driven almost entirely by how many accounts were sent:
# every account earns a line in `per_account`, a place in some group, and —
# if it was called a concern — an entry in `where_to_look_first` with a
# reason and an action. A fixed budget that comfortably fits five accounts
# silently truncates at eighteen, and a truncated tool call comes back from
# the provider as an unparseable-arguments error rather than as a short
# answer, so the whole comparison fails. Budget per account instead.
#
# These numbers are generous against the size of the visible reply on
# purpose: a reasoning model spends tokens thinking before it emits the tool
# call, and that thinking is charged to the same budget. Sizing this from
# the length of the JSON that comes back truncates in the middle of the
# answer — measured, not guessed.
OUTPUT_TOKENS_BASE = 2048
OUTPUT_TOKENS_PER_ACCOUNT = 400


def _output_token_budget(account_count: int) -> int:
    return OUTPUT_TOKENS_BASE + OUTPUT_TOKENS_PER_ACCOUNT * account_count


SYSTEM_PROMPT = """You are a commercial analyst reading several B2B retail accounts side by side \
for the sales director who owns them. Each account arrives as a short dated log of what changed in \
its trading, already written in business language by the deterministic analysis that ran before \
you. Your answer is shown on screen as a short structured brief, so every field must be tight.

EVERY NUMBER WAS COMPUTED BEFORE YOU. Never calculate, estimate, total, average, rank by a score or \
project a figure. Quote a figure only if it appears verbatim in the events, copied exactly — no \
rounding, no new units, no adding two figures. A number you made up is a serious failure.

QUOTE THE REAL FIGURES, WRITTEN FOR A MANAGER. "Discount went from 12.7% to 23.5%" beats \
"discounts rose". Write rupees as ₹1,23,456 with separators, percentages to one decimal, months \
as "April 2026" never "2026-04", a change in a rate as "down 12 points" never "-12.26pp", and \
never quote a p-value, a ratio like 0.3215, or a field name. At most two figures per sentence.

WRITE FOR A BUSINESS READER. Never use internal status names (erosion_detected, material_decline, \
creep_detected, fragmentation_detected, premiumisation_detected, downgrade_detected, \
baskets_shrinking, mild_drift, defected, insufficient_history), the words "notable", "reads_as", \
"change-point", or reworded versions such as "material decline" or "ordering fragmentation". Say \
what the customer is doing:
- revenue falling -> "spending significantly less than they used to"
- margin eroding -> "the value coming back from this customer has dropped"
- discount creeping -> "sold the same goods at a steadily deeper discount"
- tier sliding down -> "drifting into cheaper lines"; premiumisation -> "trading up into premium lines"
- order pattern splitting -> "splitting the same spend across more, smaller orders"
- baskets shrinking -> "each order is smaller than it used to be"
- category defecting -> "stopped buying <category> altogether"
- insufficient history -> "too new to judge"
Never write "fragment" in any form, and never "notable" — say "steady" or "trading as usual".

MOST ACCOUNTS ARE FINE, AND SAYING SO IS THE JOB. Flagging everything is the most common failure. \
An account that dropped one product or a couple of low-value lines while spend, margin, \
discounting and ordering stayed steady is ordinary range churn — it belongs with the steady \
accounts. Reserve concern for value genuinely leaving: spend falling away, margin dropping, \
discounts deepening, or the mix sliding into cheaper lines. Do not manufacture urgency.

BE CONSISTENT. Every account in a concern group must appear in where_to_look_first, and only \
those. Check this account by account before answering.

DIRECTION MATTERS MORE THAN MOVEMENT. Trading up into premium lines with margin improving is good \
news and an expansion opportunity — never a risk, never grouped with accounts losing value.

GROUP BY THE COMMERCIAL STORY, NOT THE SYMPTOM. Margin falling from deeper discounting and margin \
falling from drifting into cheaper lines need different conversations — separate groups.

FLAT REVENUE IS NOT SAFETY. An account holding its topline while value leaves through price or \
mix is its own story. Say so explicitly, and never group it with accounts whose spend has fallen.

TOO LITTLE HISTORY CANNOT BE COMPARED. An account whose events say its history is too short is \
neither healthy nor leaking. Put it in its own cannot_judge_yet group, say what it is waiting on, \
and never rank it beside accounts with a real baseline.

CONTEXT IS NOT CAUSE. A shared region or account manager is an observation, never a reason.

LENGTH. headline under 25 words. story under 8 words. what_they_share under 25 words. \
why_it_stands_out under 25 words, at most 3 standouts. business_read under 20 words. reason under \
25 words. suggested_action under 15 words, starting with a verb. Refer to accounts by id.

Call submit_comparison exactly once."""


def _account_digest(df: pd.DataFrame, account_id: str) -> dict:
    """One account as the comparison model sees it.

    Identity and history are carried because they are how a business reader
    orients ("the two West accounts under the same manager", "this one only
    has four months"). Every one of these values was computed by an earlier
    stage and is copied here unchanged.
    """
    pack = build_evidence_pack(df, account_id)
    history = pack.get("history") or {}
    sufficiency = pack.get("data_sufficiency") or {}

    return {
        "account_id": account_id,
        "account_name": pack.get("account_name"),
        "region": pack.get("region"),
        "account_manager": pack.get("account_manager"),
        "months_of_history": history.get("months_of_history"),
        "order_count": history.get("order_count"),
        "history_is_sufficient_to_judge": sufficiency.get("label") == "sufficient",
        # Only what MOVED. A timeline also records the checks that came back
        # normal, which are essential when justifying a single verdict but
        # are noise when reading eighteen accounts at once — they would be
        # the same handful of reassuring lines repeated on every account.
        "what_changed": [
            {
                "when": event["when"],
                "headline": event["headline"],
                "detail": event["detail"],
                "reads_as": event["reads_as"],
            }
            for event in build_timeline(pack)
            if event["notable"]
        ],
    }


def build_comparison_digest(df: pd.DataFrame, account_ids: list[str]) -> dict:
    """The full prompt payload for a set of accounts.

    Raises ValueError rather than silently truncating: a comparison that
    quietly dropped accounts the user selected would be answering a
    different question than the one they asked.
    """
    unique_ids = sorted(set(account_ids))
    if len(unique_ids) < MIN_ACCOUNTS:
        raise ValueError(
            f"Select at least {MIN_ACCOUNTS} accounts to compare; got {len(unique_ids)}."
        )
    if len(unique_ids) > MAX_ACCOUNTS:
        raise ValueError(
            f"Select at most {MAX_ACCOUNTS} accounts to compare; got {len(unique_ids)}. "
            "Narrow the selection to the accounts you actually want read side by side."
        )
    return {"accounts": [_account_digest(df, account_id) for account_id in unique_ids]}


SUBMIT_COMPARISON_TOOL = {
    "name": "submit_comparison",
    "description": (
        "Submit your finished side-by-side reading of these accounts. Call this exactly once."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "headline": {
                "type": "string",
                "description": (
                    "One sentence, under 25 words, a sales director could read on its own and "
                    "know what these accounts look like taken together."
                ),
            },
            "groups": {
                "type": "array",
                "description": (
                    "Accounts living the same commercial story. An account belongs to exactly "
                    "one group. Accounts that look fine belong in a group that says so."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "story": {
                            "type": "string",
                            "description": (
                                "The shared story in one plain business phrase under 8 words, "
                                "e.g. 'margin eroding under deeper discounting'."
                            ),
                        },
                        "account_ids": {"type": "array", "items": {"type": "string"}},
                        "what_they_share": {
                            "type": "string",
                            "description": "What is commercially true of all of them, under 25 words.",
                        },
                        "reads_as": {
                            "type": "string",
                            "enum": ["concern", "reassuring", "cannot_judge_yet", "context"],
                            "description": (
                                "How this group bears on the book. Use cannot_judge_yet for "
                                "accounts whose history is too short to assess."
                            ),
                        },
                    },
                    "required": ["story", "account_ids", "what_they_share", "reads_as"],
                    "additionalProperties": False,
                },
            },
            "standouts": {
                "type": "array",
                "description": (
                    "At most 3 accounts that break the pattern around them, in either direction. "
                    "Empty if nothing genuinely stands out — do not invent one."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "account_id": {"type": "string"},
                        "why_it_stands_out": {"type": "string", "description": "Under 25 words."},
                    },
                    "required": ["account_id", "why_it_stands_out"],
                    "additionalProperties": False,
                },
            },
            "per_account": {
                "type": "array",
                "description": "One business line for every account you were given. Omit none.",
                "items": {
                    "type": "object",
                    "properties": {
                        "account_id": {"type": "string"},
                        "business_read": {
                            "type": "string",
                            "description": (
                                "What is happening to this customer commercially, in one "
                                "sentence under 20 words, without internal status vocabulary."
                            ),
                        },
                    },
                    "required": ["account_id", "business_read"],
                    "additionalProperties": False,
                },
            },
            "where_to_look_first": {
                "type": "array",
                "description": (
                    "The accounts worth attention, most pressing first, with the commercial "
                    "reason. Empty if none of them need attention right now."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "account_id": {"type": "string"},
                        "reason": {"type": "string", "description": "What is at stake, under 25 words, with the figure that shows it."},
                        "suggested_action": {"type": "string", "description": "One action under 15 words, starting with a verb."},
                    },
                    "required": ["account_id", "reason", "suggested_action"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["headline", "groups", "standouts", "per_account", "where_to_look_first"],
        "additionalProperties": False,
    },
    "strict": True,
}


class ComparisonError(Exception):
    """Raised when the model does not return a submit_comparison call."""


def compare_accounts(client, digest: dict, model: str = DEFAULT_MODEL) -> dict:
    """One model call over several accounts. Returns the parsed
    submit_comparison input dict.

    Deliberately a single call with no drill-down tools, unlike Stage 4: the
    per-account investigation is where drilling into one account's data
    belongs, and giving this step its own route back to the raw data would
    let it reach a figure the account's own verdict never saw.
    """
    response = client.messages.create(
        model=model,
        max_tokens=_output_token_budget(len(digest.get("accounts") or [])),
        system=SYSTEM_PROMPT,
        tools=[SUBMIT_COMPARISON_TOOL],
        messages=[{"role": "user", "content": json.dumps(digest)}],
    )

    submit_block = next(
        (b for b in response.content if b.type == "tool_use" and b.name == "submit_comparison"),
        None,
    )
    if submit_block is None:
        raise ComparisonError(
            f"Model stopped (stop_reason={response.stop_reason!r}) without calling "
            "submit_comparison."
        )
    return submit_block.input
