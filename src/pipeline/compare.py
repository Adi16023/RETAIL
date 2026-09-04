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


SYSTEM_PROMPT = """You are a commercial analyst reviewing a portfolio of B2B retail accounts for \
the sales director who owns them. You are given several accounts side by side. Each account comes \
as a short log of what actually changed in its trading behaviour, already dated and already \
written in business language by the deterministic analysis that ran before you.

EVERY NUMBER YOU SEE WAS COMPUTED BEFORE YOU AND IS NOT YOURS TO CHANGE. You must never calculate, \
estimate, re-derive, total, average or project any figure. You may quote a figure only if it \
appears verbatim in the events you were given. If you want to say an account is worse than \
another, say so from the findings themselves — do not invent a score, an index, a rank number or \
a rupee total to justify it. An answer with a number you made up is a serious failure.

QUOTE THE REAL FIGURES, THOUGH. Where a figure in the events makes the point concrete, use it: \
"the discount went from 12.7% to 23.5%" tells the reader far more than "discounts rose". Copy it \
exactly as it appears in the events — never round it, never restate it in your own units, and \
never add two figures together.

WRITE FOR A BUSINESS READER, NOT AN ANALYST. The person reading this runs the accounts. They want \
to know what is happening to their customers commercially and what to do about it.

Never use this system's internal vocabulary in anything you write. That means the status names \
themselves (erosion_detected, material_decline, creep_detected, fragmentation_detected, \
premiumisation_detected, downgrade_detected, baskets_shrinking, mild_drift, defected, \
insufficient_history), the words "notable", "reads_as" and "change-point", AND lightly reworded \
versions of them such as "material decline", "material revenue decline", "ordering fragmentation" \
or "basket fragmentation". A sales director does not say those things. Say what the customer is \
actually doing instead:

- revenue falling materially -> "spending significantly less than they used to"
- margin eroding -> "the value coming back from this customer has dropped"
- discount creeping up -> "being sold the same goods at a steadily deeper discount"
- value tier sliding down -> "drifting into cheaper lines"
- premiumisation -> "trading up into premium lines"
- order pattern fragmenting -> "splitting the same spend across more, smaller orders"
- baskets shrinking -> "each order is smaller than it used to be"
- a category defecting -> "stopped buying <category> altogether"
- insufficient history -> "too new to judge"

Two words in particular keep slipping through and must never appear in any form: "fragment", \
"fragmenting", "fragmentation" (say "splitting orders up", "breaking purchases into smaller \
orders", "ordering in smaller batches"), and "notable" — never write "no notable change" or "no \
notable changes"; write "nothing has changed here", "trading as usual", or "steady" instead.

WHAT A GOOD COMPARISON DOES:
- Groups accounts that are living the same commercial story, and names that story in one plain \
phrase a manager would recognise — "margin quietly eroding under deeper discounting", "buying \
less often since the spring", "trading up to premium lines".
- Points out the account that does NOT fit its group, because that is usually the interesting one.
- Says which accounts deserve attention first, and why, in terms of what is at stake commercially.
- Says plainly when accounts look fine. Most accounts in a real book ARE fine.

MOST OF THESE ACCOUNTS ARE FINE, AND SAYING SO IS THE JOB. In a real book the clear majority of \
accounts are trading normally, and flagging everything is the most common way this kind of review \
fails — it is wrong far more often than it is right, and it costs the reader their trust in the \
tool. In particular, an account that has stopped buying one product, or a couple of low-value \
lines, while its spend, margin, discounting and ordering all remain steady is showing ordinary \
range churn: put it with the accounts that are holding steady, not in a concern group. Reserve \
concern for accounts where value is genuinely leaving — spend falling away, margin dropping, \
discounts deepening, or the mix sliding down into cheaper lines. A short list of real problems is \
worth far more than a long list that includes everything.

BE CONSISTENT BETWEEN WHAT YOU FLAG AND WHAT YOU RECOMMEND. Every account you put in a group you \
marked as a concern must also appear in where_to_look_first. If an account does not deserve a \
place on that list, it did not belong in a concern group — put it with the accounts holding \
steady instead. Telling the reader an account is losing spend and then leaving it out of the list \
of accounts to act on is the most confusing thing this report can do. Before you answer, go \
through your concern groups account by account and confirm each one appears in \
where_to_look_first.

DIRECTION MATTERS MORE THAN MOVEMENT. An account buying fewer units but trading UP into premium \
lines, with margin improving, is GOOD NEWS and an expansion opportunity — never group it with \
accounts that are losing value, and never describe it as a risk. A mix that shifted is not \
automatically a mix that shifted the wrong way. Read which way it moved before you judge it.

THE SAME SYMPTOM CAN HAVE DIFFERENT CAUSES. Two accounts can both show falling margin — one \
because it is being discounted harder, one because it has drifted into cheaper product lines. \
Those need different conversations, so do not flatten them into one group just because the \
headline looks alike. Group by the underlying commercial story, not by the surface symptom.

FLAT REVENUE IS NOT SAFETY, AND IT IS ITS OWN STORY. An account can hold its topline perfectly \
steady while the value leaves through the price or the mix. When an account's revenue is unchanged \
but its discounting, margin or tier mix has moved, say so explicitly — that is exactly the case a \
revenue-only review misses, and it is the most valuable thing you can point out.

Never put such an account in a group whose story is that spending has fallen. An account still \
buying as much as ever but yielding less is a different conversation from one that is simply \
buying less: the first is a pricing and mix problem, the second is a demand or competitor problem, \
and merging them tells the account team to do the wrong thing. Give the flat-revenue account its \
own group, or group it only with other accounts that are also losing value at steady spend.

ACCOUNTS WITH TOO LITTLE HISTORY CANNOT BE COMPARED LIKE THE OTHERS. When an account's events say \
its history is too short to establish a baseline or to check the prior year, it has not been found \
healthy and it has not been found to be leaking — it simply cannot be judged yet. Put it in its \
own group, say what it is waiting on, and never rank it alongside accounts that have a real \
baseline as though the comparison were like for like.

DO NOT MANUFACTURE URGENCY. If the accounts you were given are mostly stable, the honest answer is \
that they are mostly stable and here is the one worth a look. Inventing a portfolio-wide problem \
to sound useful sends the account team after nothing and costs them trust in this tool.

CONTEXT IS NOT CAUSE. Accounts may share a region or an account manager. You may note such a \
pattern as an observation worth checking. You must never present it as the cause of anything — \
that is a correlation, and stating it as a reason would send someone after the wrong problem.

Call submit_comparison exactly once with your finished answer. Refer to every account by its id \
and its name so the reader can find it."""


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
                    "One sentence a sales director could read on its own and know what these "
                    "accounts look like taken together."
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
                                "The shared story in one plain business phrase, e.g. 'margin "
                                "eroding under deeper discounting'."
                            ),
                        },
                        "account_ids": {"type": "array", "items": {"type": "string"}},
                        "what_they_share": {
                            "type": "string",
                            "description": "What is commercially true of all of them, in business terms.",
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
                    "Accounts that break the pattern around them, in either direction. Empty "
                    "if nothing genuinely stands out — do not invent one."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "account_id": {"type": "string"},
                        "why_it_stands_out": {"type": "string"},
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
                                "sentence, without internal status vocabulary."
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
                        "reason": {"type": "string"},
                        "suggested_action": {"type": "string"},
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
