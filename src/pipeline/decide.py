"""
Intervention pricing and the commercial recommendation — what the business
should DO about a leak the pipeline has already found and sized.

Everything before this stage answers "what is wrong and how much is it
costing?". That is a diagnosis, and nobody acts on a diagnosis: an account
owner deciding whether to reopen a discount at renewal needs to know what
each option is worth and how wrong they can afford to be about the
customer's reaction.

The module has two clearly separated halves, and the split is the whole
design:

  price_options()   deterministic. Prices every intervention available on
                    this account. No model, no network, no judgement.
  recommend()       ONE optional model call. Chooses among the priced
                    options and argues the case. Computes nothing.

The calculator is the product; the recommendation is a layer on top of it.
If the model call fails, or no key is configured, the priced options are
still on screen and still correct — which is why this is a separate call
rather than folded into Stage 4.

WHY A SECOND CALL SITE. plan.md records "one LLM call per analysis, not
four", justified on live reliability. That still holds: one call decides the
verdict. This one is an optional output surface — it runs only when the user
asks for it, it reads a finished report, it can never change a verdict, and
if it fails the page degrades to the calculator. Folding it into Stage 4 was
measured and did not bias the verdict on a five-account probe, but it would
have coupled a failure here to the verdict, required re-validating a prompt
tuned across all 18 accounts, and fixed the recommendation before the user
had chosen a scenario — which is the interaction this feature exists for.

THE ONE THING THIS MUST NEVER DO is predict how the customer reacts. Nothing
in a transaction history supports an elasticity estimate. What the numbers
below DO support is a break-even: the volume the business could lose before
an intervention stops paying. That is a tolerance, not a forecast, and the
distinction is load-bearing — it is the difference between quantified
honesty and a fabricated number.
"""

from __future__ import annotations

import json

from .detect import MARGIN_EROSION_PP

DEFAULT_MODEL = "claude-sonnet-5"

# How much of the discount creep to claw back, as fractions of the gap
# between where discounting started and where it is now. Fractions rather
# than fixed percentages so the ladder adapts to the size of the problem: a
# 3pp creep and a 20pp creep both get the same four meaningful choices
# instead of a ladder tuned to one account.
RECOVERY_FRACTIONS = (0.33, 0.50, 0.75, 1.00)


def _margin_rate(evidence_pack: dict) -> float | None:
    """The rate recovered business is assumed to earn: what this account
    earns TODAY. Using the healthier baseline rate would quietly inflate
    every recovery figure with an improvement nobody has committed to."""
    margin = evidence_pack.get("margin") or {}
    rate = margin.get("recent_margin_pct")
    return rate if rate and 0 < rate < 1 else None


def _health_gap(evidence_pack: dict) -> dict:
    """What "fixed" actually means on this account, in the pipeline's own terms.

    An intervention that recovers real money can still leave the account
    reading as leaking, and a recommendation that does not say so invites a
    manager to act, declare victory, and be surprised at the next review.
    Health is not a judgement here — it is the same threshold Stage 3 uses.
    """
    margin = evidence_pack.get("margin") or {}
    revenue = evidence_pack.get("overall_revenue") or {}
    baseline_pct = margin.get("baseline_margin_pct")
    recent_pct = margin.get("recent_margin_pct")
    baseline_revenue = revenue.get("baseline_monthly_median")
    recent_revenue = revenue.get("recent_monthly_rate")

    return {
        # Margin must land within MARGIN_EROSION_PP of where it was.
        "margin_rate_needed_pct": (
            round((baseline_pct * 100) - MARGIN_EROSION_PP, 1)
            if baseline_pct is not None else None
        ),
        "margin_rate_today_pct": None if recent_pct is None else round(recent_pct * 100, 1),
        # Revenue-side levers have to close the gap back to the old run rate.
        "monthly_revenue_gap": (
            round(baseline_revenue - recent_revenue)
            if None not in (baseline_revenue, recent_revenue) and baseline_revenue > recent_revenue
            else 0
        ),
    }


def _core(lever: str, lever_label: str, option: str, fraction: float,
          revenue_per_month: float, margin_per_month: float) -> dict:
    return {
        "lever": lever,
        "lever_label": lever_label,
        "option": option,
        "share_recovered_pct": round(fraction * 100),
        "recovers_revenue_per_month": round(revenue_per_month),
        "recovers_margin_per_month": round(margin_per_month),
        "recovers_margin_over_12_months": round(margin_per_month * 12),
    }


def _price_recovery(evidence_pack: dict) -> list[dict]:
    """Discount creep: the same goods sold at a deeper discount.

    Changing price does not change what the goods cost, so the whole price
    movement lands in margin — which is why winding the discount back to
    where it started reproduces this account's own historical margin rate.
    """
    discount = evidence_pack.get("discount") or {}
    margin = evidence_pack.get("margin") or {}
    revenue = evidence_pack.get("overall_revenue") or {}
    if discount.get("status") != "creep_detected":
        return []

    current_revenue = revenue.get("recent_monthly_rate")
    current_margin_pct = margin.get("recent_margin_pct")
    current_discount = discount.get("recent_avg_discount_pct")
    baseline_discount = discount.get("baseline_avg_discount_pct")
    if None in (current_revenue, current_margin_pct, current_discount, baseline_discount):
        return []
    if current_discount <= baseline_discount or current_discount >= 1.0 or current_revenue <= 0:
        return []

    current_margin = current_revenue * current_margin_pct
    monthly_cost = current_revenue - current_margin
    if monthly_cost <= 0:
        return []

    gap = current_discount - baseline_discount
    options, seen = [], set()
    for fraction in RECOVERY_FRACTIONS:
        target = round(current_discount - gap * fraction, 4)
        if target in seen or target >= current_discount or target < 0:
            continue
        seen.add(target)
        new_revenue = current_revenue * ((1 - target) / (1 - current_discount))
        new_margin = new_revenue - monthly_cost
        recovered = new_margin - current_margin
        if recovered <= 0 or new_margin <= 0:
            continue
        option = _core(
            "price_recovery", "Reset the price",
            f"negotiate the discount back to {round(target * 100, 1)}%",
            fraction, new_revenue - current_revenue, recovered,
        )
        option.update({
            "discount_target_pct": round(target * 100, 1),
            # Pre-computed so the model never subtracts two given figures.
            "reduction_from_today_pp": round((current_discount - target) * 100, 1),
            "resulting_margin_rate_pct": round(new_margin / new_revenue * 100, 1),
            # The honest form of "what if they push back": not a prediction
            # of lost volume, but how much could be lost before the
            # intervention stops being worth doing.
            "volume_that_could_be_lost_before_this_stops_paying_pct": round(
                (1 - current_margin / new_margin) * 100
            ),
            "restores_original_discount": fraction == 1.00,
        })
        option.update(_health_verdict_margin(evidence_pack, new_margin / new_revenue * 100))
        options.append(option)
    return options


def _health_verdict_margin(evidence_pack: dict, resulting_rate_pct: float) -> dict:
    """Does this option put the margin rate back inside the healthy band?"""
    needed = _health_gap(evidence_pack)["margin_rate_needed_pct"]
    if needed is None:
        return {"restores_account_to_healthy": None, "short_of_healthy_by_pp": None}
    short = round(needed - resulting_rate_pct, 1)
    return {
        "restores_account_to_healthy": short <= 0,
        "short_of_healthy_by_pp": max(short, 0.0),
    }


def _health_verdict_revenue(evidence_pack: dict, revenue_recovered: float) -> dict:
    """Does this option close the gap back to the old monthly run rate?"""
    gap = _health_gap(evidence_pack)["monthly_revenue_gap"]
    if not gap:
        return {"restores_account_to_healthy": None, "still_short_per_month": None}
    # One rupee of tolerance: the gap is rounded for display, so a full
    # recovery can land a fraction below it and read as still leaking.
    return {
        "restores_account_to_healthy": revenue_recovered >= gap - 1,
        "still_short_per_month": max(round(gap - revenue_recovered), 0),
    }


def _win_back(evidence_pack: dict) -> list[dict]:
    """A category that stopped completely, and what getting it back is worth.

    Sized from the run rate the category held BEFORE it stopped — a figure
    Stage 3 already computed — valued at the margin rate the account earns
    today. `max_price_concession_pct` is the number a win-back negotiation
    actually turns on: discounting cuts price without cutting cost, so
    margin on recovered business runs out exactly when the concession
    reaches the margin rate.
    """
    rate = _margin_rate(evidence_pack)
    if rate is None:
        return []
    baseline_share = (evidence_pack.get("category_mix") or {}).get("baseline_share") or {}

    lost = []
    for change in evidence_pack.get("category_changes") or []:
        if not change.get("defected"):
            continue
        monthly = change.get("before_monthly_median")
        if not monthly or monthly <= 0:
            continue
        lost.append((change["category"], monthly, change.get("consecutive_months_at_zero", 0),
                     baseline_share.get(change["category"])))
    if not lost:
        return []

    total_monthly = sum(m for _, m, _, _ in lost)
    names = ", ".join(c for c, _, _, _ in lost)
    longest_gone = max(z for _, _, z, _ in lost)

    options = []
    for fraction in RECOVERY_FRACTIONS:
        revenue_back = total_monthly * fraction
        option = _core(
            "win_back", "Win back the lost line",
            f"win back {round(fraction * 100)}% of {names}",
            fraction, revenue_back, revenue_back * rate,
        )
        option.update({
            "categories": [c for c, _, _, _ in lost],
            "lost_revenue_per_month": round(total_monthly),
            "months_already_gone": longest_gone,
            "value_lost_so_far": round(total_monthly * longest_gone),
            "share_of_account_baseline_pct": (
                round(sum(s for *_, s in lost if s) * 100, 1)
                if any(s for *_, s in lost) else None
            ),
            # Winning business back usually costs something at the table.
            # Price concession eats margin one-for-one because cost does not
            # move, so this is the point at which the recovered business
            # stops contributing anything at all.
            "max_price_concession_pct": round(rate * 100, 1),
            "margin_if_won_back_at_half_that_concession": round(
                revenue_back * (rate - rate / 2)
            ),
        })
        option.update(_health_verdict_revenue(evidence_pack, revenue_back))
        options.append(option)
    return options


def _margin_recovery(evidence_pack: dict) -> list[dict]:
    """Margin eroding for a reason other than discounting — normally the mix
    sliding into cheaper lines.

    Priced as closing part of the gap between the margin rate this account
    used to earn and the one it earns now, at today's revenue. Skipped when
    discounting explains the erosion, because `_price_recovery` already
    prices that and counting both would sell the same rupees twice.
    """
    margin = evidence_pack.get("margin") or {}
    discount = evidence_pack.get("discount") or {}
    revenue = evidence_pack.get("overall_revenue") or {}
    if margin.get("status") != "erosion_detected" or discount.get("status") == "creep_detected":
        return []

    baseline_pct = margin.get("baseline_margin_pct")
    recent_pct = margin.get("recent_margin_pct")
    current_revenue = revenue.get("recent_monthly_rate")
    if None in (baseline_pct, recent_pct, current_revenue) or baseline_pct <= recent_pct:
        return []

    full_gap = current_revenue * (baseline_pct - recent_pct)
    tier = evidence_pack.get("tier_mix") or {}
    downgrade = tier.get("status") == "downgrade_detected"
    label = "Win back the premium lines" if downgrade else "Restore the margin rate"

    options = []
    for fraction in RECOVERY_FRACTIONS:
        recovered = full_gap * fraction
        option = _core(
            "mix_recovery" if downgrade else "margin_recovery", label,
            f"win back {round(fraction * 100)}% of the lost premium mix"
            if downgrade else f"recover {round(fraction * 100)}% of the lost margin rate",
            fraction, 0.0, recovered,
        )
        option.update({
            "margin_rate_today_pct": round(recent_pct * 100, 1),
            "margin_rate_before_pct": round(baseline_pct * 100, 1),
            "resulting_margin_rate_pct": round(
                (recent_pct + (baseline_pct - recent_pct) * fraction) * 100, 1
            ),
            "high_tier_share_change_pp": tier.get("high_tier_share_change_pp"),
        })
        option.update(_health_verdict_margin(
            evidence_pack, (recent_pct + (baseline_pct - recent_pct) * fraction) * 100
        ))
        options.append(option)
    return options


def _revenue_recovery(evidence_pack: dict) -> list[dict]:
    """A whole-account decline that no single category explains.

    Only priced when no category defected: if a named line accounts for the
    loss, `_win_back` already sizes it, and pricing a whole-account recovery
    on top would count the same rupees twice — the same double-count guard
    impact.py applies.
    """
    if any(c.get("defected") for c in evidence_pack.get("category_changes") or []):
        return []
    if (evidence_pack.get("revenue_decline") or {}).get("status") != "material_decline":
        return []

    rate = _margin_rate(evidence_pack)
    revenue = evidence_pack.get("overall_revenue") or {}
    baseline = revenue.get("baseline_monthly_median")
    recent = revenue.get("recent_monthly_rate")
    if rate is None or None in (baseline, recent) or baseline <= recent:
        return []

    gap = baseline - recent
    options = []
    for fraction in RECOVERY_FRACTIONS:
        revenue_back = gap * fraction
        option = _core(
            "revenue_recovery", "Rebuild the account",
            f"rebuild {round(fraction * 100)}% of the lost monthly spend",
            fraction, revenue_back, revenue_back * rate,
        )
        option.update({
            "spend_lost_per_month": round(gap),
            "monthly_spend_today": round(recent),
            "monthly_spend_before": round(baseline),
        })
        option.update(_health_verdict_revenue(evidence_pack, revenue_back))
        options.append(option)
    return options


def _basket_recovery(evidence_pack: dict) -> list[dict]:
    """Deliberately prices nothing.

    Shrinking baskets and splitting orders are an early warning: the account
    is buying the same goods in a different shape, and its revenue has not
    materially fallen yet. There is therefore no loss to recover, and any
    "recovery" figure would be invented. Applying the old average order to
    today's higher order count — the obvious formula — claimed Rs 170,192 a
    month on ACC-108 against a real revenue gap of Rs 17,303: ten times the
    money that has actually moved.

    The lever is still real and the decision layer still recommends it; what
    it does not get is a price tag it has not earned. Acting here is cheap
    precisely because nothing has been lost, and saying so is worth more
    than a number that would not survive being checked.
    """
    return []


def price_options(evidence_pack: dict, impact: dict | None = None) -> list[dict]:
    """Every intervention available on this account, priced.

    One entry per lever per recovery depth. Levers are matched to the leak,
    because a discount reset and a win-back are different conversations and
    recommending the wrong one sends the account team after the wrong thing.
    Two double-count guards apply: a whole-account rebuild is not priced when
    a named category explains the loss, and a margin-rate recovery is not
    priced when discounting is already being reset.

    Returns [] only when nothing is actually wrong, or when the file lacks
    the margin data every valuation here depends on.
    """
    return (
        _price_recovery(evidence_pack)
        + _win_back(evidence_pack)
        + _margin_recovery(evidence_pack)
        + _revenue_recovery(evidence_pack)
        + _basket_recovery(evidence_pack)
    )


def levers_available(options: list[dict]) -> list[str]:
    """Distinct levers present, in the order they were priced."""
    seen = []
    for option in options:
        if option["lever"] not in seen:
            seen.append(option["lever"])
    return seen


def _moved_totals(evidence: dict) -> dict:
    """What the lines that moved were worth, added up once, here."""
    moved = [
        c for c in (evidence.get("product_changes") or [])
        if c.get("status") in ("disappeared", "declined")
    ]
    if not moved:
        return {}
    was = sum(c["baseline_monthly_avg"] for c in moved)
    now = sum(c["recent_monthly_avg"] for c in moved)
    return {
        "lines_that_moved": len(moved),
        "were_worth_per_month": round(was, 2),
        "now_worth_per_month": round(now, 2),
        "fallen_away_per_month": round(was - now, 2),
    }


def build_decision_input(report: dict, options: list[dict]) -> dict:
    """What the adviser model is shown: the finished verdict in business
    terms, the money, and the priced options — never the raw evidence pack.

    Uses the report's own timeline rather than the pack for the same reason
    the comparison does: those are business statements the deterministic
    stages already wrote, so the adviser reads findings rather than detector
    vocabulary it would have to translate.
    """
    impact = report.get("financial_impact") or {}
    margin_impact = impact.get("margin_impact") or {}
    evidence = report.get("evidence") or {}
    history = evidence.get("history") or {}

    return {
        "account": {
            "account_id": report.get("account_id"),
            "account_name": report.get("account_name"),
            "region": evidence.get("region"),
            "account_manager": evidence.get("account_manager"),
            "account_manager_changed_mid_history": bool(evidence.get("account_manager_changed")),
            "months_of_history": history.get("months_of_history"),
            "data_sufficiency": (report.get("data_sufficiency") or {}).get("label"),
        },
        "verdict": {
            "verdict": report.get("verdict"),
            "temporary_or_structural": report.get("temporary_or_structural"),
            "confidence": report.get("confidence"),
            "defer": report.get("defer"),
            "leak_dimensions": report.get("leak_dimensions") or [],
        },
        "what_changed": [
            {
                "when": event["when"],
                "headline": event["headline"],
                "detail": event["detail"],
                "reads_as": event["reads_as"],
            }
            for event in (report.get("evidence_timeline") or [])
            if event.get("notable")
        ],
        "money": {
            "monthly_margin_at_risk": margin_impact.get("monthly_margin_at_risk"),
            "annual_margin_exposure": margin_impact.get("annualized_margin_at_risk"),
            "monthly_revenue_at_risk": impact.get("total_monthly_revenue_at_risk"),
            # Handed over already expressed in percent. The pipeline carries
            # this as a ratio, and a model given 0.4768 will write "47.7%" —
            # correct, trivial, and still arithmetic it was told not to do.
            # Every quantity it might want must arrive in the form it needs.
            "severity_pct_of_baseline": (
                None if impact.get("overall_severity_pct_of_baseline") is None
                else round(impact["overall_severity_pct_of_baseline"] * 100, 1)
            ),
            "priority": (report.get("prioritization") or {}).get("priority"),
        },
        # The named lines that actually moved, with what each was worth. Stage 3
        # already worked this out, so an adviser that says "pull the last six
        # months of orders to find which SKUs were lost" is setting homework
        # it has the answer to. Naming them is the difference between a plan
        # and a to-do list.
        "products_that_moved": [
            {
                "product": change["product_id"],
                "category": change["category"],
                "status": change["status"],
                "was_per_month": change["baseline_monthly_avg"],
                "now_per_month": change["recent_monthly_avg"],
            }
            for change in (evidence.get("product_changes") or [])
            if change.get("status") in ("disappeared", "declined")
        ],
        # Totals, precomputed. Handed a list of six products and no total, the
        # model summed them itself and produced a figure matching nothing —
        # the exact failure the no-arithmetic rule exists to prevent. A list
        # without its total is an invitation to compute one.
        "products_that_moved_totals": _moved_totals(evidence),
        "categories_that_stopped": [
            {
                "category": change["category"],
                "was_per_month": change.get("before_monthly_median"),
                "months_at_zero": change.get("consecutive_months_at_zero"),
            }
            for change in (evidence.get("category_changes") or [])
            if change.get("defected")
        ],
        "priced_options": options,
        "note": (
            "priced_options were computed by deterministic code from this account's own "
            "figures. The break-even values are TOLERANCES, not forecasts of what the "
            "customer will do."
        ),
    }


SYSTEM_PROMPT = """You are a commercial decision adviser to the sales director who owns this B2B \
retail account. The investigation has already run: the leak is found, sized in rupees, and every \
available intervention has been PRICED by deterministic code. You do not find the problem and you \
do not do arithmetic. You say what the business should do, in a brief a manager reads on screen \
in under a minute.

EVERY NUMBER WAS COMPUTED BEFORE YOU. Never calculate, estimate, total, annualise, subtract or \
project a figure. Quote a figure only if it appears verbatim in the material — every quantity you \
need, including the size of the discount reduction, is already provided. If a number does not \
exist, make the point without it. A figure you produced yourself is a serious failure.

THAT INCLUDES ADDING UP AND ROUNDING. `products_that_moved_totals` and each priced option already \
carry their totals. ₹14,944 is not "about ₹15,000": copy every figure exactly, and drop \
"approximately", "roughly" and "around" — hedging a figure means you changed it.

WRITE NUMBERS FOR A MANAGER. Rupees as ₹14,944 with separators, percentages to one decimal as \
given, months as "April 2026" never "2026-04", a change in a rate as "down 5 points" never \
"-5.2pp". Never quote a field name, a p-value or a raw ratio. At most two figures per sentence, \
and only where the figure changes what the reader does.

NEVER PREDICT HOW THE CUSTOMER WILL REACT. You have no price sensitivity, contract terms, \
competitor or budget. The break-even figures are TOLERANCES, not forecasts: "could lose up to 38% \
of volume before this stops paying" is correct; "will probably lose about 10%" is fabrication. \
Argue only from this account's own trading history — no market rates, no "customers typically".

MATCH THE LEVER TO THE LEAK:
- Discount creeping up -> a pricing decision, the most reversible leak; revisit at renewal.
- Mix sliding into cheaper lines -> a selling decision: why did the premium lines stop being chosen?
- A category stopped completely -> the business is already elsewhere; a win-back on a longer \
timeline, never a quick fix.
- Orders splitting into more, smaller baskets -> an early warning; nothing lost yet, cheapest \
moment to act.
- Spend falling across the whole book, no category responsible -> a relationship review, not a lever.
- Margin falling with no discount movement -> establish the cause before anyone negotiates.
- Trading up or growing -> GOOD NEWS; an expansion conversation, never an intervention.

NO ACTION IS A REAL ANSWER. Use no_action when the account is healthy or the recoverable amount is \
too small to be worth a manager's quarter. If the investigation deferred, use gather_data and say \
what is missing. If no options were priced, still give the decision — the lever, the steps, the \
owner, how anyone will know — without inventing a figure.

WEIGH THE WHOLE PICTURE, NOT THE BIGGEST NUMBER. Consider effort and relationship capital, whether \
the money can simply be taken back or must be sold to win back, how wide the break-even tolerance \
is, how long the account and its owner have been in place, whether this waits for renewal, and \
proportion — a ten-point swing is a different conversation from a two-point correction. Say why \
you rejected the stronger option and why you rejected the weaker one.

A PARTIAL RECOVERY IS A GOOD OUTCOME. Choose the option you would actually back, not the largest; \
recommending the maximum by default is not judgement. Each option carries \
`restores_account_to_healthy` on the analysis's own threshold — copy it, and if the chosen option \
does not restore health say plainly what is left over and what would close it.

CONTEXT IS NOT CAUSE. A manager change, a region or a gap in the history may temper the ask; \
never present one as the reason the leak happened.

NEVER SEND THE READER TO FIND SOMETHING YOU WERE ALREADY GIVEN. `products_that_moved` and \
`categories_that_stopped` name every line that shrank or stopped, with what each was worth a \
month. Name them and their figures in the steps — never "pull the orders to identify which lines \
were lost". A line at zero is the most concrete thing to put in front of a buyer; name it. Reserve \
check_before_acting for what the data cannot settle: contract terms, who the competitor is, \
whether the customer discontinued a line themselves.

A TARGET IS NOT A DECISION. what_to_do is the actual work — who is called, what is pulled up, what \
is proposed, what is signed off — in order, starting with the first step.

WRITE FOR THE PERSON WALKING INTO THE ROOM. Plain, specific, factual. Never use internal status \
names (erosion_detected, creep_detected, material_decline, downgrade_detected, \
premiumisation_detected, fragmentation_detected, baskets_shrinking, mild_drift, defected, \
insufficient_history) or reworded versions. Say what the customer has been doing.

LENGTH. headline under 20 words. expected_result one sentence under 35 words with the chosen \
option's own figures. rationale under 60 words. why_not_more_aggressive and \
why_not_less_aggressive under 30 words each. what_to_do 3 to 5 steps, each under 25 words, \
starting with a verb. talking_points at most 3, each under 20 words. check_before_acting at most \
3. how_we_will_know_it_worked at most 3, each an observable figure and when to look. \
downside_if_wrong under 25 words. what_is_left_over under 30 words. owner under 6 words. timing \
under 10 words.

Call submit_decision exactly once."""


SUBMIT_DECISION_TOOL = {
    "name": "submit_decision",
    "description": (
        "Submit the recommended commercial decision for this account. Call this exactly once."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "action_type": {
                "type": "string",
                "enum": [
                    "price_recovery", "mix_recovery", "win_back", "consolidation",
                    "relationship_review", "investigate_cause", "expansion",
                    "no_action", "gather_data",
                ],
                "description": "The kind of commercial play this is. Must match the leak.",
            },
            "recommended_option": {
                "type": "string",
                "description": (
                    "The specific option chosen, in plain words. Use 'none' when no "
                    "intervention is recommended."
                ),
            },
            "headline": {
                "type": "string",
                "description": "One sentence under 20 words a sales director could act on without reading further.",
            },
            "rationale": {
                "type": "string",
                "description": "Why this option, under 60 words, from this account's own figures.",
            },
            "why_not_more_aggressive": {
                "type": "string",
                "description": (
                    "Why the stronger option was rejected, under 30 words. 'Not applicable' if there was none."
                ),
            },
            "why_not_less_aggressive": {
                "type": "string",
                "description": (
                    "Why the weaker option was rejected, under 30 words. 'Not applicable' if there was none."
                ),
            },
            "expected_result": {
                "type": "string",
                "description": (
                    "What the business gets if this is carried out, in one plain sentence with "
                    "the figures from the chosen option — the money per month, what happens to "
                    "the margin rate or the lost line, and whether that returns the account to "
                    "healthy. The reader must not have to work any of this out."
                ),
            },
            "what_to_do": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "3-5 concrete steps someone carries out, in order, each under 25 words and "
                    "starting with a verb. Name the specific products, categories and figures "
                    "from the brief inside the steps — 'build the proposal around PT-Cordless "
                    "Drill 18V (was Rs 45,460/month)', not 'identify which lines were lost'. "
                    "Never a restatement of the target."
                ),
            },
            "talking_points": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "At most 3 things the account manager says in the room, each under 20 words, "
                    "drawn only from this account's own trading history."
                ),
            },
            "check_before_acting": {
                "type": "array",
                "items": {"type": "string"},
                "description": "At most 3 assumptions the transaction data cannot settle. Verify these first.",
            },
            "restores_account_to_healthy": {
                "type": "boolean",
                "description": (
                    "Whether the option you chose returns this account to healthy, taken from "
                    "that option's own restores_account_to_healthy field. Do not judge this "
                    "yourself."
                ),
            },
            "what_is_left_over": {
                "type": "string",
                "description": (
                    "If the chosen option does not restore health, what remains and what would "
                    "close it. 'Nothing — this restores the account' when it does."
                ),
            },
            "downside_if_wrong": {"type": "string", "description": "Under 25 words."},
            "owner": {"type": "string", "description": "Who should carry this out, under 6 words."},
            "timing": {"type": "string", "description": "When, under 10 words."},
            "how_we_will_know_it_worked": {
                "type": "array",
                "items": {"type": "string"},
                "description": "At most 3 observable changes, each a figure and when to check it.",
            },
            "review_in_months": {"type": "integer"},
        },
        "required": [
            "action_type", "recommended_option", "headline", "rationale",
            "why_not_more_aggressive", "why_not_less_aggressive", "expected_result",
            "what_to_do",
            "talking_points",
            "check_before_acting", "restores_account_to_healthy", "what_is_left_over",
            "downside_if_wrong", "owner", "timing",
            "how_we_will_know_it_worked", "review_in_months",
        ],
        "additionalProperties": False,
    },
    "strict": True,
}


class DecisionError(Exception):
    """Raised when the model does not return a submit_decision call."""


# A reasoning model spends tokens thinking before it emits the tool call, and
# that thinking is charged to the same budget — sizing this from the length
# of the visible answer truncates mid-JSON, which the provider returns as an
# unparseable-arguments error rather than as a short reply.
#
# Capped rather than merely generous, because providers charge the REQUESTED
# ceiling against a rate limit, not the tokens actually used. At 6000 this
# call billed ~9,155 tokens against Groq's 8,000-per-minute free tier and so
# could never succeed there, however long the caller waited — a failure that
# looks like congestion and is really a request that does not fit.
#
# 4000 held until the app moved to claude-sonnet-5, which thinks before it
# answers: on the account with the most moving parts (ACC-112 — eight
# declined lines across seven categories, orders fragmenting) it spent 3,711
# tokens thinking and the tool call was cut off mid-JSON, twice. The visible
# answer is ~900 tokens, so 8000 leaves room for a long think and a full
# answer. The Groq path is no longer used by the app (validate_answer_key.py
# and the tests only), so its free-tier ceiling no longer sets this number.
OUTPUT_TOKENS = 8000


def recommend(client, decision_input: dict, model: str = DEFAULT_MODEL) -> dict:
    """One model call. Returns the parsed submit_decision input dict.

    No drill-down tools: the adviser reasons over a finished report and a
    priced set of options, and a route back into the raw data would let it
    reach a figure the account's own verdict never saw.
    """
    response = client.messages.create(
        model=model,
        max_tokens=OUTPUT_TOKENS,
        system=SYSTEM_PROMPT,
        tools=[SUBMIT_DECISION_TOOL],
        messages=[{"role": "user", "content": json.dumps(decision_input)}],
    )

    submit_block = next(
        (b for b in response.content if b.type == "tool_use" and b.name == "submit_decision"),
        None,
    )
    if submit_block is None:
        raise DecisionError(
            f"Model stopped (stop_reason={response.stop_reason!r}) without calling submit_decision."
        )

    # A response cut off at the token ceiling still carries a tool_use block —
    # the SDK parses whatever JSON had arrived — with the tail of the fields
    # missing. Accepting it would cache a recommendation with no steps, no
    # owner and no timing, which is exactly what happened on ACC-112. Refuse
    # it here so the caller sees an error rather than saving a half answer.
    if response.stop_reason == "max_tokens":
        raise DecisionError(
            "Model ran out of room mid-answer (stop_reason='max_tokens'); "
            "the recommendation is incomplete and was not accepted."
        )
    schema = SUBMIT_DECISION_TOOL["input_schema"]
    missing = [field for field in schema["required"] if field not in submit_block.input]
    if missing:
        raise DecisionError(
            f"submit_decision is missing required fields: {', '.join(missing)}."
        )
    # The steps once arrived as a single garbled string instead of a list;
    # the page iterates the field, so it would have printed one character
    # per step. A list field that is not a list of strings is a bad answer.
    malformed = [
        field for field, spec in schema["properties"].items()
        if spec.get("type") == "array"
        and not (isinstance(submit_block.input.get(field), list)
                 and all(isinstance(item, str) for item in submit_block.input[field]))
    ]
    if malformed:
        raise DecisionError(
            f"submit_decision fields must be lists of strings: {', '.join(malformed)}."
        )
    if not submit_block.input["what_to_do"]:
        raise DecisionError("submit_decision has no steps in what_to_do.")
    return submit_block.input
