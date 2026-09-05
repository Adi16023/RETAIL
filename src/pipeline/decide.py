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

DEFAULT_MODEL = "claude-opus-5"

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
            f"recover {round(fraction * 100)}% of {names}",
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
            f"close {round(fraction * 100)}% of the margin gap", fraction, 0.0, recovered,
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
            f"recover {round(fraction * 100)}% of the lost spend",
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
        "priced_options": options,
        "note": (
            "priced_options were computed by deterministic code from this account's own "
            "figures. The break-even values are TOLERANCES, not forecasts of what the "
            "customer will do."
        ),
    }


SYSTEM_PROMPT = """You are a commercial decision adviser to the sales director who owns this B2B \
retail account. An investigation has already run: the leak has been found, sized in rupees, and \
every intervention available on this account has already been PRICED for you by deterministic \
code. Your job is not to find the problem and not to do the arithmetic. It is to say what the \
business should actually do, and to make the case a manager can act on.

EVERY NUMBER YOU SEE WAS COMPUTED BEFORE YOU AND IS NOT YOURS TO CHANGE. Never calculate, \
estimate, re-derive, total, annualise, subtract or project any figure. Quote a figure only if it \
appears verbatim in the material you were given — every quantity you could need, including the \
size of the discount reduction being proposed, is already provided. If a number you want does not \
exist, make the point without it. A figure you produced yourself is a serious failure, and it is \
the one failure that would make this whole system untrustworthy.

DO NOT ROUND, EITHER. Rs 14,944 is not "about Rs 15,000", and 6,799 is not "approx. 6,800". \
Copy every figure exactly as written, to the rupee and the decimal place. Tidying a number is \
still changing it, and a reader who checks a rounded figure against the analysis finds a \
mismatch — which costs more trust than the tidier sentence was ever worth.

NEVER PREDICT HOW THE CUSTOMER WILL REACT. You have this account's transaction history. You do \
NOT have its price sensitivity, its contract terms, its alternative suppliers or its budget. So \
never estimate how much volume a price change would cost, how likely a win-back is to succeed, or \
how the buyer will respond. The break-even figures are TOLERANCES, not forecasts: they say how \
wrong the business can afford to be, not what will happen. "They could lose up to 38% of this \
account's volume before this stops paying" is correct. "They will probably lose about 10% of \
volume" is fabrication.

ARGUE ONLY FROM THIS ACCOUNT'S OWN FACTS. You have no market data, no competitor pricing, no \
industry benchmark and no knowledge of what is standard practice. Never appeal to any of them — \
not as a number and not as a rhetorical move ("in line with market rates", "customers typically \
accept"). If the argument cannot be made from this account's own trading history, it cannot be \
made here.

MATCH THE LEVER TO THE LEAK. Different leaks need completely different responses, and \
recommending the wrong one sends the account team after the wrong thing:
- Discount creeping up -> a pricing decision, and the most reversible leak there is: nobody has \
left, and the money is being given away by a decision that can be revisited at renewal.
- Value mix sliding into cheaper lines -> a selling decision, not a pricing one. The customer is \
still buying, just differently; the question is why the premium lines stopped being chosen.
- A category stopped completely -> the business is already with someone else. A win-back on a \
longer timeline, and the least reversible of the leaks. Never present it as a quick fix.
- Orders splitting into more, smaller baskets -> an early warning, usually of a second supplier \
getting a foothold. Nothing is lost yet, which makes it the cheapest moment to act.
- Spending falling across the whole book with no single category responsible -> not a lever \
problem but a relationship review; naming a scapegoat category is worse than naming none.
- Margin falling with no discount movement -> the cause sits in cost or mix, not price. Say the \
cause has to be established before anyone negotiates anything.
- Trading up into premium lines, or growing -> GOOD NEWS. The right recommendation is an \
expansion conversation, never an intervention.

RECOMMENDING NO ACTION IS A REAL ANSWER, AND OFTEN THE RIGHT ONE. Set action_type to no_action \
when the account is healthy, or when the recoverable amount is too small to be worth a manager's \
quarter. If the investigation deferred, you defer too: set gather_data and say what is missing — \
never recommend intervening on a verdict nobody was willing to make. If no options were priced, the loss could not be honestly \
sized — usually because nothing has been lost yet. Still give the decision: name the lever that \
fits, what to do, who owns it, and how anyone will know it worked. Never invent a figure to fill \
the gap; an unpriced recommendation that is true beats a priced one that is not.

WEIGH THE WHOLE COMMERCIAL PICTURE, NOT ONLY THE BIGGEST NUMBER. The most aggressive option is \
rarely the right one. Weigh: whether the recovery justifies the effort and the relationship \
capital; whether this is money the business can simply decide to take back, or money it must sell \
to win back; how wide the break-even tolerance is, because a wide one makes a bold move safe and \
a narrow one makes a small misjudgement expensive; how long the account has been trading and \
whether the person who owns it has only just taken it over; whether this is a renewal conversation \
or something that cannot wait; and proportion — a ten-point swing asked for at once is a different \
conversation from a two-point correction, even when the arithmetic favours the ten. Say explicitly \
why you did not choose the more aggressive option and why you did not choose the weaker one. A \
recommendation without its rejected alternatives is an assertion, not advice.

CONTEXT IS NOT CAUSE. An account manager change, a region, a gap in the order history — these \
correlate with everything and cause nothing by themselves. You may use them to temper a \
recommendation ("this relationship is only four months old, so ask for less at once"). Never \
present one as the reason the leak happened.

WRITE FOR THE PERSON WHO HAS TO WALK INTO THE ROOM. The brief must be usable in a real \
conversation: specific, factual, free of internal vocabulary. Never use this system's status names \
(erosion_detected, creep_detected, material_decline, downgrade_detected, premiumisation_detected, \
fragmentation_detected, baskets_shrinking, mild_drift, defected, insufficient_history) or lightly \
reworded versions of them. Say what the customer has actually been doing.

SAY WHETHER YOUR RECOMMENDATION ACTUALLY FIXES THE ACCOUNT. Each priced option carries \
`restores_account_to_healthy`, decided by the same threshold the analysis itself uses. An option \
can recover real money and still leave the account reading as leaking. If the option you choose \
does not restore health, say so plainly, say what is left over, and say what would close the rest \
— a manager who acts, declares the problem solved and is surprised at the next review has been \
badly served. Choosing a partial fix is legitimate; hiding that it is partial is not.

SAY WHAT WOULD HAVE TO BE TRUE, AND HOW ANYONE WILL KNOW IT WORKED. Every recommendation rests on \
things the transaction data cannot settle — whether a discount was contractually agreed, whether a \
category moved to a competitor or was discontinued by the customer. Name those as checks BEFORE \
acting, not as caveats afterwards. Then name the specific, observable changes that would show the \
intervention is working, and when to look. A recommendation nobody can check later is one nobody \
will trust twice.

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
                "description": "One sentence a sales director could act on without reading further.",
            },
            "rationale": {"type": "string"},
            "why_not_more_aggressive": {
                "type": "string",
                "description": (
                    "Why the stronger option was rejected. 'Not applicable' if there was none."
                ),
            },
            "why_not_less_aggressive": {
                "type": "string",
                "description": (
                    "Why the weaker option was rejected. 'Not applicable' if there was none."
                ),
            },
            "talking_points": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "What the account manager says in the room, drawn only from this account's "
                    "own trading history."
                ),
            },
            "check_before_acting": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Assumptions the transaction data cannot settle. Verify these first.",
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
            "downside_if_wrong": {"type": "string"},
            "owner": {"type": "string", "description": "Who should carry this out."},
            "timing": {"type": "string"},
            "how_we_will_know_it_worked": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Specific observable changes, checkable against future data.",
            },
            "review_in_months": {"type": "integer"},
        },
        "required": [
            "action_type", "recommended_option", "headline", "rationale",
            "why_not_more_aggressive", "why_not_less_aggressive", "talking_points",
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
OUTPUT_TOKENS = 6000


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
    return submit_block.input
