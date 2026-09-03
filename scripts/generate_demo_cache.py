"""
Generates the offline demo cache under demo_cache/ — pre-computed reports
for the reference dataset's most instructive accounts, used by app.py's
"View offline demo" mode when live LLM access isn't available (venue has no
internet, no API key, rate limited — see plan.md Phase 4 "live-failure
hardening").

These reports are produced with a SCRIPTED verdict standing in for the LLM
call, not a real model response — clearly labeled as such in the output and
in the UI. They exist so the demo can run end-to-end on stage even if Stage
4 is unreachable; they are not a substitute for the real pipeline when it is
available, and they are never silently swapped in for a failed live run.

The four accounts cached are chosen to cover the four outcomes a judge will
ask about: the disclosed jury scenario, a leak with no revenue signal at
all, the deferral case, and a healthy account that superficially resembles
a leak. Every rupee figure in them is computed by the real Stages 1-3 and
5-7 over the real data — only the Stage 4 judgement is scripted.

Run from the repo root: python scripts/generate_demo_cache.py
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "tests"))

from pipeline.orchestrate import run_for_account

from fakes import ScriptedClient, message, tool_use_block

TRANSACTIONS = REPO_ROOT / "data" / "meridian" / "transactions.csv"
OUT_DIR = REPO_ROOT / "demo_cache"


def scripted_client(verdict: dict) -> ScriptedClient:
    return ScriptedClient([
        message([tool_use_block("submit_verdict", verdict, "toolu_1")], stop_reason="tool_use")
    ])


def verdict(**overrides) -> dict:
    base = {
        "verdict": "healthy", "temporary_or_structural": "not_applicable",
        "confidence": "high", "defer": False, "leak_dimensions": [],
        "attributed_categories": [], "cited_facts": [], "narrative": "",
        "recommended_actions": [], "data_needed_if_deferring": [],
    }
    base.update(overrides)
    return base


CACHED_ACCOUNTS = {
    # The scenario the challenge doc describes: flat topline, value quietly
    # rotating out of the high-value tier.
    "acc101_hidden_mix_collapse": ("ACC-101", verdict(
        verdict="leakage_detected", temporary_or_structural="structural",
        confidence="high", leak_dimensions=["margin", "tier_mix"],
        cited_facts=[
            "Revenue is flat (+1.3% half over half) with no change-point at total level",
            "Margin rate fell 32.2% -> 19.9% (-12.3pp) over the same period",
            "High-tier share of revenue fell 66.2% -> 22.5% (-43.8pp); Low tier backfilled it",
        ],
        narrative=(
            "This account looks healthy at the top line — revenue is essentially flat. But the "
            "mix underneath it has inverted: high-value lines that were two thirds of this "
            "account's business are now under a quarter of it, and low-value volume has been "
            "backfilled to hold the topline up. Margin has fallen by more than twelve points as "
            "a result. This is real value loss hidden behind a stable revenue number."
        ),
        recommended_actions=[
            "Have the account owner ask specifically about the high-value lines — find out whether they moved to a competitor or to another budget line.",
            "Treat this as a retention priority despite the stable topline; revenue will follow the mix eventually.",
        ],
    )),
    # The margin-vs-revenue test: nothing moves except the price.
    "acc107_discount_creep": ("ACC-107", verdict(
        verdict="leakage_detected", temporary_or_structural="structural",
        confidence="high", leak_dimensions=["discount", "margin"],
        cited_facts=[
            "Average discount rose 12.5% -> 23.4% (+10.8pp) between baseline and recent windows",
            "Margin rate fell 22.6% -> 11.5% (-11.1pp)",
            "Revenue is flat (-0.9%) and tier mix is stable — volume and mix are unchanged",
        ],
        narrative=(
            "Volume and product mix on this account are unchanged, and revenue is flat. The "
            "money is leaving through the price: average discount has roughly doubled, and "
            "margin has halved with it. A revenue-only review would show nothing wrong here."
        ),
        recommended_actions=[
            "Review the discount authority applied to this account over the last two quarters.",
            "Re-baseline pricing at renewal; the current discount level was never a deliberate decision.",
        ],
    )),
    # The rubric's heaviest line: not enough history to call it.
    "acc109_defer_thin_history": ("ACC-109", verdict(
        verdict="insufficient_data", temporary_or_structural="not_applicable",
        confidence="low", defer=True,
        cited_facts=[
            "Only 4.4 months of history (7 orders); sufficiency label is 'insufficient'",
            "Flags: history_too_short_for_baseline, cannot_check_prior_year_seasonality",
            "No baseline window exists, so margin, discount and mix comparisons all return insufficient_history",
        ],
        narrative=(
            "This account was onboarded recently and has under five months of history. The "
            "recent softness may be a genuine decline, an onboarding ramp, or ordinary noise — "
            "with no prior year to compare against, those cannot be told apart. Deferring to a "
            "human rather than guessing."
        ),
        recommended_actions=["Route to the account owner for context on the onboarding ramp."],
        data_needed_if_deferring=[
            "At least 12 months of order history, to establish a baseline and permit a prior-year seasonality check",
            "Any pre-onboarding purchasing history, if this account was migrated from another system",
            "Confirmation from the account owner of the expected ramp profile for a new account",
        ],
    )),
    # Looks exactly like the leak above at revenue level, and is the opposite.
    "acc111_premiumisation_not_leakage": ("ACC-111", verdict(
        verdict="healthy", temporary_or_structural="not_applicable",
        confidence="high",
        cited_facts=[
            "High-tier share of revenue rose 52.8% -> 64.6% (+11.7pp)",
            "Margin rate improved 29.7% -> 32.3% (+2.6pp)",
            "Revenue up 8.2%; no change-point and no dip episodes",
        ],
        narrative=(
            "The mix on this account has shifted substantially — but upward. It is buying fewer "
            "units and trading up to premium lines, so revenue is flat-to-up and margin is "
            "better than it was. A mix shift is not automatically a leak; this one is good news "
            "and should be read as an expansion signal, not a retention risk."
        ),
        recommended_actions=["Surface as an expansion opportunity in the premium range."],
    )),
}


def main():
    if not TRANSACTIONS.exists():
        raise SystemExit(
            f"Reference transactions not found at {TRANSACTIONS}. "
            "Run `python scripts/prepare_dataset.py` first."
        )

    OUT_DIR.mkdir(exist_ok=True)
    # Clear stale entries so a renamed or dropped account cannot linger in
    # the on-stage fallback.
    for stale in OUT_DIR.glob("*.json"):
        stale.unlink()

    for name, (account_id, scripted_verdict) in CACHED_ACCOUNTS.items():
        report = run_for_account(scripted_client(scripted_verdict), str(TRANSACTIONS), account_id)
        report["_cache_metadata"] = {
            "archetype": name,
            "account_id": account_id,
            "source": "scripted demo verdict, not a real LLM response",
        }
        path = OUT_DIR / f"{name}.json"
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Wrote {path.relative_to(REPO_ROOT)}  ({account_id}, {report['prioritization']['priority']})")


if __name__ == "__main__":
    main()
