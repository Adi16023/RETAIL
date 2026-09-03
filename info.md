Stage 2: Baseline Engine — establishing "normal" for this account
Everything in Stage 2 hinges on one split: history minus the trailing 6 months = "baseline", and the trailing 6 months = "recent" (RECENT_MONTHS = 6). Why 6, not 3: we originally used 3, but verification showed a genuinely healthy account swinging -59% purely from random order-count noise — 3 months wasn't enough orders to average out, so it got widened.

Stage 2 produces three attribute groups, all in build_baseline():

1. overall_revenue — the account's total revenue trend

baseline_monthly_median: the median (not average) of monthly revenue in the baseline window. Median specifically because a single freak month (e.g. one huge historical bulk order) can't drag a median the way it drags a mean — this is what stops the "one-off bulk order" trap from creating a false baseline.
recent_monthly_rate: total recent-window revenue ÷ number of recent months — a rate, not a median, because the recent window is short (6 months) and a rate uses every order in the window rather than throwing away information by taking a middle value.
pct_change: recent vs. baseline, simple percentage.
monthly_revenue_coefficient_of_variation: how naturally noisy this specific account is (std dev ÷ mean of its monthly revenue). This exists so Stage 4 can tell "this account is just volatile by nature" apart from "this is a real shift" — without it, a naturally spiky account would look alarming even with nothing wrong.
change_point: does the total revenue itself show a genuine sustained shift? (mechanism explained under Stage 3 below, since it's the same algorithm.)
2. category_mix — this is where "high-value" gets defined, and it's fully empirical

For the baseline window and recent window separately, computes each category's share of total revenue (not absolute ₹, share — so it's comparable across accounts of any size).
Then ranks categories by baseline share, descending, and keeps adding categories to a "high-value" list until their cumulative share crosses 70% (HIGH_VALUE_CUMULATIVE_SHARE = 0.70). Everything else is "low-value."
This is the mechanism behind what I told you earlier — there's no hardcoded list of "premium categories" anywhere in this code. It's derived fresh, per account, from that account's own spending history. Two different accounts could have completely different categories labeled "high-value."
3. order_behavior — how orders themselves are shaped, independent of category

order_frequency: orders per month, baseline vs recent.
aov (average order value): median order revenue, baseline vs recent.
basket_width: median number of distinct products per order, baseline vs recent.
This is what catches the order-size-shrinkage archetype — a category can look totally fine while orders are quietly getting smaller (fewer items each time), and this is the only place that signal lives.
Stage 3: Change Detector — the actual change-point algorithm
This is the core piece, shared between total-revenue and per-category detection via changepoint.py. It's a real algorithm, not a threshold check on two numbers — here's exactly what it does, step by step, for a single revenue series (say, one category's monthly revenue):

Step 1 — Winsorize. Cap every value at 1.5× the series' own median (WINSOR_CAP_MULTIPLE = 1.5). This exists specifically to stop a single freak month (a one-off bulk order) from faking a change-point — without it, that one huge month gets smeared across neighboring months in the next step and looks like a sustained level, not a blip.

Step 2 — Smooth. Apply a trailing 3-month rolling sum (ROLLING_WINDOW = 3). This pools enough orders together that ordinary month-to-month noise gets averaged out, rather than every random dip looking like a "decline."

Step 3 — Scan for the best split point. This is a brute-force search: try every possible point in the series (as long as at least 5 months sit on each side — MIN_SEGMENT_MONTHS = 5, so a couple of noisy months can't dominate a tiny segment), and for each candidate split, compute the median of the "before" segment and the median of the "after" segment. Whichever split point produces the biggest percentage drop is the winner.

Step 4 — Require the drop to be real. The winning split only counts as a genuine change-point if the decline is ≥35% (DECLINE_THRESHOLD = 0.35) and the "before" segment wasn't already negligible (MIN_BASELINE_MONTHLY_REVENUE = 50, filters out categories too tiny to matter at all).

Step 5 — Check if it recovered. Look at everything after the change-point: did revenue ever climb back to ≥75% of the pre-change level (RECOVERY_FRACTION = 0.75)? If yes, recovered: true — this is the single fact that most separates "temporary blip" from "structural loss" downstream. (There's a subtle fix here: the rolling sum from step 2 blends a couple of pre-change months into the first points right after the split, which can make even a permanent collapse-to-zero briefly look "recovered" — so the recovery check specifically skips those blended boundary points.)

Step 6 — Seasonal precedent. For whatever calendar month the change-point falls in, check: did that same calendar month, in earlier years only, also run meaningfully below its peer months (below 65% of the median of all other months, SEASONAL_INDEX_THRESHOLD = 0.65)? This deliberately requires at least 2 prior occurrences of that calendar month before it will ever say "confirmed" — one prior data point isn't a pattern, it's a coincidence, and treating it as confirmation was literally the bug I fixed a few turns back.

Step 7 — Product-level detail. Separately (product_changes), for every individual product: compare its baseline monthly average to its recent monthly average. Zero in recent but nonzero before → "disappeared." Down ≥35% → "declined." This is what lets the report say "these three specific SKUs stopped selling," not just "this category is down."

The one number that gets normalized after the fact: because the scan runs on a rolling 3-month sum, the raw before/after medians it produces are 3-month totals, not monthly rates — normalize_medians_to_monthly_rate() divides them back down before they reach any ₹ calculation downstream. (This is exactly the bug that caused the "132% of revenue at risk" screenshot earlier — worth mentioning if someone asks about the debugging process.)

Everything Stage 3 outputs is a fact, never a verdict — it never says "structural" or "temporary" itself. That judgment is deliberately left to Stage 4, because deciding why something changed requires holistic reasoning across all these facts together, which is exactly the one place we wanted the LLM, not more code.

Stage 4: Investigation & Attribution Agent — the one LLM call
This is the only stage that reasons rather than computes. Here's exactly what happens, mechanically.

What it receives
The entire evidence pack from Stages 1-3 — serialized as JSON — becomes the first message sent to the model:


messages = [{"role": "user", "content": json.dumps(evidence_pack)}]
It never sees the raw transaction rows. Just the structured facts: overall revenue trend, category mix, per-category change-points, product changes, order behavior, data sufficiency.

The system prompt — explicit criteria, not vibes
This is the part that actually encodes judgment. It's not "figure out if there's leakage" — it's a specific rulebook:

What counts as structural: a sustained decline past a change-point, no seasonal excuse, hasn't recovered, has persisted several months, or replacement by cheaper substitutes.
What counts as temporary: seasonal precedent, a single anomalous period, recovered: true, or a one-off historical event.
Category materiality rule: a change-point in a category that's only a small slice of the account's revenue is weak evidence — this was added after live testing showed the model getting fooled by noise in a tiny category (that's the seasonal_dip fix from earlier).
Order-behavior rule: explicitly told to check basket-width/frequency/AOV as their own signal, not just as a side note to category findings — added after the model missed a real basket-shrinkage case entirely.
The defer instruction, called out as the single most important line in the prompt: if history's too short, or signals conflict, or seasonality is plausible but unconfirmable — set defer=true, use low confidence, and name exactly what data would resolve it. Explicitly told a deferred honest answer scores higher than a forced confident one.
The tools — how it can dig deeper
Three read-only functions that call straight back into Stage 2/3 code (never re-implemented, never a separate calculation path):

get_category_monthly_series — raw, unsmoothed monthly numbers for one category, for when the evidence pack's summary isn't enough detail
get_category_seasonal_breakdown — that category's revenue grouped by calendar month across all years, specifically so the model can judge seasonality itself when Stage 3's conservative seasonal_precedent field says "insufficient_history" rather than guess
get_product_changes — product-level detail, optionally filtered to one category, for naming specific SKUs rather than just a category name
The output contract — enforced, not requested
This is important: the final answer isn't parsed out of free-text. There's a submit_verdict tool with a strict JSON schema (strict: True, additionalProperties: False) — the model is forced to produce exactly this shape:


verdict: healthy | leakage_detected | insufficient_data
temporary_or_structural: temporary | structural | not_applicable
confidence: low | medium | high
defer: true/false
attributed_categories: [...]
cited_facts: [...]        ← must reference specific evidence-pack facts
narrative: "..."          ← the stakeholder-facing explanation
recommended_actions: [...]
data_needed_if_deferring: [...]
There's no "hope the model outputs valid JSON" step — the API layer itself rejects anything that doesn't match this schema.

The loop — how a single investigation actually runs

for _ in range(6):  # MAX_ITERATIONS
    response = client.messages.create(...)
    
    if a submit_verdict call is in the response:
        return it — done
    
    if there are other tool calls:
        execute them for real (against real data)
        feed the results back
        continue the loop
    
    if there's no tool call at all:
        raise AgentError — something's wrong, don't guess
So a typical run looks like: model reads evidence pack → maybe calls get_category_seasonal_breakdown once or twice to check something → then calls submit_verdict. Usually 1-3 round trips. It's capped at 6 iterations specifically so a confused model can't loop forever — if it hits that cap without submitting, that's a clear AgentError, not a silent bad answer.

Why it's built this way, specifically
The client is injected, not constructed inside this file — that's the whole reason I could test this against a scripted fake before you had any API key, and swap Groq in without touching this file at all.
Errors are explicit, never swallowed. An unknown tool call, a model that stops without answering, exhausting the iteration cap — all of these raise a named AgentError rather than returning something half-formed. The UI layer catches these and shows the deterministic evidence anyway, with a clear message, instead of crashing.
The prompt has been iterated based on real live failures, not written once and assumed correct — the materiality rule and the order-behavior rule both exist because a real model, on real data, got specific cases wrong, and I traced why and patched the actual instruction that was missing.

Stage 5: Leakage / Impact Engine — turning a verdict into ₹
This stage is short and deliberately dumb — it doesn't investigate anything, it only does arithmetic on facts Stage 3 already computed and Stage 4 already pointed at. Here's exactly what it does, in order.

What it takes as input
Two things: the evidence pack (from Stage 1-3), and attributed_categories — the list of categories Stage 4's verdict named as responsible. If the account is healthy, or the model deferred, that list is empty and this stage produces nothing to size.

Step-by-step, per attributed category
1. Look up that category's change-point fact. It goes back into evidence_pack["category_changes"] — the exact same dict Stage 3 built — and finds the entry for that category.

2. Handle the "attributed but not quantifiable" case explicitly. If the model named a category that Stage 3 never actually detected a change-point for (this can genuinely happen — e.g. the model attributes leakage based on an order-behavior signal, not a category collapse), this stage doesn't crash and doesn't fake a number. It records:


quantifiable: false, reason: "no change-point detected for this category in the evidence pack"
This is a real, tested code path (I built a test specifically for it) — it exists because the LLM's reasoning and the deterministic detector don't always land on exactly the same category, and hiding that mismatch would be worse than surfacing it.

3. If it is quantifiable, compute four numbers, all straight subtraction/multiplication on Stage 3's numbers:

monthly_revenue_at_risk = before_monthly_median − after_monthly_median — the category's change-point fact, directly. This is the core number everything else derives from.
historical_value_lost_to_date = monthly_revenue_at_risk × sustained_months_since_change_point — i.e., "how much has actually already been lost since this started." Uses Stage 3's own count of how many months have stayed down.
annualized_run_rate_loss = monthly_revenue_at_risk × 12 — "if this exact monthly loss rate continued for a year."
category_baseline_revenue_share — pulled straight from Stage 2's category_mix.baseline_share, carried along so downstream (Stage 6, and the report) knows how big a deal this category actually was to the account, not just the ₹ figure in isolation.
No new logic here — every one of these four numbers is either a direct copy from Stage 3, or one arithmetic operation on Stage 3 numbers. That's the whole design principle: the LLM decided which categories matter, this stage just does the multiplication.

Rolling it up to the account level
total_monthly_revenue_at_risk — sum across all quantifiable attributed categories
total_historical_value_lost — same, summed
pct_of_baseline_monthly_revenue_at_risk = total at-risk ÷ the account's overall baseline monthly revenue (from Stage 2) — this turns an absolute rupee number into a relative one, which is what Stage 6 actually uses to decide priority (a ₹50K/month loss means something very different for a ₹5L/month account vs. a ₹50L/month one).
Why this bug mattered when it happened
This is the exact stage where the earlier 3x scaling bug lived — Stage 3's before_monthly_median/after_monthly_median were briefly on the wrong scale (a 3-month rolling sum instead of a true monthly rate), and since Stage 5 just does direct subtraction on those numbers with zero validation of its own, the error flowed straight through into every ₹ figure without any of this stage's own logic being wrong. That's actually the intended design working correctly, in a sense — Stage 5 has no independent judgment to catch a bad upstream number, on purpose, because it's not supposed to second-guess Stage 3. The fix belonged upstream, in changepoint.py, not here — which is exactly where I made it.


Stage 6: Prioritisation Engine — "where should sales act, and how urgently"
No LLM call here either — this is a plain rule table. It takes Stage 5's ₹ numbers and Stage 4's confidence, and turns them into one of four buckets: High / Medium / Low / Deferred / None.

First — two early exits, before any math happens

if verdict.defer is true, or verdict == "insufficient_data":
    → priority = "Deferred"

if verdict == "healthy" (or nothing was attributed):
    → priority = "None"
These are checked before touching the ₹ numbers at all, because a deferred or healthy verdict has no meaningful magnitude to rank — there's nothing to prioritize.

The actual ranking — a 2D matrix, not a single score
This was a deliberate design choice over a black-box weighted formula. Two independent tiers get computed, then looked up in a table:

Magnitude tier — how big is the ₹ impact, as a share of the account's baseline revenue (not an absolute rupee number, since ₹50K/month means something completely different for a small account vs. a large one):


≥15% of baseline revenue at risk  → "high"
≥5%                                → "medium"
below that                         → "low"
Confidence tier — taken straight from Stage 4's verdict: low / medium / high.

Then the two get crossed in a 3×3 table:

conf: high	conf: medium	conf: low
magnitude: high	High	High	Medium
magnitude: medium	Medium	Medium	Low
magnitude: low	Low	Low	Low
The one cell worth pointing out on stage: high magnitude + low confidence lands on Medium, not High. That's not a rounding artifact — it's a deliberate rule. A big rupee number resting on shaky evidence shouldn't top the priority list, because acting on weak evidence is itself a risk — this is the code-level expression of the same "judgment under uncertainty" theme the whole rubric weights at 30%. A table like this is also something you can defend line-by-line under jury questioning, in a way a single opaque weighted score can't be.

The churn-risk projection
Only computed when there's an actual monthly loss to project:


projected_12_month_loss = monthly_run_rate_loss × 12
Same style as Stage 5 — one multiplication, nothing fancier. But it always ships with an explicit caveat string attached:

"Assumes the current monthly loss rate persists unchanged for 12 months — not a forecast of further deterioration, just the cost of inaction at today's rate."

That sentence is there on purpose. This number is easy to misread as "the AI predicts you'll lose ₹X" — it's not a prediction, it's a cost-of-inaction calculation: if nothing changes and nobody intervenes, here's what today's rate adds up to over a year. Important distinction if a jury member pushes on it — this system isn't forecasting the future, it's quantifying the present trend.

What's notably absent here
There's no LLM step, no "recommendation strength," no extra scoring — this stage is intentionally the plainest one in the whole pipeline. It exists because plan.md's original draft had an LLM call here too ("Prioritisation Engine [LLM + rules]"), and one of the earliest decisions in this whole project was cutting that down — a ranking that's just magnitude-of-impact × confidence doesn't need a model call, it needs a formula, and a formula is cheaper, faster, and fully auditable.


Stage 7: Report Assembly — the simplest stage, on purpose
This one is genuinely just a dict merge. Here's the entire function, no simplification:


def assemble_report(account_id, evidence_pack, verdict, impact, priority):
    return {
        "account_id": account_id,
        "verdict": verdict["verdict"],
        "temporary_or_structural": verdict["temporary_or_structural"],
        "confidence": verdict["confidence"],
        "defer": verdict["defer"],
        "narrative": verdict["narrative"],
        "attributed_categories": verdict["attributed_categories"],
        "cited_evidence": verdict["cited_facts"],
        "recommended_actions": verdict["recommended_actions"],
        "data_needed_if_deferring": verdict["data_needed_if_deferring"],
        "financial_impact": impact,           # ← all of Stage 5's output
        "prioritization": priority,           # ← all of Stage 6's output
        "data_sufficiency": evidence_pack["data_sufficiency"],  # ← from Stage 1
    }
That's it. It takes what came out of the four preceding stages — the verdict from Stage 4, the ₹ numbers from Stage 5, the priority bucket from Stage 6, and the sufficiency score from Stage 1 — and lays them out into one flat structure that's easy for the UI to render and easy for anyone (jury included) to read top to bottom.

Why it's still its own stage, even though it's trivial
Two reasons, both deliberate:

No new LLM call. This is the point I keep coming back to across every stage — the narrative text was written once, by Stage 4, when it had the full evidence pack in front of it. This stage doesn't ask the model to "write a summary" again. If it did, you'd risk a second call drifting slightly from what the first call actually found — two narratives that don't perfectly agree with each other. Keeping it to one narrative, generated once and just carried forward, means what you read in the final report is guaranteed to match what the model actually reasoned about, not a re-paraphrase of it.

Single, traceable source per field. Every field in the final report has exactly one place it came from — verdict fields from Stage 4, financial_impact from Stage 5, prioritization from Stage 6, data_sufficiency from Stage 1. If a jury member points at any number in the final output and asks "where did that come from," the answer is always one specific stage, never "the LLM summarized it."

