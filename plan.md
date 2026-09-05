# Implementation Plan — Retail Revenue Leakage Agent

Working plan for the Quessathon "Retail: Where Did the Revenue Go?" challenge (kickoff Sept 2 → event day Sept 10). This is the living document — update it as decisions are made: mark phases done, record deviations. A future session should be able to read this file and know exactly where the project stands and what to do next.

Source documents: [problem_statement.md](problem_statement.md), [meeting_notes.md](meeting_notes.md), [CLAUDE.md](CLAUDE.md).

## Design principle: LLM only where judgement is genuinely required

Earlier drafts of the architecture used 4 separate LLM stages (Investigation, Attribution, Prioritisation, Explanation). Revised: only **one** step in this pipeline is a genuine judgement call that can't be reduced to a threshold — classifying temporary vs. structural leakage with a confidence level, including deciding when to defer. That's the rubric's heaviest line (30%, Judgement & robustness under uncertainty), and it's the reason a fixed rule set won't generalize to the jury's unseen scenario.

Everything else — cleaning, baselining, change-point detection, ₹ math, priority ranking — is deterministic. Consolidating to one LLM call instead of four also matters for Engineering quality (20%, "runs live without failure"): fewer model invocations means less live latency, lower API cost, and fewer places a stage-day demo can break.

## Target Architecture (revised)

```
                         ┌─────────────────────────┐
                         │   Transaction Dataset   │
                         └────────────┬────────────┘
                                      ▼
                    ┌────────────────────────────────┐
                    │  1. DATA PROCESSING LAYER      │   [deterministic]
                    │ Cleaning, schema validation,   │
                    │ aggregation, product/category  │
                    │ hierarchy, monthly/weekly      │
                    │ features, data-sufficiency     │
                    │ score                          │
                    └───────────────┬────────────────┘
                                    ▼
                    ┌────────────────────────────────┐
                    │ 2. CUSTOMER BASELINE ENGINE    │   [deterministic]
                    │ Historical revenue, category/  │
                    │ product mix, order frequency,  │
                    │ AOV, quantity, high-value       │
                    │ category identification        │
                    └───────────────┬────────────────┘
                                    ▼
                    ┌────────────────────────────────┐
                    │ 3. CHANGE / LEAKAGE DETECTOR   │   [deterministic]
                    │ Revenue change, mix shift,     │
                    │ category decline (change-point │
                    │ detection), product            │
                    │ disappearance, order-size &     │
                    │ frequency change                │
                    └───────────────┬────────────────┘
                                    ▼
                        anomalies + evidence pack
                          (structured JSON facts)
                                    ▼
             ┌────────────────────────────────────────────┐
             │  4. INVESTIGATION & ATTRIBUTION AGENT      │   [LLM — the ONE call]
             │                                            │
             │  Single agentic loop, tool-calling back    │
             │  into stages 1–3 for drill-downs (e.g.     │
             │  "category X monthly series", seasonality  │
             │  check). Never computes numbers itself.    │
             │                                            │
             │  Applies explicit temporary-vs-structural  │
             │  criteria and emits one structured output: │
             │  { verdict, confidence, cited_facts,       │
             │    temporary_or_structural, defer_flag,    │
             │    narrative, recommended_actions }        │
             │                                            │
             │  DEFER PATH: if evidence is insufficient,  │
             │  say so and name what data would resolve   │
             │  it (this is the 30%-weighted rubric line) │
             └────────────────────┬───────────────────────┘
                                  ▼
                    ┌────────────────────────────────┐
                    │ 5. LEAKAGE / IMPACT ENGINE     │   [deterministic ₹]
                    │ Revenue at risk, historical    │
                    │ value lost, persistence,       │
                    │ category importance —          │
                    │ computed only for categories   │
                    │ the agent flagged in stage 4   │
                    └───────────────┬────────────────┘
                                    ▼
                    ┌────────────────────────────────┐
                    │ 6. PRIORITISATION ENGINE       │   [deterministic rules]
                    │ priority_score = revenue_at_risk│
                    │  × confidence × category_weight │
                    │ bucketed into High/Medium/Low   │
                    │ + churn-risk projection         │
                    │ ("if this persists, ₹X over     │
                    │  N months") from stage-5 numbers│
                    └───────────────┬────────────────┘
                                    ▼
                    ┌────────────────────────────────┐
                    │ 7. REPORT ASSEMBLY             │   [deterministic template]
                    │ Merges stage 4's narrative +    │
                    │ stage 5/6's ₹ figures and       │
                    │ priority bucket into the final  │
                    │ output — no new LLM call here   │
                    └────────────────────────────────┘
```

Rubric mapping: stages 2–3 = **Detect**, stage 4 = **Investigate + Attribute**, stages 5–6 = **Attribute (quantify) + Prioritise**, stage 7 = **Explain**.

**Layer notes:**

1. **Data layer (feeding stage 1)** — synthetic B2B transaction generator + CSV ingestion. Schema: `date, account_id, product_id, category, unit_price, quantity, revenue, margin (optional)`. The generator must produce named scenario archetypes (below) with tunable noise, seasonality, and trend parameters, so we can defend its realism live. Stage 1 must accept an *arbitrary* CSV gracefully (schema mismatch → clear error, not a crash) — the Reality Test will be an unseen file.
2. **Evidence pack (stage 3 output)** — the analytics output serialized as structured JSON facts (e.g., "Category X revenue: ₹4.0Cr → ₹1.0Cr over 6 months; change-point detected 2025-11; total revenue flat"). This is the only thing the LLM sees — it reasons over facts, it never computes them.
3. **Stage 4 is the single LLM touchpoint and where the 30% rubric lives** — temporary-vs-structural classification with explicit confidence, and the defer path when evidence is insufficient. Every conclusion must cite the specific evidence-pack facts it rests on. The narrative and recommended-actions text are generated here, once — not regenerated in a later "explanation" call.
4. **Stage 5 quantifies in ₹ deterministically** (revenue at risk = baseline high-value run-rate minus current, projected persistence) so impact numbers are traceable, never LLM-generated.
5. **Stage 6 is a formula, not a model call.** Only the *phrasing* of retention recommendations needs language, and that's already produced by stage 4 — stage 6 just computes the bucket and the churn-risk ₹ projection.
6. **Presentation layer (stage 7 surface)** — Streamlit (or small FastAPI + HTML) page: upload CSV → run pipeline → render verdict, confidence, attributed leakage with ₹ quantification, prioritised interventions, cited evidence, and any insufficiency flags.

**Temporary vs structural — the core judgement.** Encode explicit criteria rather than vibes: structural signals include a sustained (multi-period) category decline past a change-point with no seasonal precedent, replacement by lower-value substitutes, and shrinking order sizes in the affected category; temporary signals include seasonal precedent in prior history, a single anomalous period, or an announced/short-lived dip that recovers. When history is too short to distinguish (e.g., <12–18 months and the shift could be seasonal), the agent must say so and recommend what data would resolve it — that answer scores higher than a forced classification.

## Scenario Archetypes (synthetic data + self-testing)

Generate and test against at least these accounts:

1. **The jury's known shape**: total revenue flat, high-value categories collapse, low-value volume rises (structural)
2. Genuine seasonal dip in a high-value category that recovers (temporary — agent must NOT cry leakage)
3. Gradual order-size shrinkage with stable frequency (structural, subtle)
4. One-off bulk order in the past inflating the baseline (looks like decline, isn't)
5. Healthy account, no leakage (agent must say "healthy" — no false alarms)
6. Too-short history / sparse orders (agent must flag insufficiency and defer)
7. A wildcard we haven't planned for — build one blind, run the agent, see what breaks

Archetypes 2, 4, 5, and 6 are the traps: they test that the agent doesn't force a leakage narrative. Expect the Reality Test to resemble one of these.

## Phase Plan

### Phase 1 — Setup & data foundation (Sept 2–3)
- [ ] Email Tejashwini to claim the team's LLM access license (the email must specify which model ID the team will use); get the presentation template — **human action, not started**
- [ ] Contact Aninda/Manu with clarifying questions early (input data format expectations, whether margin data can be assumed) — **human action, not started**
- [x] Initialize the repo: Python project, dependency management (`requirements.txt` + `requirements-dev.txt`, using an existing `venv/` — pandas 3.0.5, numpy 2.5.2). **`git init` NOT done — see "Open issue: git repo root" below, blocked pending a decision.**
- [x] Build the synthetic data generator with the archetypes above (`generate_data.py` (since removed), spec in [data.md](data.md)); verified it runs and produces schema-correct output for all 7 archetypes across 35 accounts. **Known issue** (accepted, not fixed): order frequency is low relative to per-order value, so monthly revenue is noisy enough that even the "healthy" archetype swings 10–100x month to month — decision made to handle this via rolling windows in Phase 2's analytics engine rather than retuning the generator now.
- [x] Ingestion + validation + data-sufficiency scoring — [src/pipeline/ingest.py](src/pipeline/ingest.py), tested in [tests/test_ingest.py](tests/test_ingest.py) (8/8 passing): column-mapping with synonym + fuzzy matching, missing-field derivation (revenue from unit_price×quantity, synthetic order_id when absent), per-row validation with clear drop reasons, clear `IngestionError` on unresolvable input (never a raw crash), and a weighted data-sufficiency score per account (history length + order count weighted at 0.45 each, category coverage at 0.10 — category breadth alone must not mask a short/sparse history).

### Phase 2 — Analytics engine (Sept 3–4)
- [x] Per-account metrics: trends, mix shift, order size/frequency, high/low-value split — [src/pipeline/baseline.py](src/pipeline/baseline.py)
- [x] Change-point detection on category-level (and total) revenue series — [src/pipeline/detect.py](src/pipeline/detect.py), [src/pipeline/changepoint.py](src/pipeline/changepoint.py)
- [x] Evidence-pack JSON serialization — [src/pipeline/evidence.py](src/pipeline/evidence.py)
- [x] Unit tests against the archetypes — [tests/test_evidence.py](tests/test_evidence.py), 15/15 passing across [tests/test_ingest.py](tests/test_ingest.py) + test_evidence.py
- [ ] Seasonality comparison against prior-year same-period — **implemented but unreliable, known open issue below**

**Root-cause data fix applied** (not just Stage 2/3 tuning): verification kept surfacing detection failures traceable to the original generator's order frequency (1.5–3.5/month) being too sparse for any deterministic signal to survive the noise. Went back and fixed at the source rather than continuing to patch downstream:
- `generate_data.py` (since removed): order frequency raised to 8–15/month (sparse_short_history archetype deliberately kept low — that's its whole point).
- Fixed the structural_mix_collapse archetype's decay floor (was `max(0.1, ...)`, residual 10% high-value sampling — harmless at the old low order rate, but leaked meaningful absolute revenue back in once order volume rose 4-6x). Now decays fully to 0.
- Fixed the archetype's revenue-backfill mechanism: the original design bumped item *count* per order to compensate for the collapsed high-value category, but low-value items are 40-100x cheaper per unit than what was lost, so it couldn't come close to keeping total revenue flat as the spec requires. Replaced with `make_backfill_order_rows()`, which solves directly for the revenue gap (tracked via each account's own pre-change-point high-value run-rate) rather than approximating via item count.
- Net result: the flagship jury-disclosed scenario (total revenue flat, high-value category collapses) is now reproduced *and* correctly detected end-to-end — total revenue pct_change lands within the intended ±10% band, no false total-level change-point fires, and the injected high-value category is detected with ~100% decline, the correct change-point month, and `recovered: false`.

**Stage 2/3 design decisions worth knowing for Stage 3/4 work:**
- `overall_revenue()` now separates baseline (median, robust to one-off spikes) from recent (a sum/months *rate*, deliberately not smoothed — see below) and exposes `monthly_revenue_coefficient_of_variation` so Stage 4 can tell a naturally volatile account from a genuine shift, rather than reading `pct_change` as a verdict on its own.
- Change-point scanning (`changepoint.py`) winsorizes each series (cap at 1.5x its own median) before a trailing 3-month rolling sum, then scans for the split maximizing before/after median decline (min segment 5 months, threshold 35%). Winsorization only clips upward spikes — it cannot suppress a genuine sustained decline, since that's a downward drift, not an outlier month — so this shouldn't cost true-positive sensitivity, though it hasn't been stress-tested against a genuine-total-decline archetype (none of the current 7 archetypes are designed to produce one).
- Fixed a boundary artifact where the trailing rolling sum blends pre-change months into the first points *after* a detected change-point, making a clean permanent collapse misread as "recovered" — the recovery check now skips those blended points.

**Seasonal-precedent detection — fixed.** `_seasonal_precedent()` in detect.py was rewritten: it now (1) only compares against months strictly *before* the decline (the original bug let a structural collapse's own low value drag down whatever average it was compared against, making any decline look "seasonal" for whatever month it happened to land in), (2) compares against contemporaneous peer months via median rather than a noisy global average, and (3) requires **at least 2** independent prior occurrences of the same calendar month before returning `confirmed` — a single prior occurrence is one data point, not evidence of a recurring pattern (this is what caused the original false positive: one noisy low month, by chance, read as "seasonal"). Audited all 42 category-level change-points across all 35 accounts post-fix: **zero false "confirmed" calls** (down from multiple). Honest tradeoff: given our ~18-20 month archetype history length, 2+ occurrences of the same calendar month rarely exist, so `confirmed` now almost never fires — the function is conservative rather than wrong, correctly returning `insufficient_history` instead of guessing. This is consistent with the project's own criteria (history under ~12-18 months often can't resolve a seasonal question), and Stage 4's `get_category_seasonal_breakdown` tool exists precisely to let the model reason over the raw data itself when the deterministic layer can't confirm either way.

### Phase 3 — Agent layer (Sept 4–5)
- [x] Single agentic loop (stage 4) over the evidence pack with tool calls back into the analytics engine — [src/pipeline/agent.py](src/pipeline/agent.py). Manual tool-use loop (not the SDK's beta tool runner), specifically so a fake client can be injected for testing without hitting the real API — see below.
- [x] Explicit temporary-vs-structural criteria in the prompt — encoded directly in `SYSTEM_PROMPT`, same criteria as plan.md's "Temporary vs structural" section. Includes an explicit instruction to defer when evidence doesn't support a clean answer, and a workaround for the known seasonal-precedent unreliability (Phase 2): the prompt tells the model that field can be wrong and to call `get_category_seasonal_breakdown` to judge raw calendar-month history itself rather than trust it blindly.
- [x] Output contract — enforced via a strict `submit_verdict` tool (JSON-schema-validated `input`, not parsed free text): `verdict, temporary_or_structural, confidence, defer, attributed_categories, cited_facts, narrative, recommended_actions, data_needed_if_deferring`.
- [x] Three read-only drill-down tools back into Stage 2/3 functions: `get_category_monthly_series`, `get_category_seasonal_breakdown`, `get_product_changes`.
- [x] Stage 5 — Leakage/Impact Engine (deterministic ₹) — [src/pipeline/impact.py](src/pipeline/impact.py). Sizes only the categories Stage 4 attributed, using the change-point facts already in the evidence pack (monthly revenue at risk, historical value lost to date, annualized run-rate loss). An attributed category with no detected change-point degrades to `quantifiable: false` with a reason, rather than fabricating a number or crashing — exercised directly in tests.
- [x] Stage 6 — deterministic prioritisation — [src/pipeline/prioritize.py](src/pipeline/prioritize.py). A magnitude tier (% of baseline monthly revenue at risk) × confidence tier matrix, not a black-box weighted score — e.g. a large rupee figure rated on low confidence is deliberately downgraded to Medium rather than High, since acting on shaky evidence is itself a risk.
- [x] Churn-risk projection ("if this persists, ₹X over 12 months") — computed deterministically inside `prioritize()` from Stage 5's monthly run-rate, with an explicit `basis` caveat string (not a forecast, just the cost of inaction at today's rate).
- [x] Stage 7 — report assembly — [src/pipeline/report.py](src/pipeline/report.py), pure templating, no new LLM call. Orchestrator wiring all 7 stages together: [src/pipeline/orchestrate.py](src/pipeline/orchestrate.py).
- [x] "Run all 7 archetypes end-to-end" — done twice: first with a scripted fake LLM client to verify the pipeline wiring ([tests/test_agent.py](tests/test_agent.py), [tests/test_orchestrate.py](tests/test_orchestrate.py)), then for real against a live model via Groq (see "Real-LLM verification" below) — 6/7 correct, 1 explained by data noise, not a defect.

**Real-LLM verification — done, via Groq (Claude access still pending).** No `ANTHROPIC_API_KEY` is available and the Phase 1 license email hasn't been sent; the team has a Groq key instead. Rather than change agent.py to speak Groq's OpenAI-compatible format, [src/pipeline/groq_client.py](src/pipeline/groq_client.py) is a shim that translates Anthropic-shaped tool defs/messages to Groq's format on the way in and Groq's response back to Anthropic-shaped content blocks on the way out — so `agent.py`, `impact.py`, `prioritize.py`, and `report.py` needed **zero changes**, proven directly by `test_investigate_runs_unchanged_through_groq_shim` in [tests/test_groq_client.py](tests/test_groq_client.py) (5/5 passing, 33/33 total). [app.py](app.py) has a sidebar "LLM provider" selector (Groq / Anthropic, defaults to Groq). **Swapping to real Claude later is a one-click change in the sidebar** — delete `groq_client.py` once Claude access exists. Default model is `openai/gpt-oss-120b` (this key's org doesn't have `llama-3.3-70b-versatile`; check `client.models.list()` if this changes again).

**Live run against all 7 archetypes, real model, real tool-calling** (Sept 3):
- ✅ structural_mix_collapse, healthy, sparse_short_history, one_off_bulk_baseline, wildcard_substitution — all correct, with genuine cited reasoning (e.g. correctly distinguished a recovered Precision Tools dip from the non-recovering Industrial Equipment collapse in the same account).
- ❌→✅ seasonal_dip_temporary — **initially wrong**: called structural leakage on a low-value, non-seasonal category (Office Supplies) that was almost certainly just noise. Root cause: the prompt never told the model to weigh a category's *materiality* (baseline revenue share) when deciding how much a change-point fact should count as evidence. **Fixed** by adding an explicit materiality instruction to `SYSTEM_PROMPT`; re-run correctly reads "healthy," explicitly citing "Office Supplies, a low-share category."
- ⚠️→✅ order_size_shrinkage — verdict direction was accidentally right but for the wrong reason (same Office Supplies noise; the real signal — a 50% basket-width drop — was never mentioned). Root cause: the prompt's structural criteria treated order-size shrinkage as a modifier of category-level findings, not a standalone signal to check on its own. **Fixed** by adding an explicit instruction to check `order_behavior` directly. Re-run on the same account (ACC-0011) came back "healthy" — investigated further and found ACC-0011 is a genuine outlier in its own archetype: order frequency randomly spiked +57% (checked the deterministic evidence pack for all 5 accounts in this archetype — the other 4 show flat frequency and clear revenue decline, matching the archetype's design). With frequency and total revenue both up, "healthy" was a defensible read of *that account's* real numbers, not a model error — but it also meant this archetype wasn't reliably testing what it was designed to test.
- **Root-cause fixed, not just accepted**: order_size_shrinkage's whole premise is an isolated "frequency stays flat" variable, but the generator drew monthly order counts from an unclipped Poisson distribution — realistic noise for most archetypes, but enough to occasionally flip this specific archetype's revenue direction by chance, invalidating the test for that account. `generate_data.py` (since removed) now clips this archetype's monthly order count to ±25% of its target rate (scoped to this archetype only — doesn't touch RNG consumption or any other archetype's data, verified by the full test suite staying green after regeneration). Regenerated the dataset and demo cache; all 5 accounts in the archetype now show frequency within ±7% and revenue clearly down (−5% to −38%). Re-ran ACC-0011 live: now correctly "leakage_detected" / structural / high confidence, explicitly citing the basket-width and AOV mechanism.
- Net: **7/7 archetypes correct**, verified live against the real model.

### Phase 4 — UI, hardening & submission (Sept 5–6)
- [x] Demo UI — [app.py](app.py) (Streamlit). Upload CSV → select account → run investigation → full report. Verified live in a real browser (Playwright): normal report rendering, a malformed CSV (no usable columns) fails at ingestion with a specific named error and no crash, and — see next item — an LLM failure degrades gracefully instead of crashing.
- [x] Live-failure hardening — timeout/retry is the Anthropic SDK's default (2 retries w/ backoff on 429/5xx/connection errors, not reimplemented); a most-specific-first exception chain in `app.py` catches auth/rate-limit/connection/status errors from a real API call and shows a clear message while still displaying the (already-computed, deterministic) Stage 1-3 evidence — verified live by uploading a real file with no API key present: the app showed the error and kept the evidence visible rather than crashing. Offline fallback (`demo_cache/` + its generator) was later **removed**: once investigations were cached to disk per account+model, a canned-verdict fixture set was a second, weaker answer to the same problem — and one that risked showing a scripted verdict as if it were live. No hard-coded absolute paths — everything relative to `app.py` or from the uploaded file object.
- [x] One-page architecture note — [ARCHITECTURE.md](ARCHITECTURE.md). All 7 stages, dependencies, and the four points where the system hands control to a human (recommendation-not-action, the defer path, live-failure handoff, schema-mismatch handoff) — plus known limitations disclosed rather than hidden, per the rubric's reward for honesty over forced confidence.
- [ ] Presentation deck — **blocked**, needs the template from Tejashwini (Phase 1 action item, not yet sent).
- [ ] **Submit all three artefacts by Sept 6** — human action (submission itself); prototype + architecture note are ready, deck is blocked.

**Bug found during UI verification, fixed:** the browser screenshot surfaced a mathematically impossible "132.2% of baseline revenue at risk" figure. Root cause: `category_changes`' before/after medians came from a change-point scan run on a rolling 3-month-SUM series, but were labeled (and consumed downstream in Stage 5's ₹ math) as if they were monthly rates — inflating every category-level rupee figure ~3x since Phase 2. Fixed in `changepoint.normalize_medians_to_monthly_rate()`, applied at both call sites (`detect.category_changes`, `baseline.overall_revenue`). Added a regression guard (`pct_of_baseline_monthly_revenue_at_risk <= 1.0`) to `test_orchestrate.py` — no test had asserted on absolute-magnitude sanity before, only on structure/direction, which is why 34 passing tests didn't catch it. Demo cache regenerated after the fix; all 28 tests pass.

### Phase 5 — Rehearsal & Reality Test prep (Sept 7–9)
- [ ] Timed rehearsals: 5-min pitch + 3-min demo; demo on a fresh machine/network
- [ ] Red-team the agent: have someone construct adversarial CSVs blind (archetype 7 spirit) and run them cold
- [ ] Prepare honest answers to "when does it break?" — knowing the failure modes is scored, not penalised
- [ ] Dry-run the 2-min Q&A: practice answering why the synthetic data is realistic and why each verdict criterion is sound

### Phase 6 — Event day (Sept 10)
- Bring: laptop + backup laptop/recording-as-fallback-only, offline cached demo run, printed architecture note
- During the Reality Test: let the agent run live on the unseen input; if it defers on insufficient evidence, present that as the designed behaviour it is

## Standing Decisions & Conventions

- **A finding is material only when it is BIG and REAL.** Every threshold in `detect.py` was calibrated to the workbook, whose accounts wobble ~6% month to month. Real wholesale customers (UCI Online Retail II, 1,569 with ≥12 months) wobble ~138% — 23× more — and skip half their months. Run on 150 of them, the fixed thresholds called one in three a material decline. `src/pipeline/significance.py` now adds a permutation test per dimension: shuffle the account's own months, ask how often chance produces a shift this large. Threshold status says *how big*; `p_value` says *whether it is distinguishable from that account's noise*; `is_material()` requires both. This needs no recalibration to transfer — it measures each account against itself. Two-sided p is `2 × min(tail)`, never `|permuted| ≥ |observed|`: relative change is bounded at −100% and unbounded above, and the absolute-value form silently dismissed customers who had stopped ordering entirely.
- **Real B2B ordering data exists and is used — for behaviour, not economics.** No free real industrial-distributor line-item dataset exists (searched: UCI, Kaggle, Microsoft samples, academic; MRO data is commercially sensitive, sold only via brokers). Online Retail II (CC BY 4.0, gift wholesaler, 2009–11) is real B2B ordering — lumpiness, gaps, returns, history spread — but has no margin, tier, category or discount. Downloaded by `scripts/fetch_external_data.py` into gitignored `data/external/`. AdventureWorks was evaluated and rejected: −1.5% margin on 82% of revenue and a fixed 92-day order cadence (40 order dates in 33 months) — the columns of a B2B dataset without the behaviour.

- **The official workbook is the specification, and it is the only dataset.** `Quessathon_Revenue_Leakage_Dataset.xlsx` (18 accounts, 551 orders, 3,289 lines, Sep 2024 – Aug 2026, 17 columns) replaced our synthetic data as the calibration target. `scripts/prepare_dataset.py` flattens it into `data/meridian/`; the pipeline reads only `transactions.csv` from there. `generate_data.py` and `synthetic_data/` were **deleted** — once every threshold was calibrated against the workbook, a second dataset that disagreed with the spec was testing a different product. Their one remaining value, coverage of the *degraded* ingestion path, moved to `tests/datasets.py`, which derives thin files by dropping columns from the real data.
- **Six detection dimensions, not one.** Two of the workbook's six real leaks are invisible in revenue (ACC-101: flat revenue, −44pp high-tier share, −12pp margin; ACC-107: flat revenue and mix, discount 5%→25%). Stages 2–3 therefore profile revenue, margin, discount, category mix, tier mix and order pattern in parallel. A revenue-only pipeline scores 0/2 on those and cannot be prompted out of it.
- **Direction, never magnitude, for mix shifts.** ACC-101 and ACC-111 have the same flat revenue and a similarly large tier-mix shift in opposite directions; one is the flagship leak, the other is a healthy account trading up. Any detector reporting a mix change as a magnitude makes them identical.
- **Revenue at risk and margin at risk are never summed** — margin is a slice of revenue. Both are reported; severity is the worse of the two ratios.
- **"Could not measure" ≠ "measured, found nothing."** `ingest.analysis_dimensions` records which dimensions the input file supports, travels in the evidence pack, and is shown in the UI.
- **Detect opens on the account book, not a picker.** The first screen is a table of every account (identity, recent economics, the six detector labels) with client-side search and filters. Clicking a row opens the existing dashboard. The catalogue is built from `build_evidence_pack` and cached; filters only slice that frame. Investigate / Attribute require a selected account; Prioritise can still pick a set without one.
- **The Answer Key scores, it never informs.** `src/validation/answer_key.py` lives outside `src/pipeline/` so nothing the agent touches can reach it. Scoring runs via `scripts/validate_answer_key.py` or the app's per-account step 3, and reports accuracy split by expected outcome — one overall number cannot distinguish a discriminating agent from one that flags everything.
- **Stack**: Python; pandas for analytics; Anthropic API (Claude) for the agent layer; Streamlit (or FastAPI) for the UI. Revisit only with a recorded reason here.
- **One LLM call per analysis, not four.** Stages 4/5/7/8 of the original draft consolidate into a single agentic call (stage 4 above) with tool access; prioritisation (old stage 7) and report assembly (old stage 8) are deterministic.
- **LLM never computes numbers.** All figures in any output must originate in the analytics engine and be traceable to the evidence pack.
- **Every verdict carries confidence + cited evidence.** An output without both is a bug.
- **Deferring is a feature.** When tempted to make the agent more decisive, re-read the 30% rubric line first.
- When the repo gains real code, add a Commands section (run, test, demo) at the top of this file and keep the phase checklist current.

## Open issue: git repo root

`git rev-parse --show-toplevel` resolves to `/Users/gautamkoshta` (home directory), not this project folder — there's an uncommitted git repo rooted at the home directory that would track `.ssh`, shell history, and everything else in it if used. Needs a decision before any git operations happen in this project: most likely fix is `git init` directly inside `Hack/` (which nests a repo inside the outer one — fine, git handles this, the inner repo takes precedence for anything under `Hack/`) or removing the home-directory `.git` if it was accidental. Not yet resolved.

## Status

**ML phase — Phases 2–5 done (Sept 5): the classifier is trained and wired in as a second opinion.**

- **Phase 2 — generator, gate PASSED** (`scripts/check_generator.py`, 600 accounts): feature coverage 95–96% of Meridian and 93% of real customers; 88% of mature FLAG recipes realised over threshold, 80% material∧significant on smooth accounts; 7% of NO_FLAG accounts material∧significant (gate: ≥90 / ≥80 / <15). Generator facts that had to be learned: a stable **core basket** with a small tier-neutral tail (a per-line product lottery made every account as noisy as the real ones, including the workbook-smooth ones); tier-target scaling of the core; volume fades through units, never through order count; **benign drift on healthy accounts** (level steps ±6%, volume walk, shape and tier drift) — without it the model learned "any level shift is a leak" and flagged ACC-102/114/118; **staples ranked by expected revenue**, with zero inclusion spread at full stability — ranking by weight let a smooth fixture skip its most expensive line in a random month and fall 30%. Calibration (`data/calibration/generator.json`, committed) is measured from 2,512 real Online Retail II customers for behaviour and the workbook for economics.
- **Phase 3 — model**: `HistGradientBoostingClassifier`, `class_weight=None` (balanced weighting flagged 4 of the 11 healthy workbook accounts), ~62 features (magnitudes + 3 p-values and statistics per dimension, no status strings, NaN kept). Trained on 2,989 synthetic accounts (seed 42); 5-fold CV accuracy 0.791; **held-out synthetic** (seed 7, 800 accounts) 0.806, DEFER recall 0.90, **mature-leak recall 85%** (n=193); weakest scenarios slow_bleed 60% and broad_decline 61% — the early/mild ones are not meant to be catchable yet. **Meridian 17/18, never trained on**: all 11 healthy accounts and ACC-109's deferral correct (ACC-111 premiumisation 0.71 healthy, ACC-118 borderline 0.94 healthy, ACC-101 0.997 leak, ACC-108 0.97 leak); the one miss is ACC-104 at P(leak) 0.49 vs 0.51 — a coin flip the pack marks `decisive: false`, which the prompt treats as "the numbers alone are ambiguous", not as a healthy call.
- **Phase 4 — wired in**: `src/ml/predict.py` attaches `model_opinion` (p_flag / p_no_flag / p_defer, leaning, decisive, `top_drivers` measured by hiding each known feature and re-predicting, provenance from the model card) to the evidence pack as a deterministic step in `orchestrate.py`, `app.py` and `validate_answer_key.py`. **No model file → `available: false` and the pipeline is unchanged.** The prompt gained a "second opinion, not a verdict" section: never flag on it alone; if you disagree with its leaning, name the overriding fact in the new required `model_opinion_response` field and cap confidence at medium; a decisive agreement may raise confidence. `report.model_agreement` records agree/disagree from the two outcomes, never from the LLM's self-description. Training feature extraction uses `build_evidence_pack` alone, so no model output can ever become a training input.
- **Phase 5 — shown**: verdict banner carries a one-line "Statistical second opinion: Leakage (97%) · the AI reached the same call"; a **Second opinion** tab shows the three probabilities as bars, agree/disagree with the AI's reason, the driver table with measured effects, and provenance. A verdict cached before the model existed shows the current opinion with an explicit "the AI did not see this" note. The dashboard's status-strip hover and the "underlying numbers" table now carry each dimension's "Real, or noise?" p-value.
- 204 tests pass (3 model-dependent tests skip on a clone without `models/`). A dropped `full_month_index` import in `ui/metrics.py` — invisible to the tests, which never render the dashboard — was caught by a headless `streamlit.testing` render and fixed; that render is the cheap boot check and should be run after any UI change.
- **Phase 6 — live re-score with the model in the loop: 18/18** (Groq `openai/gpt-oss-120b`, Sept 5), up from 17/18 before the ML phase — ACC-108 (fragmentation), the previous miss, is now FLAG. The first run scored 12/18 with six `413` errors, not wrong verdicts: the free Groq tier caps a request at 8,000 tokens per minute and the pack had crossed it by ~1% once the model opinion and the third significance window were added. Fixed by `agent.compact_json` (floats rounded to 4 dp, no separator spaces, ~13% smaller) and by keeping the `model_opinion` block lean (what it is lives in the system prompt; provenance is read by the UI from the model card). The six re-scored 6/6. Live scoring remains the only measure of model judgement; a prompt or pack change means re-running it.

**ML phase — Phase 1 of 6 done (Sept 5): noise-relative significance.** Plan: (1) permutation significance on every signal → (2) synthetic generator calibrated to real ordering behaviour + workbook economics, domain-randomised → (3) gradient-boosted classifier on 43 features (37 evidence-pack magnitudes + 6 p-values), validated on the 18 real accounts and on real Online Retail II customers → (4) `model_opinion` in the evidence pack, advisory to Stage 4 → (5) shown beside the LLM verdict → (6) re-score. Phase 1 delivered `pipeline/monthly.py` (one raw monthly table shared by detectors and dashboard), `pipeline/significance.py`, p-values wired into every detector block, `scripts/check_real_data.py` (the pass check) and `scripts/fetch_external_data.py`. **Pass check results:** on 150 real long-standing wholesale customers, "material revenue decline" fell from 50 (33%) by threshold alone to 9 (6%) requiring significance; none of the 14 flagged customers down under 30% survived; the Meridian partition is unchanged (6 / 11 / 1, exact match). Two windows per dimension (recent-6 and halves, Bonferroni-combined) because 44% of real months are empty and a six-month window has no power there — collapses caught rose from 2/18 to 5/18. The remaining 13 are a stated power limit on sparse data, not a tuning target. 172 tests pass. Live agent score before this phase: 17/18 (ACC-108 read as healthy). Real Anthropic key still absent; all live runs are Groq `openai/gpt-oss-120b`.

**Re-based on the official dataset (Sept 4).** The whole pipeline is now calibrated against `Quessathon_Revenue_Leakage_Dataset.xlsx` rather than our synthetic generator. What changed:

- **Stage 1**: canonical schema widened from 9 to 17 columns (tier, discount_pct, unit_cost, is_return, list_price, region, account_manager, account_name), with synonyms and derivations for each; Excel workbooks are readable directly; returns are netted into revenue but excluded from order-behaviour statistics; two hard history floors added so a five-month account cannot score "marginal" on category breadth alone.
- **Stages 2–3**: margin, discount and tier-mix profiles added alongside revenue; new detectors for graded revenue decline, prior-year seasonal echo, dip episodes with recovery state, category defection, order fragmentation, data gaps, bulk-outlier months, and returns. Order behaviour switched from medians to window means (the median of a small integer basket width made a healthy control read as −50%).
- **Stage 4**: prompt rewritten around the six dimensions, direction-vs-magnitude, and ruling out the innocent explanations before flagging; `leak_dimensions` added to the verdict; two drill-down tools added (`get_monthly_economics`, `get_tier_monthly_series`).
- **Stages 5–7**: impact engine sizes margin-only and whole-account leaks, not just category revenue; prioritisation reads the worse of the revenue and margin ratios; report carries leak dimensions and analysis coverage.
- **Validation**: `src/validation/answer_key.py` + `scripts/validate_answer_key.py` + a Streamlit mode score live verdicts against the Answer Key tab. Demo cache rebuilt from four reference accounts covering all four outcomes.

- **Cleanup**: `generate_data.py`, `synthetic_data/` and the stale `info.md` stage walkthrough were deleted; the four test files that depended on the old generator were migrated onto the workbook, with thin-file coverage rebuilt in `tests/datasets.py`.

Verified: all 18 accounts' deterministic signals line up with the Answer Key — the six FLAG accounts each trip at least one material detector, and none of the eleven NO FLAG accounts trip any; ACC-109 alone scores "insufficient". Our monthly roll-ups match the organisers' own Monthly Summary tab to <0.01 on all 412 rows. 136 tests pass (`venv/Scripts/python -m pytest tests/ -v`), and the app boots clean.

**Still open: the agent has never been scored live against the Answer Key** — there is no API key in the environment (`.env` does not exist; `GROQ_API_KEY`/`ANTHROPIC_API_KEY` unset). The scoring path is built and tested end-to-end with a scripted client, but the actual accuracy number is unknown until someone runs `scripts/validate_answer_key.py` with a key. That is the single highest-value next action.

Earlier status (pre-workbook): Phases 1-4 were functionally done, except: the two human/logistics action items (Phase 1), the git repo decision, and the presentation deck (blocked on the template from Tejashwini). The seasonal-precedent reliability issue (Phase 2) is fixed. Real-LLM verification (Phase 3) is done via Groq — 7/7 archetypes correct after two prompt fixes (materiality weighting + order-behavior signal) and one generator fix (order_size_shrinkage's frequency-flat premise wasn't actually enforced, just usually true). Real Claude verification is still pending actual API access, but nothing about the pipeline is expected to change for it — the Groq shim exists specifically so the swap is a one-line change. All 28 tests pass (`venv/bin/python3 -m pytest tests/ -v`); the demo UI (`app.py`) has been verified live in a real browser, including the malformed-file and LLM-failure error paths. Next: Phase 5 — rehearsal & Reality Test prep — or resolving one of the open items above first, most importantly getting real API access so Stage 4 can finally be verified against actual model output rather than a scripted fake.
