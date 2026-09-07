# Revenue Leakage Investigator

Quessathon "Retail: Where Did the Revenue Go?" challenge. An AI agent that investigates a B2B account's transaction history to find where and why revenue is leaking — even when the account still looks healthy at the top line — and flags when the evidence isn't enough to say for sure.

Built against the official dataset, `Quessathon_Revenue_Leakage_Dataset.xlsx`: 18 Meridian Supply Co accounts, 551 orders, 3,289 transaction lines, Sep 2024 – Aug 2026. Only ~6 of those 18 accounts are genuine structural leaks, so the discrimination — not the alarm — is the product.

See [plan.md](plan.md) for the 7-stage pipeline design, full build log, and current status.

## Setup

```bash
python -m venv venv
venv/Scripts/pip install -r requirements-dev.txt   # venv/bin/pip on POSIX
cp .env.example .env   # fill in GROQ_API_KEY (or ANTHROPIC_API_KEY once available)

venv/Scripts/python scripts/prepare_dataset.py     # workbook -> data/meridian/*.csv
```

## Run

```bash
venv/Scripts/streamlit run app.py
```

One screen. Pick an account, then work down three steps:

1. **What the data says** — the deterministic evidence: status strip, KPI tiles, and tabs for trend, value mix, pricing, order pattern, data quality and the raw monthly table. No model, no API calls.
2. **What the AI concludes** — one LLM call. Choose **Open source** or **Proprietary**; the verdict arrives with tabs for evidence, what was ruled out, the statistical model's second opinion, what to do, and confidence limits. Results are cached per account + model, so re-visiting is instant and free.
3. **Lifetime value** — what the account is expected to be worth over the next 12 and 24 months on its baseline path and on its current path, and the gap between them: the leak's lifetime cost. Expected activity comes from a BG/NBD model fitted on the whole book; value is margin per month. With no account open, the whole book is ranked by value at risk. `scripts/check_lifetime_model.py` backtests the projection on the last six months.
4. **AryaChat** — ask about any account by name, or about the whole book, in plain words ("why is Northgate flagged", "how often does Harbor order", "which accounts should I call first", "compare Northgate with Harbor"). No account to pick and nothing pre-written: the analyst works out which account you mean, keeps it in focus for your follow-ups, and looks the answer up with the right tool. The answer streams in as it is written, with a chip for every look-up the analyst made on the way (open one to see what came back), and two or three next-question chips underneath — tap one to ask it. Answers come from the analysis and its drill-down tools, never from the model's own arithmetic: every figure in an answer is checked against what the analysis actually produced, and one that cannot be found is flagged. Open as many chats as you like per account; each keeps its own context, compacted automatically when it grows long. Conversations are stored as JSON under `.cache/aryachat/`.

## The statistical second opinion

Beside the LLM sits a gradient-boosted classifier that reads the same evidence pack, reduced to numbers, and returns P(leakage) / P(healthy) / P(defer) with the facts those rest on. It is trained on generated accounts whose problems are known by construction — calibrated to real B2B ordering behaviour (UCI Online Retail II) and to the workbook's economics — and validated on the 18 real accounts it never trained on. The LLM is told what it is and must explain any disagreement; the UI shows both and whether they agree.

```bash
venv/Scripts/python scripts/fetch_external_data.py      # real ordering behaviour (45 MB, gitignored)
venv/Scripts/python scripts/check_generator.py          # generator gate: must print PHASE 2 PASS
venv/Scripts/python scripts/generate_training_data.py --accounts 3000 --seed 42
venv/Scripts/python scripts/generate_training_data.py --accounts 800 --seed 7 --name test_7
venv/Scripts/python scripts/train_leak_classifier.py    # -> models/leak_classifier.joblib (gitignored)
```

Without a trained model the app and pipeline run exactly as before; the second-opinion tab simply says none is available.

## Score the agent against the Answer Key

```bash
venv/Scripts/python scripts/validate_answer_key.py                    # all 18 accounts
venv/Scripts/python scripts/validate_answer_key.py --accounts ACC-101 ACC-109
venv/Scripts/python scripts/validate_answer_key.py --out scorecard.json
```

Accuracy is reported overall and split by expected outcome (FLAG / NO FLAG / DEFER) — a single number can't tell a discriminating agent from one that flags everything. The Answer Key is used only to score finished reports; it is never shown to the agent.

## Test

```bash
venv/Scripts/python -m pytest tests/ -v
```

The suite runs with a scripted stand-in for the model, so it verifies the deterministic stages and the scorecard logic without an API key. It does not measure model judgement — that's what `validate_answer_key.py` is for.
