# Revenue Leakage Investigator

Quessathon "Retail: Where Did the Revenue Go?" challenge. An AI agent that investigates a B2B account's transaction history to find where and why revenue is leaking — even when the account still looks healthy at the top line — and flags when the evidence isn't enough to say for sure.

See [plan.md](plan.md) for the 7-stage pipeline design, full build log, and current status.

## Setup

```bash
python3 -m venv venv
venv/bin/pip install -r requirements-dev.txt
cp .env.example .env   # fill in GROQ_API_KEY (or ANTHROPIC_API_KEY once available)
```

## Run

```bash
venv/bin/streamlit run app.py
```

## Test

```bash
venv/bin/python3 -m pytest tests/ -v
```
