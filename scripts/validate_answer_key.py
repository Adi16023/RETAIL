"""
Scores the live agent against the workbook's Answer Key from the command
line — the same comparison the Streamlit "Answer Key validation" mode shows,
without needing the UI.

Runs the full seven-stage pipeline once per account (one live LLM call each),
normalises each verdict to FLAG / NO FLAG / DEFER, and compares it against
the key. The key is only ever used to score the finished report; it is never
passed to the agent.

Run from the repo root:

    python scripts/validate_answer_key.py                      # all 18 accounts
    python scripts/validate_answer_key.py --accounts ACC-101 ACC-109
    python scripts/validate_answer_key.py --provider anthropic --model claude-opus-5
    python scripts/validate_answer_key.py --out scorecard.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(REPO_ROOT / ".env")
sys.path.insert(0, str(REPO_ROOT / "src"))

from pipeline.agent import DEFAULT_MODEL
from pipeline.evidence import build_evidence_pack
from pipeline.impact import compute_impact
from pipeline.ingest import ingest
from pipeline.prioritize import prioritize
from pipeline.report import assemble_report
from validation.answer_key import load_answer_key, score_error, score_report, summarize

TRANSACTIONS = REPO_ROOT / "data" / "meridian" / "transactions.csv"

OUTCOME_WIDTH = 8


def build_client(provider: str, model: str | None):
    if provider == "groq":
        from pipeline.groq_client import GroqShimClient
        return GroqShimClient(model_override=model)
    import anthropic
    return anthropic.Anthropic()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--provider", choices=["groq", "anthropic"], default="groq")
    parser.add_argument("--model", default=None, help="Model id; defaults to the provider's default.")
    parser.add_argument("--accounts", nargs="*", default=None, help="Subset of account ids to score.")
    parser.add_argument("--out", default=None, help="Write the full scorecard JSON here.")
    args = parser.parse_args()

    if not TRANSACTIONS.exists():
        print(f"Reference transactions not found at {TRANSACTIONS}.", file=sys.stderr)
        print("Run `python scripts/prepare_dataset.py` first.", file=sys.stderr)
        return 1

    key = load_answer_key()
    df, _ = ingest(str(TRANSACTIONS))
    accounts = args.accounts or sorted(df["account_id"].unique())

    model = args.model or (None if args.provider == "groq" else DEFAULT_MODEL)
    try:
        client = build_client(args.provider, model)
    except Exception as e:
        key_name = "GROQ_API_KEY" if args.provider == "groq" else "ANTHROPIC_API_KEY"
        print(f"Could not initialise the {args.provider} client: {e}", file=sys.stderr)
        print(
            f"Set {key_name} in .env (copy .env.example) — scoring the agent needs live "
            "model access; there is no offline substitute for a real verdict.",
            file=sys.stderr,
        )
        return 2
    # The Groq shim substitutes its own default for an Anthropic model id, so
    # passing DEFAULT_MODEL through is harmless either way.
    call_model = model or DEFAULT_MODEL

    results = []
    for account_id in accounts:
        if account_id not in key:
            print(f"  {account_id}: not in the answer key, skipped", file=sys.stderr)
            continue
        try:
            pack = build_evidence_pack(df, account_id)
            from pipeline.agent import investigate
            verdict = investigate(client, df, account_id, pack, model=call_model)
            impact = compute_impact(pack, verdict)
            priority = prioritize(impact, verdict)
            report = assemble_report(account_id, pack, verdict, impact, priority)
            result = score_report(report, key[account_id])
        except Exception as e:
            result = score_error(account_id, key[account_id], f"{type(e).__name__}: {e}")
        results.append(result)

        mark = "PASS" if result["outcome_match"] else "FAIL"
        print(
            f"{mark}  {result['account_id']}  "
            f"expected {result['expected_outcome']:<{OUTCOME_WIDTH}} "
            f"got {result['actual_outcome']:<{OUTCOME_WIDTH}} "
            f"({result['archetype']})"
        )
        if result.get("error"):
            print(f"      error: {result['error']}")

    summary = summarize(results)
    print()
    print(f"Accuracy: {summary['matched']}/{summary['total']}"
          + (f" ({summary['accuracy'] * 100:.0f}%)" if summary["accuracy"] is not None else ""))
    for outcome, bucket in sorted(summary["by_expected_outcome"].items()):
        print(f"  expected {outcome:<8} {bucket['matched']}/{bucket['total']}")
    if summary["mismatches"]:
        print("\nMismatches:")
        for line in summary["mismatches"]:
            print(f"  {line}")

    if args.out:
        Path(args.out).write_text(
            json.dumps({"summary": summary, "results": results}, indent=2), encoding="utf-8"
        )
        print(f"\nWrote {args.out}")

    return 0 if summary["matched"] == summary["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
