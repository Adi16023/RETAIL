"""
Generates the offline demo cache under demo_cache/ — pre-computed reports
for known archetype accounts, used by app.py's "View offline demo" mode
when live LLM access isn't available (venue has no internet, no API key,
rate limited, etc. — see plan.md Phase 4 "live-failure hardening").

These reports are produced with a SCRIPTED verdict standing in for the LLM
call, not a real model response — clearly labeled as such in the output
and in the UI. They exist so the demo can run end-to-end on stage even if
Stage 4 is unreachable; they are not a substitute for the real pipeline
when it's available.

Run from the repo root: venv/bin/python3 scripts/generate_demo_cache.py
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "tests"))

from pipeline.evidence import build_evidence_pack
from pipeline.ingest import ingest
from pipeline.orchestrate import run_for_account

from fakes import ScriptedClient, message, tool_use_block

GENERATED_CSV = REPO_ROOT / "synthetic_data" / "transactions.csv"
GROUND_TRUTH = REPO_ROOT / "synthetic_data" / "ground_truth.json"
OUT_DIR = REPO_ROOT / "demo_cache"


def account_for(archetype: str) -> str:
    gt = json.load(open(GROUND_TRUTH))
    return next(g["account_id"] for g in gt if g["archetype"] == archetype)


def scripted_client(verdict: dict) -> ScriptedClient:
    return ScriptedClient([message([tool_use_block("submit_verdict", verdict, "toolu_1")], stop_reason="tool_use")])


def build_cache_entry(archetype: str, account_id: str, verdict: dict) -> dict:
    report = run_for_account(scripted_client(verdict), str(GENERATED_CSV), account_id)
    report["_cache_metadata"] = {
        "archetype": archetype,
        "source": "scripted demo verdict, not a real LLM response",
    }
    return report


def main():
    OUT_DIR.mkdir(exist_ok=True)

    structural_account = account_for("structural_mix_collapse")
    df, _ = ingest(str(GENERATED_CSV))
    pack = build_evidence_pack(df, structural_account)
    hv_cat = next(
        c["category"] for c in pack["category_changes"]
        if c["change_point_detected"] and c["pct_decline"] >= 0.5
    )
    structural_verdict = {
        "verdict": "leakage_detected", "temporary_or_structural": "structural",
        "confidence": "high", "defer": False, "attributed_categories": [hv_cat],
        "cited_facts": [
            f"{hv_cat} revenue collapsed to near zero with a detected change-point and no recovery",
            "Total account revenue stayed roughly flat over the same period",
        ],
        "narrative": (
            f"This account looks healthy at the top line — total revenue is essentially flat. "
            f"But {hv_cat}, previously a major share of this account's business, has completely "
            f"stopped, replaced by lower-value purchases elsewhere. This is the classic "
            f"'quietly losing value while still buying' pattern."
        ),
        "recommended_actions": [
            f"Have the account owner reach out specifically about {hv_cat} — find out if it moved to a competitor or another budget line.",
            "Treat this as a retention priority despite stable top-line revenue.",
        ],
        "data_needed_if_deferring": [],
    }

    healthy_account = account_for("healthy")
    healthy_verdict = {
        "verdict": "healthy", "temporary_or_structural": "not_applicable",
        "confidence": "high", "defer": False, "attributed_categories": [],
        "cited_facts": ["No change-point detected at total or category level"],
        "narrative": "Purchasing behaviour is stable across revenue, category mix, and order size — no leakage signal.",
        "recommended_actions": [], "data_needed_if_deferring": [],
    }

    entries = {
        "structural_mix_collapse": build_cache_entry("structural_mix_collapse", structural_account, structural_verdict),
        "healthy": build_cache_entry("healthy", healthy_account, healthy_verdict),
    }

    for name, report in entries.items():
        path = OUT_DIR / f"{name}.json"
        with open(path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
