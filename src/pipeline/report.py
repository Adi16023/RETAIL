"""
Stage 7: Report Assembly — merges Stage 4's narrative with Stage 5/6's
rupee figures and priority bucket into the final output. Purely
templating/assembly, deterministic, no new LLM call — the narrative text
was already generated once, in Stage 4.
"""

from __future__ import annotations


def assemble_report(account_id: str, evidence_pack: dict, verdict: dict, impact: dict, priority: dict) -> dict:
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
        "financial_impact": impact,
        "prioritization": priority,
        "data_sufficiency": evidence_pack["data_sufficiency"],
    }
