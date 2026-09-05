"""
Stage 7: Report Assembly — merges Stage 4's narrative with Stage 5/6's
rupee figures and priority bucket into the final output. Purely
templating/assembly, deterministic, no new LLM call — the narrative text
was already generated once, in Stage 4.

The report carries the full evidence pack and a derived evidence timeline
alongside the verdict. Without them the report is a conclusion with no way
back to the facts: the PDF export, and anything else that has to justify a
verdict rather than merely state it, would have no dates, no series and no
thresholds to show. Keeping them on the report also means a cached report
in demo_cache/ is as explainable offline as a live run is.
"""

from __future__ import annotations

from .timeline import build_timeline

# The verdict enum in the model's own three-way vocabulary. `defer` wins: an
# agent that says leakage_detected but defers has declined to make the call.
_VERDICT_OUTCOME = {"leakage_detected": "FLAG", "healthy": "NO_FLAG", "insufficient_data": "DEFER"}


def verdict_outcome(verdict: dict) -> str:
    if verdict.get("defer"):
        return "DEFER"
    return _VERDICT_OUTCOME.get(verdict.get("verdict"), "UNKNOWN")


def model_agreement(evidence_pack: dict, verdict: dict) -> dict | None:
    """Did the LLM land where the classifier leaned? Computed here, from the
    two outcomes, so the report never has to trust the model's own account of
    whether it agreed with the second opinion."""
    opinion = evidence_pack.get("model_opinion") or {}
    if not opinion.get("available"):
        return None
    agent = verdict_outcome(verdict)
    return {
        "agrees": agent == opinion["leaning_outcome"],
        "agent_outcome": agent,
        "model_outcome": opinion["leaning_outcome"],
        "model_decisive": bool(opinion.get("decisive")),
        "agent_response": (verdict.get("model_opinion_response") or "").strip(),
    }


def assemble_report(account_id: str, evidence_pack: dict, verdict: dict, impact: dict, priority: dict) -> dict:
    return {
        "account_id": account_id,
        "account_name": evidence_pack.get("account_name"),
        "verdict": verdict["verdict"],
        "temporary_or_structural": verdict["temporary_or_structural"],
        # Which dimension the value is leaving through (revenue, margin, mix,
        # discount, order_pattern). Two accounts can share a verdict of
        # "structural" and need completely different interventions; without
        # this the report cannot say which.
        "leak_dimensions": verdict.get("leak_dimensions", []),
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
        "analysis_dimensions": evidence_pack.get("analysis_dimensions"),
        # The classifier's second opinion and whether the LLM agreed with it.
        # Both come from the pack and the verdict enum — nothing here is the
        # model's own description of itself.
        "model_opinion": evidence_pack.get("model_opinion"),
        "model_agreement": model_agreement(evidence_pack, verdict),
        # The deterministic case behind the verdict, kept with it: a dated
        # event log for reading, and the pack itself so every figure in any
        # rendering is traceable to the stage that computed it.
        "evidence_timeline": build_timeline(evidence_pack),
        "evidence": evidence_pack,
    }
