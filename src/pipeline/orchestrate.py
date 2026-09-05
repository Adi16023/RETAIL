"""
Top-level pipeline orchestrator: wires Stage 1 (ingest) through Stage 7
(report) together for one account. The `client` parameter is an injected
Anthropic-compatible client (real or, in tests, a scripted fake) — see
agent.py's docstring for why it's injected rather than constructed here.
"""

from __future__ import annotations

import pandas as pd

from ml.predict import attach_model_opinion

from .agent import DEFAULT_MODEL, investigate
from .evidence import build_evidence_pack
from .impact import compute_impact
from .ingest import ingest
from .prioritize import prioritize
from .report import assemble_report


def run_for_account(
    client,
    source: str | pd.DataFrame,
    account_id: str,
    model: str = DEFAULT_MODEL,
) -> dict:
    df, _ingestion_report = ingest(source)
    # The classifier's second opinion is attached as a deterministic step
    # between evidence assembly and the LLM. With no trained model on disk it
    # attaches `available: false` and the run is exactly the old pipeline.
    evidence_pack = attach_model_opinion(build_evidence_pack(df, account_id))
    verdict = investigate(client, df, account_id, evidence_pack, model=model)
    impact = compute_impact(evidence_pack, verdict)
    priority = prioritize(impact, verdict)
    return assemble_report(account_id, evidence_pack, verdict, impact, priority)
