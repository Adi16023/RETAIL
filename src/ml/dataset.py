"""
Turning transaction tables into feature matrices — the one path the trainer,
the generator gate and any evaluation script share.

Every account, synthetic or real, goes through the SAME steps as a live
investigation: ingest -> build_evidence_pack -> extract_features. Nothing is
computed for the model that the pipeline would not compute for the LLM, so
train-time and run-time features cannot drift apart.

Building an evidence pack is the slow step (twelve permutation tests of 2,000
shuffles each, plus the detectors). At ~0.2s an account, 3,000 accounts is
ten minutes single-threaded, so this runs across processes.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from ml.features import FEATURE_NAMES, extract_features
from pipeline.evidence import build_evidence_pack
from pipeline.ingest import ingest


def _features_for_one(args: tuple) -> tuple[str, dict]:
    account_id, account_df = args
    pack = build_evidence_pack(account_df, account_id)
    return account_id, extract_features(pack)


def features_for_transactions(transactions: pd.DataFrame, account_ids: list[str] | None = None,
                              workers: int | None = None) -> pd.DataFrame:
    """Feature row per account for an already-INGESTED transactions frame.

    Each worker receives only its account's rows: `build_evidence_pack` only
    needs the account slice (data_sufficiency is computed per account), and
    shipping the whole table to every process would dominate the runtime.
    """
    account_ids = account_ids or sorted(transactions["account_id"].unique())
    jobs = [(account_id, transactions[transactions["account_id"] == account_id].reset_index(drop=True))
            for account_id in account_ids]

    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    if workers == 1 or len(jobs) < 8:
        results = [_features_for_one(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_features_for_one, jobs, chunksize=8))

    frame = pd.DataFrame.from_dict(dict(results), orient="index")[FEATURE_NAMES]
    frame.index.name = "account_id"
    return frame.loc[account_ids]


def features_for_raw(raw: pd.DataFrame, workers: int | None = None) -> pd.DataFrame:
    """Same, from a raw file in any supported vocabulary (runs ingestion first)."""
    clean, _ = ingest(raw)
    return features_for_transactions(clean, workers=workers)
