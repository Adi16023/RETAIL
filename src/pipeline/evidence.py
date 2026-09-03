"""
Evidence pack assembly — combines Stage 1 (sufficiency), Stage 2 (baseline)
and Stage 3 (change detection) into the single structured JSON object the
Stage 4 LLM reasons over. The LLM never sees raw transactions and never
computes a number; every figure here traces back to deterministic code.
"""

from __future__ import annotations

import pandas as pd

from .baseline import build_baseline
from .detect import category_changes, product_changes
from .ingest import account_sufficiency


def build_evidence_pack(df: pd.DataFrame, account_id: str) -> dict:
    acc_df = df[df["account_id"] == account_id].reset_index(drop=True)
    if acc_df.empty:
        raise ValueError(f"No rows for account_id={account_id!r}")

    sufficiency = account_sufficiency(df).get(account_id)
    baseline = build_baseline(df, account_id)
    changes = category_changes(acc_df, baseline["category_mix"]["high_value_categories"])
    products = product_changes(acc_df)

    return {
        "account_id": account_id,
        "data_sufficiency": sufficiency,
        "history": {
            "start_date": str(acc_df["date"].min().date()),
            "end_date": str(acc_df["date"].max().date()),
            "months_of_history": sufficiency["history_months"] if sufficiency else None,
            "order_count": sufficiency["order_count"] if sufficiency else None,
            "category_count": sufficiency["category_count"] if sufficiency else None,
        },
        "overall_revenue": baseline["overall_revenue"],
        "category_mix": baseline["category_mix"],
        "order_behavior": baseline["order_behavior"],
        "category_changes": changes,
        "product_changes": products,
    }
