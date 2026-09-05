"""
Phase 1 pass check: does noise-relative significance stop the pipeline from
over-flagging REAL wholesale customers?

Runs the actual pipeline on a sample of long-standing customers from UCI
Online Retail II (real B2B ordering data, revenue-only) and compares two
definitions of a material revenue signal:

  threshold only         revenue_decline.status == material_decline
  threshold + p < 0.05   ...and the permutation test agrees it is not noise

The workbook's own thesis is that flagging most of the book is THE failure
mode, so the target is a flag rate that looks like a real book — roughly one
in ten — not one in three. Also prints the Meridian split as a regression
check: adding significance must leave the 6 / 11 / 1 partition intact.

Needs data/external/online_retail_II.csv. Run scripts/fetch_external_data.py
first if it is missing.

    python scripts/check_real_data.py [--customers 150] [--seed 0]
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pipeline.evidence import build_evidence_pack
from pipeline.ingest import ingest
from pipeline.significance import SIGNIFICANCE_LEVEL

REAL_CSV = REPO_ROOT / "data" / "external" / "online_retail_II.csv"
MERIDIAN_CSV = REPO_ROOT / "data" / "meridian" / "transactions.csv"

MIN_HISTORY_MONTHS = 18
MIN_ORDERS = 10


def load_real_sample(n_customers: int, seed: int) -> pd.DataFrame:
    raw = pd.read_csv(REAL_CSV).dropna(subset=["Customer ID"])
    raw["is_return"] = raw["Invoice"].astype(str).str.startswith("C")
    raw["revenue"] = raw["Quantity"] * raw["Price"]
    raw["month"] = pd.to_datetime(raw["InvoiceDate"]).dt.to_period("M")

    sales = raw[~raw["is_return"]]
    per_customer = sales.groupby("Customer ID").agg(
        first=("month", "min"), last=("month", "max"), orders=("Invoice", "nunique"))
    per_customer["history"] = (per_customer["last"] - per_customer["first"]).apply(lambda d: d.n) + 1
    core = per_customer[(per_customer["history"] >= MIN_HISTORY_MONTHS)
                        & (per_customer["orders"] >= MIN_ORDERS)].index
    chosen = pd.Series(core).sample(min(n_customers, len(core)), random_state=seed)
    sub = raw[raw["Customer ID"].isin(chosen)]

    # Revenue-only file in the pipeline's canonical vocabulary. StockCode
    # prefix stands in for category — the dataset has none.
    return pd.DataFrame({
        "order_id": sub["Invoice"].astype(str),
        "account_id": "C" + sub["Customer ID"].astype(int).astype(str),
        "order_date": sub["InvoiceDate"],
        "product_id": sub["StockCode"].astype(str),
        "category": sub["StockCode"].astype(str).str[:2],
        "unit_price": sub["Price"],
        "quantity": sub["Quantity"],
        "revenue": sub["revenue"],
    })


def flag_rates(df: pd.DataFrame) -> dict:
    n = df["account_id"].nunique()
    by_threshold, by_both, statuses = 0, 0, Counter()
    p_values = []
    for account_id in sorted(df["account_id"].unique()):
        pack = build_evidence_pack(df, account_id)
        status = pack["revenue_decline"]["status"]
        statuses[status] += 1
        sig = pack["significance"]["revenue"]
        if sig["p_value"] is not None:
            p_values.append(sig["p_value"])
        if status == "material_decline":
            by_threshold += 1
            if sig.get("significant"):
                by_both += 1
    return {"n": n, "threshold_only": by_threshold, "threshold_and_p": by_both,
            "statuses": statuses, "median_p": float(np.median(p_values)) if p_values else None}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--customers", type=int, default=150)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not REAL_CSV.exists():
        print(f"Real data not found at {REAL_CSV}. Run scripts/fetch_external_data.py first.", file=sys.stderr)
        return 2

    print(f"Loading {args.customers} real wholesale customers "
          f"(>= {MIN_HISTORY_MONTHS} months, >= {MIN_ORDERS} orders)...")
    df, _ = ingest(load_real_sample(args.customers, args.seed))
    real = flag_rates(df)

    n = real["n"]
    print(f"\n=== REAL customers (n={n}) — 'material revenue decline' ===")
    print(f"  threshold only            {real['threshold_only']:4}  ({real['threshold_only']/n*100:.0f}%)")
    print(f"  threshold AND p<{SIGNIFICANCE_LEVEL}      {real['threshold_and_p']:4}  ({real['threshold_and_p']/n*100:.0f}%)")
    print(f"  median revenue p-value    {real['median_p']:.2f}")
    print("  status mix:", dict(real["statuses"]))

    print("\n=== MERIDIAN regression check ===")
    mdf, _ = ingest(str(MERIDIAN_CSV))
    material_statuses = {"revenue_decline": ("material_decline",), "margin": ("erosion_detected",),
                         "discount": ("creep_detected",), "tier_mix": ("downgrade_detected",)}
    flagged = []
    for account_id in sorted(mdf["account_id"].unique()):
        pack = build_evidence_pack(mdf, account_id)
        hits = []
        for block, statuses in material_statuses.items():
            b = pack.get(block) or {}
            if b.get("status") in statuses and (b.get("p_value") is None or b.get("significant")):
                hits.append(block)
        if pack["order_pattern"]["status"] == "fragmentation_detected" and pack["order_pattern"]["significant"]:
            hits.append("order_pattern")
        if any(c["defected"] for c in pack["category_changes"]):
            hits.append("defection")
        if hits:
            flagged.append(account_id)
    expected = ["ACC-101", "ACC-104", "ACC-106", "ACC-107", "ACC-108", "ACC-112"]
    ok = flagged == expected
    print(f"  material AND significant: {flagged}")
    print(f"  expected FLAG accounts:   {expected}")
    print("  ->", "MATCH" if ok else "MISMATCH")

    passed = ok and real["threshold_and_p"] / n <= 0.15
    print("\nPHASE 1 PASS" if passed else "\nPHASE 1 FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
