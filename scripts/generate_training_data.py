"""
Generates the labelled synthetic accounts the leak classifier trains on.

    python scripts/generate_training_data.py                       # 3000 accounts, seed 42
    python scripts/generate_training_data.py --accounts 500 --seed 7
    python scripts/generate_training_data.py --scenario discount_creep --accounts 50

Writes data/synthetic/<name>/transactions.csv (workbook column vocabulary,
goes through the same ingestion as a real file) and labels.csv (one row per
account: label, scenario, leak dimensions, onset month, every generation
parameter). Both are gitignored — they are a function of the seed and the
calibration file, and regenerate in about a minute.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ml.calibration import CALIBRATION_PATH, Calibration
from ml.synthesize import SCENARIOS, generate_dataset

OUT_ROOT = REPO_ROOT / "data" / "synthetic"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--accounts", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--name", default=None, help="Output folder name (default: train_<seed>).")
    parser.add_argument("--scenario", choices=sorted(SCENARIOS), default=None,
                        help="Generate only this scenario (for debugging a recipe).")
    args = parser.parse_args()

    calibration = Calibration.load()
    source = CALIBRATION_PATH.relative_to(REPO_ROOT) if CALIBRATION_PATH.exists() else "built-in defaults"
    print(f"Calibration: {source}")
    print(f"  behaviour: {calibration.behaviour.source}")
    print(f"  economics: {calibration.economics.source}")

    started = time.time()
    transactions, labels = generate_dataset(args.accounts, args.seed, calibration, scenario=args.scenario)

    name = args.name or (f"{args.scenario}_{args.seed}" if args.scenario else f"train_{args.seed}")
    out = OUT_ROOT / name
    out.mkdir(parents=True, exist_ok=True)
    transactions.to_csv(out / "transactions.csv", index=False)
    labels.to_csv(out / "labels.csv", index=False)

    print(f"\nWrote {out.relative_to(REPO_ROOT)}/  in {time.time() - started:.0f}s")
    print(f"  accounts {len(labels):,}   orders {transactions['order_id'].nunique():,}   lines {len(transactions):,}")
    print("\nlabels:", labels["label"].value_counts().to_dict())
    print("scenarios:")
    for scenario, count in labels["scenario"].value_counts().sort_index().items():
        print(f"  {scenario:20} {count:5}")
    print("column variants:", labels["columns_variant"].value_counts().to_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
