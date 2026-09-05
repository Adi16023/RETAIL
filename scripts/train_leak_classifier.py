"""
Trains the leak classifier on generated accounts and reports how it does on
data it never saw — a held-out synthetic set from another seed, and the 18
workbook accounts.

    python scripts/train_leak_classifier.py
    python scripts/train_leak_classifier.py --train data/synthetic/train_42 --test data/synthetic/test_7

Writes models/leak_classifier.joblib (the model) and models/leak_classifier.json
(the model card: what it was trained on, cross-validation, held-out results,
the Meridian result account by account, and permutation feature importances).
Both are gitignored and rebuilt by this script.

The Meridian result is reported as N/18. It is the only real-data number and
it is deliberately not a percentage: eighteen accounts is a spot check, not a
benchmark, and presenting it otherwise would overstate what is known.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ml.train import train_and_evaluate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", default="data/synthetic/train_42")
    parser.add_argument("--test", default="data/synthetic/test_7")
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    train_dir = REPO_ROOT / args.train
    test_dir = REPO_ROOT / args.test if args.test else None
    for folder in (train_dir, test_dir):
        if folder is not None and not (folder / "transactions.csv").exists():
            print(f"{folder.relative_to(REPO_ROOT)} not found — run scripts/generate_training_data.py first.",
                  file=sys.stderr)
            return 2

    train_and_evaluate(train_dir, test_dir, workers=args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
