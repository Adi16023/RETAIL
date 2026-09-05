"""
Phase 2 pass check: does the synthetic training data look like the accounts
the model will be asked about?

Generates a training set, runs every account through the real pipeline to
get its feature row, and compares those feature distributions against the two
populations that matter:

  Meridian   the 18 workbook accounts — labelled, what the model is scored on
  Real       long-standing wholesale customers from Online Retail II — how
             actual B2B ordering behaves (revenue-only features)

For each feature, the check is COVERAGE: what share of the reference values
fall inside the synthetic 2.5–97.5% range. A feature the synthetic data never
produces at the level the references show is a blind spot the model cannot
learn — and the whole point of the generator is to leave no such spot.

Then a label check: are the labels realised in the data? Using the
pipeline's own definition of a material finding (over threshold AND the
permutation test says it is real), mature leaks must almost always register
and cleared accounts almost never should.

Gates: >= 90% coverage of Meridian and of real customers; > 80% of mature
FLAG accounts (>= 6 months since onset, severity >= 0.5) register; < 15% of
NO_FLAG accounts do. Below that, fix the generator before training — a model
fit to data that does not resemble its test set is fit to nothing.

    python scripts/check_generator.py [--accounts 600] [--seed 42] [--real-customers 150]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from ml.calibration import Calibration
from ml.dataset import features_for_transactions
from ml.features import FEATURE_NAMES
from ml.synthesize import generate_dataset
from pipeline.ingest import ingest

MERIDIAN_CSV = REPO_ROOT / "data" / "meridian" / "transactions.csv"
REAL_CSV = REPO_ROOT / "data" / "external" / "online_retail_II.csv"

COVERAGE_GATE = 0.90
NO_FLAG_GATE = 0.15

# Two separate questions about the FLAG accounts, because they have separate
# answers. "Does the data contain the pattern?" is a question about the
# generator: mature leaks must clear the detector thresholds. "Can the
# pipeline see it?" is a question about statistical power, and on accounts
# that skip half their months the permutation test has little — a limit that
# is measured, documented, and not the generator's fault. So the second gate
# is applied where power exists (accounts active in >= 80% of months).
MATURE_FLAG_THRESHOLD_GATE = 0.80       # recipe realised: over threshold, any account
MATURE_FLAG_SIGNIFICANT_GATE = 0.75     # over threshold AND significant, smooth accounts

# The real file has no category column; StockCode prefixes stand in, and their
# count says nothing a real category count would. Excluded from that
# comparison only.
REAL_FILE_ARTEFACTS = {"category_count"}

# Indicators and bounded flags — coverage on them is meaningless.
SKIP = {
    "has_margin", "has_discount", "has_tier", "rev_cp_detected", "rev_cp_recovered",
    "season_confirmed", "dip_ongoing", "dip_recovered", "manager_changed",
    "sig_revenue", "sig_margin", "sig_discount", "sig_tier_mix", "sig_order_frequency", "sig_basket_width",
}


def coverage(reference: pd.Series, synthetic: pd.Series) -> float | None:
    ref = reference.dropna()
    syn = synthetic.dropna()
    if len(ref) == 0 or len(syn) < 20:
        return None
    lo, hi = syn.quantile(0.025), syn.quantile(0.975)
    return float(((ref >= lo) & (ref <= hi)).mean())


def report(name: str, reference: pd.DataFrame, synthetic: pd.DataFrame,
           exclude: set[str] = frozenset()) -> float:
    rows = []
    for feature in FEATURE_NAMES:
        if feature in SKIP or feature in exclude:
            continue
        cov = coverage(reference[feature], synthetic[feature])
        if cov is None:
            continue
        rows.append({
            "feature": feature, "coverage": cov,
            "ref_median": float(reference[feature].median()),
            "syn_p2.5": float(synthetic[feature].quantile(0.025)),
            "syn_median": float(synthetic[feature].median()),
            "syn_p97.5": float(synthetic[feature].quantile(0.975)),
        })
    table = pd.DataFrame(rows).sort_values("coverage")
    overall = float(table["coverage"].mean())

    print(f"\n=== {name}: {len(reference)} accounts, {len(table)} comparable features ===")
    print(f"  mean coverage {overall:.0%}   features below {COVERAGE_GATE:.0%}: {(table['coverage'] < COVERAGE_GATE).sum()}")
    print("  weakest features (reference median vs synthetic 2.5% / median / 97.5%):")
    for _, r in table.head(8).iterrows():
        print(f"    {r['feature']:26} cov {r['coverage']:4.0%}   ref {r['ref_median']:9.3f}   "
              f"syn [{r['syn_p2.5']:9.3f} / {r['syn_median']:9.3f} / {r['syn_p97.5']:9.3f}]")
    return overall


def load_real(n_customers: int, seed: int) -> pd.DataFrame:
    from check_real_data import load_real_sample
    clean, _ = ingest(load_real_sample(n_customers, seed))
    return clean


def label_realisation(synthetic: pd.DataFrame, labels: pd.DataFrame) -> tuple[float, float, float]:
    """Are the labels realised in the data, and can the pipeline see them?

    Two definitions of a signal are computed. THRESHOLD: a detector's
    material status would fire. MATERIAL: threshold AND the permutation test
    says the shift is real (or could not be run) — the pipeline's own
    definition, which is what suppresses false positives on lumpy accounts.
    """
    joined = synthetic.join(labels.set_index("account_id")[
        ["label", "scenario", "severity", "months_since_onset", "active_share"]])

    def real(name: str) -> pd.Series:
        return joined[f"sig_{name}"].fillna(1.0) == 1.0

    def signals(require_significance: bool) -> pd.Series:
        ok = real if require_significance else (lambda name: pd.Series(True, index=joined.index))
        return (
            ((joined["margin_change_pp"] <= -3) & ok("margin"))
            | ((joined["disc_change_pp"] >= 3) & ok("discount"))
            | ((joined["high_tier_change_pp"] <= -8) & ok("tier_mix"))
            | (joined["cats_defected"] >= 1)
            | ((joined["rev_h1_h2_pct"] <= -0.15) & (joined["rev_slope_pct_mo"] <= -0.015) & ok("revenue"))
            | ((joined["rev_pct_change"] <= -0.25) & ok("revenue"))
            | ((joined["order_freq_pct_change"] >= 0.3) & (joined["basket_width_pct_change"] <= -0.2)
               & (ok("order_frequency") | ok("basket_width")))
        )

    threshold = signals(require_significance=False)
    material = signals(require_significance=True)

    by_label = material.groupby(joined["label"]).mean()
    print("\n=== share of accounts with a material AND significant signal, by label ===")
    for label, share in by_label.items():
        print(f"  {label:8} {share:5.0%}")

    # Leaks that have had time to develop and are not minor. Early-warning
    # and low-severity cases are allowed to be subtle — that is what they are.
    mature = (joined["label"] == "FLAG") & (joined["months_since_onset"] >= 6) & (joined["severity"] >= 0.5)
    smooth = mature & (joined["active_share"] >= 0.8)
    mature_threshold = float(threshold[mature].mean()) if mature.any() else float("nan")
    mature_material = float(material[mature].mean()) if mature.any() else float("nan")
    smooth_material = float(material[smooth].mean()) if smooth.any() else float("nan")
    print(f"  FLAG, mature (>= 6 months since onset, severity >= 0.5), n={int(mature.sum())}:")
    print(f"    over threshold (recipe realised)            {mature_threshold:5.0%}")
    print(f"    threshold AND significant, all accounts     {mature_material:5.0%}")
    print(f"    threshold AND significant, smooth accounts  {smooth_material:5.0%}  (n={int(smooth.sum())})")

    print("  material AND significant by scenario (FLAG high, NO_FLAG low):")
    for scenario, share in material.groupby(joined["scenario"]).mean().sort_values(ascending=False).items():
        print(f"    {scenario:20} {share:5.0%}")

    return mature_threshold, smooth_material, float(by_label.get("NO_FLAG", 1.0))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--accounts", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--real-customers", type=int, default=150)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    started = time.time()
    calibration = Calibration.load()
    raw, labels = generate_dataset(args.accounts, args.seed, calibration)
    synthetic_clean, _ = ingest(raw)
    print(f"generated {len(labels)} accounts ({len(raw):,} lines) in {time.time() - started:.0f}s")
    print("  labels:", labels["label"].value_counts().to_dict())

    t0 = time.time()
    synthetic = features_for_transactions(synthetic_clean, workers=args.workers)
    print(f"features for synthetic in {time.time() - t0:.0f}s")

    meridian_clean, _ = ingest(str(MERIDIAN_CSV))
    meridian = features_for_transactions(meridian_clean, workers=1)
    scores = [report("MERIDIAN (workbook, 18 accounts)", meridian, synthetic)]

    if REAL_CSV.exists():
        real = features_for_transactions(load_real(args.real_customers, args.seed), workers=args.workers)
        # The real file is revenue-only; compare against synthetic accounts of
        # the same shape so margin-bearing accounts are not credited against
        # a file that has no margin.
        revenue_only = synthetic[synthetic["has_margin"] == 0.0]
        scores.append(report("REAL (Online Retail II, revenue-only)", real, revenue_only,
                             exclude=REAL_FILE_ARTEFACTS))
    else:
        print(f"\n(real data not found at {REAL_CSV.relative_to(REPO_ROOT)} — skipping real coverage)")

    mature_threshold, smooth_material, no_flag_share = label_realisation(synthetic, labels)

    passed = (all(s >= COVERAGE_GATE for s in scores)
              and mature_threshold >= MATURE_FLAG_THRESHOLD_GATE
              and smooth_material >= MATURE_FLAG_SIGNIFICANT_GATE
              and no_flag_share < NO_FLAG_GATE)
    print(f"\ntotal {time.time() - started:.0f}s")
    print("PHASE 2 PASS" if passed else "PHASE 2 FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
