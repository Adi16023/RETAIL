"""
The trained leak classifier, applied to one evidence pack at investigation time.

This is the deterministic step that turns the model into a SECOND OPINION for
the LLM: P(FLAG), P(NO_FLAG), P(DEFER) plus the handful of facts that drove
them, attached to the evidence pack as `model_opinion` before Stage 4 reads it.

Three rules, each load-bearing:

- The model never decides. It is one more block in the pack, alongside the
  p-values and the detector statuses. Stage 4 is told what it is, what it was
  trained on, and that it must say why when it disagrees. The verdict stays
  the LLM's, with the reasoning on record.
- The pipeline runs unchanged without it. No model file (a fresh clone, the
  gitignored models/ folder missing) means `model_opinion.available == false`
  and everything downstream proceeds exactly as before. Nothing here may
  raise into the pipeline.
- Drivers are measured, not guessed. For each feature the model actually had
  a value for, we re-predict with that one feature hidden (set to missing —
  the state a tree model treats as "not known") and report how far the leading
  probability moves. That is a local, model-specific answer to "what is this
  opinion resting on", and it costs sixty fast predictions per account.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from ml.features import FEATURE_NAMES, extract_features

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = REPO_ROOT / "models" / "leak_classifier.joblib"
MODEL_CARD_PATH = REPO_ROOT / "models" / "leak_classifier.json"

TOP_DRIVERS = 4
# Below this, the model has not really taken a side and the prompt says so.
LEANING_MIN_PROBABILITY = 0.6

# How each feature reads to a person. Anything not listed falls back to the
# raw feature name, so a new feature is never a crash — only a less friendly label.
FEATURE_LABELS = {
    "history_months": "months of history",
    "order_count": "orders in history",
    "category_count": "categories bought",
    "sufficiency_score": "data sufficiency score",
    "has_margin": "margin column present",
    "has_discount": "discount column present",
    "has_tier": "tier column present",
    "rev_pct_change": "revenue, recent vs baseline",
    "rev_cv": "month-to-month revenue noise",
    "rev_slope_pct_mo": "revenue trend per month",
    "rev_h1_h2_pct": "revenue, second half vs first half",
    "rev_cp_detected": "revenue change-point found",
    "rev_cp_pct_decline": "drop at the revenue change-point",
    "rev_cp_recovered": "change-point recovered",
    "rev_cp_sustained_months": "months since the change-point",
    "season_confirmed": "prior-year echo of the dip",
    "season_dip_months": "months in the recent dip",
    "season_echo_months": "prior-year months echoing it",
    "dip_episodes": "dip episodes in history",
    "dip_ongoing": "a dip is still ongoing",
    "dip_recovered": "an earlier dip recovered",
    "margin_base_pct": "baseline margin %",
    "margin_recent_pct": "recent margin %",
    "margin_change_pp": "margin change (pp)",
    "disc_base_pct": "baseline discount %",
    "disc_recent_pct": "recent discount %",
    "disc_change_pp": "discount change (pp)",
    "high_tier_base": "baseline high-tier share",
    "high_tier_recent": "recent high-tier share",
    "high_tier_change_pp": "high-tier share change (pp)",
    "cats_with_changepoint": "categories with a change-point",
    "cats_defected": "categories that stopped",
    "defected_baseline_share": "baseline share of stopped categories",
    "max_cat_pct_decline": "largest category decline",
    "order_freq_pct_change": "order frequency change",
    "basket_width_pct_change": "basket width change",
    "aov_pct_change": "average order value change",
    "gap_months": "months with no orders",
    "gap_share": "share of months with no orders",
    "outlier_months": "bulk-order months",
    "max_outlier_multiple": "largest bulk month vs median",
    "return_lines": "return / credit lines",
    "manager_changed": "account manager changed",
}
_DIMENSION_WORDS = {
    "revenue": "revenue", "margin": "margin", "discount": "discount", "tier_mix": "high-tier share",
    "order_frequency": "order frequency", "basket_width": "basket width",
}
for _dim, _word in _DIMENSION_WORDS.items():
    FEATURE_LABELS[f"p_{_dim}"] = f"p-value, {_word} (recent window)"
    FEATURE_LABELS[f"p_{_dim}_halves"] = f"p-value, {_word} (half vs half)"
    FEATURE_LABELS[f"p_{_dim}_trend"] = f"p-value, {_word} trend"
    FEATURE_LABELS[f"stat_{_dim}"] = f"{_word} shift statistic"
    FEATURE_LABELS[f"stat_{_dim}_trend"] = f"{_word} trend statistic"
    FEATURE_LABELS[f"sig_{_dim}"] = f"{_word} shift is significant"

OUTCOME_WORDS = {"FLAG": "leakage", "NO_FLAG": "healthy", "DEFER": "defer"}


@lru_cache(maxsize=1)
def load_model(path: str = str(MODEL_PATH)) -> dict | None:
    """The saved bundle {model, features, classes}, or None when absent or
    unreadable. Cached: the app calls this on every rerun."""
    file = Path(path)
    if not file.exists():
        return None
    try:
        import joblib
        bundle = joblib.load(file)
    except Exception:
        return None
    if not isinstance(bundle, dict) or "model" not in bundle:
        return None
    if list(bundle.get("features") or []) != FEATURE_NAMES:
        # A model trained on a different feature list would read the columns
        # wrong and say something confident about nothing. Treat it as absent.
        return None
    return bundle


@lru_cache(maxsize=1)
def model_provenance(path: str = str(MODEL_CARD_PATH)) -> dict:
    """What the model card says about where the model came from — for the
    prompt and the UI, never for the prediction."""
    file = Path(path)
    if not file.exists():
        return {}
    try:
        card = json.loads(file.read_text(encoding="utf-8"))
    except Exception:
        return {}
    held_out = card.get("held_out_synthetic") or {}
    meridian = card.get("meridian") or {}
    return {
        "trained_at": card.get("trained_at"),
        "trained_on": f"{card.get('n_train', 0):,} generated accounts with labels known by construction",
        "held_out_accuracy": held_out.get("accuracy"),
        "reference_check": (f"{meridian['correct']}/{meridian['n']} on the reference workbook"
                            if meridian else None),
    }


def _probabilities(bundle: dict, row: pd.DataFrame) -> dict[str, float]:
    proba = bundle["model"].predict_proba(row[FEATURE_NAMES])[0]
    return {str(c): float(p) for c, p in zip(bundle["model"].classes_, proba)}


def _drivers(bundle: dict, row: pd.DataFrame, leaning: str, base: float) -> list[dict]:
    """Which known facts the leading probability rests on, by how much it
    moves when each one is hidden from the model."""
    present = [f for f in FEATURE_NAMES if not np.isnan(row.iloc[0][f])]
    if not present:
        return []
    # One frame, one row per hidden feature, one predict call.
    trial = pd.concat([row] * len(present), ignore_index=True)
    for i, feature in enumerate(present):
        trial.loc[i, feature] = np.nan
    proba = bundle["model"].predict_proba(trial[FEATURE_NAMES])
    classes = list(bundle["model"].classes_)
    without = proba[:, classes.index(leaning)]
    effects = base - without                                       # > 0: knowing it pushed toward the leaning
    order = np.argsort(-np.abs(effects))
    drivers = []
    for i in order[:TOP_DRIVERS]:
        if abs(effects[i]) < 0.001:
            break
        feature = present[i]
        drivers.append({
            "feature": feature,
            "label": FEATURE_LABELS.get(feature, feature),
            "value": round(float(row.iloc[0][feature]), 4),
            "effect_on_leaning": round(float(effects[i]), 3),
            "direction": "supports" if effects[i] > 0 else "argues against",
        })
    return drivers


def model_opinion(pack: dict, bundle: dict | None = None) -> dict:
    """The second-opinion block for one evidence pack. Never raises."""
    bundle = bundle or load_model()
    unavailable = {
        "available": False,
        "note": "No trained classifier found (models/leak_classifier.joblib). "
                "Run scripts/train_leak_classifier.py to add the second opinion.",
    }
    if bundle is None:
        return unavailable
    try:
        row = pd.DataFrame([extract_features(pack)])[FEATURE_NAMES]
        probabilities = _probabilities(bundle, row)
        leaning = max(probabilities, key=probabilities.get)
        top = probabilities[leaning]
        drivers = _drivers(bundle, row, leaning, top)
    except Exception as e:                                        # pragma: no cover — defensive
        return {**unavailable, "note": f"Classifier could not score this account: {type(e).__name__}."}

    # Kept lean on purpose: the block travels to the LLM inside the evidence
    # pack, whose size is budgeted (see agent.compact_json). What the model is
    # and how to treat it is in the system prompt once, not repeated per
    # account; provenance is for people and is read by the UI directly.
    return {
        "available": True,
        "p_flag": round(probabilities.get("FLAG", 0.0), 3),
        "p_no_flag": round(probabilities.get("NO_FLAG", 0.0), 3),
        "p_defer": round(probabilities.get("DEFER", 0.0), 3),
        "leaning": OUTCOME_WORDS[leaning],
        "leaning_outcome": leaning,
        "decisive": bool(top >= LEANING_MIN_PROBABILITY),
        "top_drivers": drivers,
        # When several facts all say the same thing, hiding any ONE of them
        # barely moves the model. That redundancy is itself information — the
        # opinion does not hang on a single number — and is said outright
        # rather than left as a suspiciously short driver list.
        "drivers_note": (
            "No single fact carries this read: hiding any one of them moves the leading "
            "probability by under 5 points, because several point the same way."
            if drivers and max(abs(d["effect_on_leaning"]) for d in drivers) < 0.05 else None
        ),
    }


def classifier_enabled() -> bool:
    """The one switch. `LEAK_CLASSIFIER=off` in .env (or the environment)
    removes the second opinion everywhere — app, orchestrator, scoring
    script — without touching code or deleting the model file."""
    return os.environ.get("LEAK_CLASSIFIER", "on").strip().lower() not in ("off", "0", "false", "no")


def attach_model_opinion(pack: dict) -> dict:
    """Add `model_opinion` to an evidence pack in place and return it."""
    if not classifier_enabled():
        pack["model_opinion"] = {
            "available": False,
            "note": "The statistical second opinion is switched off (LEAK_CLASSIFIER=off).",
        }
        return pack
    pack["model_opinion"] = model_opinion(pack)
    return pack
