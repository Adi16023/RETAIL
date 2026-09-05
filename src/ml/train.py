"""
Training and evaluating the leak classifier.

What the model is
-----------------
A gradient-boosted tree classifier over the evidence-pack features
(ml/features.py): P(FLAG), P(NO_FLAG), P(DEFER) for one account. It is a
SECOND OPINION that sits beside the LLM's verdict — never the verdict itself.
Trees were chosen deliberately: on a few thousand rows of tabular features
they outperform anything deeper, they handle missing values natively (and
"missing" is the DEFER signal), they train in seconds, and they can say which
features drove a prediction.

What it is trained on, and why that is the honest choice
--------------------------------------------------------
Synthetic accounts from ml/synthesize.py. The real workbook has 18 labelled
accounts with 6 positives and ONE deferral — nothing can be learned from
that; it can only be tested against. So the 18 real accounts are held out
entirely and reported as the validation result, and a second synthetic set
generated from a different seed is the test set. Training rows never include
either.

No class re-weighting is applied: the generator already over-represents leaks
relative to a real book, and "balanced" weighting pushed the model further
toward FLAG — it flagged four of the workbook's eleven healthy accounts. A real
book is mostly fine, and the model must not learn otherwise.

Evaluation is reported three ways because they answer different questions:
  cross-validation on synthetic   did it learn the patterns?
  held-out synthetic (other seed)  does it generalise within the generator?
  the 18 workbook accounts         does it transfer to the spec? (N/18, never a %)
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from ml.dataset import features_for_transactions
from ml.features import FEATURE_NAMES
from pipeline.ingest import ingest

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = REPO_ROOT / "models"
MODEL_PATH = MODELS_DIR / "leak_classifier.joblib"
MODEL_CARD_PATH = MODELS_DIR / "leak_classifier.json"

CLASSES = ["FLAG", "NO_FLAG", "DEFER"]
RANDOM_STATE = 0


# --- Data -----------------------------------------------------------------------

def load_split(folder: Path, workers: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Feature matrix and labels for one generated folder.

    Feature extraction is the slow step (a dozen permutation tests per
    account), so the matrix is cached beside the transactions and reused
    while it is newer than them.
    """
    transactions_path = folder / "transactions.csv"
    labels = pd.read_csv(folder / "labels.csv").set_index("account_id")
    cache = folder / "features.csv"
    if cache.exists() and cache.stat().st_mtime >= transactions_path.stat().st_mtime:
        features = pd.read_csv(cache, index_col="account_id")
        if list(features.columns) == FEATURE_NAMES and set(features.index) == set(labels.index):
            return features.loc[labels.index], labels
    clean, _ = ingest(str(transactions_path))
    features = features_for_transactions(clean, account_ids=list(labels.index), workers=workers)
    features.to_csv(cache)
    return features, labels


def meridian_split() -> tuple[pd.DataFrame, pd.Series]:
    """The 18 workbook accounts and their Answer Key outcomes, in the model's
    label vocabulary. Read here for EVALUATION only — nothing under
    src/pipeline/ touches the key, and this module is not part of the pipeline."""
    from validation.answer_key import expected_outcome, load_answer_key
    key = load_answer_key()
    clean, _ = ingest(str(REPO_ROOT / "data" / "meridian" / "transactions.csv"))
    features = features_for_transactions(clean, workers=1)
    outcomes = pd.Series({a: expected_outcome(row).replace(" ", "_") for a, row in key.items()})
    return features.loc[outcomes.index], outcomes


# --- Model ---------------------------------------------------------------------------

def make_model() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.05,
        max_iter=600,
        max_leaf_nodes=15,
        min_samples_leaf=20,
        l2_regularization=1.0,
        # No class re-weighting. The generator already over-represents leaks
        # (about 40% FLAG against a real book's ~30%), and "balanced" pushed the
        # argmax further toward FLAG — measured: four of eleven healthy workbook
        # accounts flagged, versus three with no weighting and no loss on the
        # leaks. A real book is mostly fine; the model must not learn otherwise.
        class_weight=None,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=30,
        random_state=RANDOM_STATE,
    )


@dataclass
class Evaluation:
    n: int
    accuracy: float
    per_class_recall: dict
    confusion: list
    misses: list = field(default_factory=list)


def evaluate(model, X: pd.DataFrame, y: pd.Series, names: pd.Index | None = None) -> Evaluation:
    pred = pd.Series(model.predict(X[FEATURE_NAMES]), index=X.index)
    cm = confusion_matrix(y, pred, labels=CLASSES)
    recall = {c: (float(cm[i, i] / cm[i].sum()) if cm[i].sum() else float("nan")) for i, c in enumerate(CLASSES)}
    misses = [{"account_id": str(a), "expected": str(y[a]), "predicted": str(pred[a])}
              for a in y.index if y[a] != pred[a]]
    return Evaluation(n=int(len(y)), accuracy=float((pred == y).mean()), per_class_recall=recall,
                      confusion=cm.tolist(), misses=misses)


def cross_validate(X: pd.DataFrame, y: pd.Series, folds: int = 5) -> Evaluation:
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    pred = pd.Series(cross_val_predict(make_model(), X[FEATURE_NAMES], y, cv=skf), index=X.index)
    cm = confusion_matrix(y, pred, labels=CLASSES)
    recall = {c: float(cm[i, i] / cm[i].sum()) for i, c in enumerate(CLASSES)}
    return Evaluation(n=int(len(y)), accuracy=float((pred == y).mean()), per_class_recall=recall,
                      confusion=cm.tolist())


def importances(model, X: pd.DataFrame, y: pd.Series, top: int = 15) -> list[dict]:
    """Permutation importance on held-out data — what the model actually
    leans on, not what it could have."""
    result = permutation_importance(model, X[FEATURE_NAMES], y, n_repeats=5,
                                    random_state=RANDOM_STATE, scoring="balanced_accuracy")
    order = np.argsort(-result.importances_mean)[:top]
    return [{"feature": FEATURE_NAMES[i], "importance": round(float(result.importances_mean[i]), 4)}
            for i in order]


# --- Orchestration -------------------------------------------------------------------

def train_and_evaluate(train_dir: Path, test_dir: Path | None, workers: int | None = None) -> dict:
    started = time.time()
    X_train, labels_train = load_split(train_dir, workers)
    y_train = labels_train["label"]
    print(f"train: {len(y_train)} accounts  {dict(y_train.value_counts())}")

    cv = cross_validate(X_train, y_train)
    print(f"cross-val: accuracy {cv.accuracy:.3f}  recall {cv.per_class_recall}")

    model = make_model().fit(X_train[FEATURE_NAMES], y_train)

    card = {
        "trained_at": time.strftime("%Y-%m-%d %H:%M"),
        "train_dir": str(train_dir.relative_to(REPO_ROOT)),
        "n_train": int(len(y_train)),
        "class_counts": {k: int(v) for k, v in y_train.value_counts().items()},
        "features": FEATURE_NAMES,
        "model": "HistGradientBoostingClassifier",
        "params": make_model().get_params(),
        "cross_validation": asdict(cv),
    }

    if test_dir is not None:
        X_test, labels_test = load_split(test_dir, workers)
        y_test = labels_test["label"]
        held_out = evaluate(model, X_test, y_test)
        print(f"held-out synthetic ({test_dir.name}): accuracy {held_out.accuracy:.3f}  recall {held_out.per_class_recall}")
        card["held_out_synthetic"] = {**asdict(held_out), "misses": len(held_out.misses)}
        card["feature_importance"] = importances(model, X_test, y_test)
        # By scenario: which archetypes does it get and which does it miss?
        pred = pd.Series(model.predict(X_test[FEATURE_NAMES]), index=X_test.index)
        by_scenario = (pred == y_test).groupby(labels_test["scenario"]).mean().sort_values()
        card["held_out_by_scenario"] = {k: round(float(v), 3) for k, v in by_scenario.items()}
        # Many FLAG accounts are early (2-5 months in) or mild by design, and
        # are not meant to be catchable yet. Recall on MATURE leaks is the fair
        # question: given a leak that has had time to show, does the model see it?
        mature = ((labels_test["label"] == "FLAG") & (labels_test["months_since_onset"] >= 6)
                  & (labels_test["severity"] >= 0.5))
        mature_recall = float((pred[mature] == "FLAG").mean()) if mature.any() else float("nan")
        card["held_out_mature_flag_recall"] = {"recall": round(mature_recall, 3), "n": int(mature.sum())}
        print(f"  mature FLAG recall (>= 6 months since onset, severity >= 0.5): {mature_recall:.0%}  (n={int(mature.sum())})")
        print("  by scenario (weakest first):")
        for scenario, acc in by_scenario.items():
            print(f"    {scenario:20} {acc:5.0%}")

    X_real, y_real = meridian_split()
    real = evaluate(model, X_real, y_real)
    print(f"MERIDIAN (18 real, never trained on): {int(real.accuracy * real.n)}/{real.n}  recall {real.per_class_recall}")
    for miss in real.misses:
        print(f"    miss: {miss['account_id']}  expected {miss['expected']}  predicted {miss['predicted']}")
    proba = pd.DataFrame(model.predict_proba(X_real[FEATURE_NAMES]), index=X_real.index, columns=model.classes_)
    card["meridian"] = {
        "correct": int(real.accuracy * real.n), "n": real.n,
        "per_class_recall": real.per_class_recall, "misses": real.misses,
        "probabilities": {a: {c: round(float(proba.loc[a, c]), 3) for c in model.classes_} for a in proba.index},
    }

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump({"model": model, "features": FEATURE_NAMES, "classes": list(model.classes_)}, MODEL_PATH)
    MODEL_CARD_PATH.write_text(json.dumps(card, indent=2, default=str), encoding="utf-8")
    print(f"\nsaved {MODEL_PATH.relative_to(REPO_ROOT)} and {MODEL_CARD_PATH.relative_to(REPO_ROOT)}  ({time.time() - started:.0f}s)")
    return card
