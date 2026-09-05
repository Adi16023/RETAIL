"""
The classifier as a second opinion (src/ml/predict.py) and how the report
records agreement with it.

Two situations have to work, because the model file is gitignored and a
fresh clone has none:

- No model on disk: the pack carries `model_opinion.available == false` and
  every downstream stage behaves exactly as before.
- A model on disk: probabilities are a distribution, the leaning is the
  argmax, drivers name real features with real values, and the report says
  whether the LLM landed where the model leaned — computed from the two
  outcomes, never from the model's own account of itself.

The trained model's accuracy is NOT asserted here. That is what
scripts/train_leak_classifier.py reports, on data the model never saw.
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ml.features import FEATURE_NAMES
from ml.predict import MODEL_PATH, attach_model_opinion, load_model, model_opinion
from pipeline.evidence import build_evidence_pack
from pipeline.ingest import ingest
from pipeline.orchestrate import run_for_account
from pipeline.report import model_agreement, verdict_outcome

from datasets import MERIDIAN_CSV
from fakes import ScriptedClient, message, tool_use_block


@pytest.fixture(scope="module")
def df():
    clean, _ = ingest(str(MERIDIAN_CSV))
    return clean


def a_verdict(**overrides) -> dict:
    verdict = {
        "verdict": "healthy", "temporary_or_structural": "not_applicable",
        "confidence": "high", "defer": False, "leak_dimensions": [],
        "attributed_categories": [], "cited_facts": ["placeholder"],
        "narrative": "placeholder", "recommended_actions": [],
        "data_needed_if_deferring": [], "model_opinion_response": "",
    }
    verdict.update(overrides)
    return verdict


# --- Without a model ---------------------------------------------------------------

def test_missing_model_file_is_unavailable_not_an_error(df, tmp_path, monkeypatch):
    load_model.cache_clear()
    assert load_model(str(tmp_path / "nowhere.joblib")) is None
    # `bundle=None` falls back to whatever is on disk, so stand in for "nothing".
    from ml import predict
    monkeypatch.setattr(predict, "load_model", lambda *a, **k: None)
    opinion = predict.model_opinion(build_evidence_pack(df, "ACC-104"))
    assert opinion["available"] is False
    assert "leak_classifier" in opinion["note"]
    assert model_agreement({"model_opinion": opinion}, a_verdict()) is None


def test_the_switch_turns_the_opinion_off_everywhere(df, monkeypatch):
    monkeypatch.setenv("LEAK_CLASSIFIER", "off")
    client = ScriptedClient([message([tool_use_block("submit_verdict", a_verdict(), "t1")], stop_reason="tool_use")])
    report = run_for_account(client, df, "ACC-102")
    assert report["model_opinion"]["available"] is False
    assert "LEAK_CLASSIFIER" in report["model_opinion"]["note"]
    assert report["model_agreement"] is None
    # And the LLM was not shown any opinion beyond the off marker.
    sent = client.calls[0]["messages"][0]["content"]
    assert '"p_flag"' not in sent


def test_pipeline_runs_end_to_end_without_a_model(df, monkeypatch):
    from ml import predict
    monkeypatch.setattr(predict, "load_model", lambda *a, **k: None)
    client = ScriptedClient([message([tool_use_block("submit_verdict", a_verdict(), "t1")], stop_reason="tool_use")])
    report = run_for_account(client, df, "ACC-102")
    assert report["verdict"] == "healthy"
    assert report["model_opinion"]["available"] is False
    assert report["model_agreement"] is None


# --- Agreement is computed, not reported ------------------------------------------

def test_verdict_outcome_lets_defer_win():
    assert verdict_outcome(a_verdict()) == "NO_FLAG"
    assert verdict_outcome(a_verdict(verdict="leakage_detected")) == "FLAG"
    assert verdict_outcome(a_verdict(verdict="leakage_detected", defer=True)) == "DEFER"
    assert verdict_outcome(a_verdict(verdict="insufficient_data")) == "DEFER"


def test_agreement_compares_outcomes_and_keeps_the_agents_reason():
    opinion = {"available": True, "leaning_outcome": "FLAG", "decisive": True}
    agreed = model_agreement({"model_opinion": opinion}, a_verdict(verdict="leakage_detected"))
    assert agreed["agrees"] is True and agreed["model_decisive"] is True
    disagreed = model_agreement(
        {"model_opinion": opinion},
        a_verdict(verdict="healthy", model_opinion_response="Prior-year echo confirmed: seasonal, not a leak."),
    )
    assert disagreed["agrees"] is False
    assert disagreed["agent_outcome"] == "NO_FLAG" and disagreed["model_outcome"] == "FLAG"
    assert "seasonal" in disagreed["agent_response"]


# --- With the trained model (skipped on a clone without one) -----------------------

needs_model = pytest.mark.skipif(not MODEL_PATH.exists(), reason="no trained model in models/")


@needs_model
def test_opinion_is_a_distribution_with_named_drivers(df):
    load_model.cache_clear()
    pack = attach_model_opinion(build_evidence_pack(df, "ACC-101"))
    opinion = pack["model_opinion"]
    assert opinion["available"] is True
    total = opinion["p_flag"] + opinion["p_no_flag"] + opinion["p_defer"]
    assert abs(total - 1.0) < 0.01
    assert opinion["leaning_outcome"] in ("FLAG", "NO_FLAG", "DEFER")
    assert opinion["leaning"] in ("leakage", "healthy", "defer")
    assert opinion["top_drivers"], "a full-column account must have at least one measurable driver"
    for driver in opinion["top_drivers"]:
        assert driver["feature"] in FEATURE_NAMES
        assert driver["direction"] in ("supports", "argues against")
        assert isinstance(driver["value"], float)
    from ml.predict import model_provenance
    assert model_provenance().get("trained_on")
    # The block travels to the LLM inside a size-budgeted pack: no prose in it.
    assert "what_this_is" not in opinion and "provenance" not in opinion


@needs_model
def test_thin_history_leans_defer_on_the_features_not_the_label(df):
    load_model.cache_clear()
    opinion = attach_model_opinion(build_evidence_pack(df, "ACC-109"))["model_opinion"]
    assert opinion["leaning_outcome"] == "DEFER", opinion


@needs_model
def test_opinion_is_json_serialisable_and_reaches_the_report(df):
    import json
    client = ScriptedClient([message([tool_use_block("submit_verdict", a_verdict(verdict="leakage_detected",
                                                                                 leak_dimensions=["margin"]), "t1")],
                                     stop_reason="tool_use")])
    report = run_for_account(client, df, "ACC-101")
    json.dumps(report)                       # the evidence pack is sent to the LLM as JSON
    assert report["model_opinion"]["available"] is True
    assert report["model_agreement"]["agent_outcome"] == "FLAG"
