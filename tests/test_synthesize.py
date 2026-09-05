"""
The synthetic generator (src/ml/synthesize.py) and feature extractor.

What must hold for the training data to be worth anything:

- Labels are true BY CONSTRUCTION. A strong discount-creep account, run
  through the real pipeline, must show discount creep; a mix-collapse account
  must show a falling high-tier share at flat revenue. If the recipe does not
  produce the pattern it is labelled with, the model learns a lie.
- Output goes through the same ingestion as a real file, columns dropped per
  variant read as "not present", and the whole thing is reproducible from a
  seed.
- The feature extractor is total: any pack, any missing block, one row of the
  same columns.

Recipes are checked with severity forced high and noise forced low, so the
assertions are about the recipe, not about a lucky draw.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ml.calibration import Calibration
from ml.features import FEATURE_NAMES, extract_features, features_frame
from ml.synthesize import (
    COLUMN_VARIANTS,
    RAW_COLUMNS,
    SCENARIOS,
    AccountSpec,
    draw_spec,
    generate_account,
    generate_dataset,
)
from pipeline.evidence import build_evidence_pack
from pipeline.ingest import ingest

from datasets import MERIDIAN_CSV


@pytest.fixture(scope="module")
def calibration():
    return Calibration.load()


def clean_spec(scenario: str, calibration, seed: int = 11, **overrides) -> AccountSpec:
    """A workbook-like account (smooth, no gaps, full columns) with the
    scenario at high severity, so the recipe's effect is unmistakable."""
    rng = np.random.default_rng(seed)
    spec = draw_spec(1, rng, calibration, scenario=scenario)
    defaults = dict(history_months=24, active_share=1.0, orders_per_active=2.0, lines_per_order=6.0,
                    units_scale=8.0, price_scale=1.0, noise_sigma=0.05, seasonal_amp=0.0,
                    seasonal_months=(), basket_stability=1.0, n_categories=8, drift_step_pct=0.0, drift_step_month=None,
                    drift_walk_sigma=0.0, shape_drift=(1.0, 1.0), tier_drift=0.0, columns_variant="full", severity=0.9, manager_changed=False,
                    return_rate=0.0, extra_gaps=0)
    defaults.update(overrides)
    for key, value in defaults.items():
        setattr(spec, key, value)
    return spec


def pack_for(spec: AccountSpec, calibration) -> dict:
    raw = generate_account(spec, calibration)
    df, _ = ingest(raw)
    return build_evidence_pack(df, spec.account_id)


# --- Shape and reproducibility ---------------------------------------------

def test_output_has_the_workbook_columns_and_ingests_cleanly(calibration):
    transactions, labels = generate_dataset(25, seed=3, calibration=calibration)
    assert list(transactions.columns) == RAW_COLUMNS
    df, report = ingest(transactions)
    assert report["rows_dropped_invalid"] == 0
    assert report["n_accounts"] == len(labels) == 25
    assert set(labels["label"]) <= {"FLAG", "NO_FLAG", "DEFER"}


def test_same_seed_same_data(calibration):
    a, la = generate_dataset(15, seed=5, calibration=calibration)
    b, lb = generate_dataset(15, seed=5, calibration=calibration)
    pd.testing.assert_frame_equal(a, b)
    pd.testing.assert_frame_equal(la, lb)


def test_adding_accounts_does_not_change_earlier_ones(calibration):
    small, _ = generate_dataset(10, seed=8, calibration=calibration)
    large, _ = generate_dataset(20, seed=8, calibration=calibration)
    first_ten = large[large["account_id"].isin(small["account_id"].unique())]
    pd.testing.assert_frame_equal(small.reset_index(drop=True), first_ten.reset_index(drop=True))


def test_every_scenario_generates(calibration):
    for scenario in SCENARIOS:
        spec = clean_spec(scenario, calibration)
        frame = generate_account(spec, calibration)
        assert not frame.empty, scenario
        assert spec.label == SCENARIOS[scenario][0]


# --- Column variants ---------------------------------------------------------

@pytest.mark.parametrize("variant", list(COLUMN_VARIANTS))
def test_dropped_columns_read_as_not_present(calibration, variant):
    spec = clean_spec("healthy", calibration, columns_variant=variant)
    pack = pack_for(spec, calibration)
    dropped = COLUMN_VARIANTS[variant][1]
    dims = pack["analysis_dimensions"]
    assert dims["tier_mix"] == ("tier" not in dropped)
    assert dims["discount"] == ("discount_pct" not in dropped)
    assert dims["margin"] == ("line_margin" not in dropped)
    # And nothing downstream fabricated a number for a missing dimension.
    if "tier" in dropped:
        assert pack["tier_mix"] is None
    if "line_margin" in dropped:
        assert pack["margin"] is None


# --- Recipes produce what they are labelled with ---------------------------

def test_mix_collapse_holds_revenue_flat_while_high_tier_falls(calibration):
    pack = pack_for(clean_spec("mix_collapse", calibration, onset_month=8), calibration)
    assert pack["tier_mix"]["high_tier_share_change_pp"] < -15, pack["tier_mix"]
    assert pack["margin"]["margin_pct_change_pp"] < -2, pack["margin"]
    assert abs(pack["overall_revenue"]["pct_change"]) < 0.25, pack["overall_revenue"]["pct_change"]


def test_premiumisation_is_the_mirror_image(calibration):
    pack = pack_for(clean_spec("premiumisation", calibration, onset_month=8), calibration)
    assert pack["tier_mix"]["high_tier_share_change_pp"] > 8, pack["tier_mix"]
    assert pack["margin"]["margin_pct_change_pp"] > 0, pack["margin"]


def test_discount_creep_moves_discount_and_margin_not_volume(calibration):
    pack = pack_for(clean_spec("discount_creep", calibration, onset_month=8), calibration)
    assert pack["discount"]["discount_pct_change_pp"] > 3, pack["discount"]
    assert pack["margin"]["margin_pct_change_pp"] < -2, pack["margin"]
    assert pack["tier_mix"]["status"] == "stable"


def test_category_defection_removes_one_whole_category(calibration):
    pack = pack_for(clean_spec("category_defection", calibration, onset_month=10), calibration)
    defected = [c for c in pack["category_changes"] if c["defected"]]
    assert len(defected) == 1, [c["category"] for c in pack["category_changes"] if c["defected"]]
    assert defected[0]["consecutive_months_at_zero"] >= 12


def test_broad_decline_is_material_and_across_the_board(calibration):
    pack = pack_for(clean_spec("broad_decline", calibration, onset_month=10), calibration)
    assert pack["revenue_decline"]["status"] == "material_decline", pack["revenue_decline"]
    assert not any(c["defected"] for c in pack["category_changes"])


def test_fragmentation_shows_in_order_shape_not_revenue(calibration):
    # Onset late enough that the baseline window holds no fragmented months —
    # otherwise the comparison is contaminated by the effect it measures.
    pack = pack_for(clean_spec("fragmentation", calibration, onset_month=16), calibration)
    order = pack["order_pattern"]
    assert order["order_frequency_pct_change"] > 0.3, order
    assert order["basket_width_pct_change"] < -0.2, order


def test_seasonal_dip_leaves_a_prior_year_echo(calibration):
    spec = clean_spec("seasonal_dip", calibration, history_months=26,
                      seasonal_amp=0.6, seasonal_months=(1, 2, 3))
    pack = pack_for(spec, calibration)
    assert pack["seasonality"]["status"] == "confirmed", pack["seasonality"]


def test_recovered_dip_is_recovered(calibration):
    pack = pack_for(clean_spec("recovered_dip", calibration, onset_month=6), calibration)
    episodes = pack["dip_episodes"]
    assert episodes and all(e["recovered"] for e in episodes), episodes
    assert pack["revenue_decline"]["status"] != "material_decline"


def test_bulk_month_is_an_outlier_not_a_trend(calibration):
    pack = pack_for(clean_spec("bulk_month", calibration, onset_month=12), calibration)
    assert len(pack["data_quality"]["outlier_months"]) >= 1
    assert pack["revenue_decline"]["status"] in ("stable", "growth", "mild_drift")


def test_thin_history_is_insufficient(calibration):
    pack = pack_for(clean_spec("thin_history", calibration, history_months=5), calibration)
    assert pack["data_sufficiency"]["label"] == "insufficient"


def test_healthy_trips_nothing_material(calibration):
    pack = pack_for(clean_spec("healthy", calibration), calibration)
    assert pack["revenue_decline"]["status"] in ("stable", "growth", "mild_drift")
    assert pack["margin"]["status"] == "stable"
    assert pack["discount"]["status"] == "stable"
    assert pack["tier_mix"]["status"] == "stable"
    assert not any(c["defected"] for c in pack["category_changes"])


def test_returns_are_netted_and_flagged(calibration):
    pack = pack_for(clean_spec("returns_heavy", calibration, return_rate=0.08), calibration)
    returns = pack["data_quality"]["returns"]
    assert returns["return_lines"] > 0
    assert returns["net_return_value"] < 0


# --- Features ------------------------------------------------------------------

def test_feature_extractor_is_total():
    row = extract_features({})
    assert list(row) == FEATURE_NAMES
    assert all(isinstance(v, float) for v in row.values())


def test_features_from_real_accounts_have_the_expected_shape():
    df, _ = ingest(str(MERIDIAN_CSV))
    packs = {a: build_evidence_pack(df, a) for a in ("ACC-101", "ACC-109", "ACC-111")}
    frame = features_frame(packs)
    assert list(frame.columns) == FEATURE_NAMES
    assert frame.loc["ACC-101", "margin_change_pp"] < -10
    assert frame.loc["ACC-111", "high_tier_change_pp"] > 8
    # Thin history: comparative features are NaN, not zero.
    assert np.isnan(frame.loc["ACC-109", "margin_change_pp"])
    assert np.isnan(frame.loc["ACC-109", "p_revenue"])
    # No status strings leaked in.
    assert all(frame.dtypes == float)
