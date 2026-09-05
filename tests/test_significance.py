"""
Noise-relative significance (src/pipeline/significance.py).

Two things are tested. First, the permutation test itself behaves: a flat
noisy series is not significant, a real step is, thin history refuses to
answer, and the same input always gives the same p-value. Second — the reason
the module exists — on the reference dataset, requiring a finding to be both
above threshold AND significant reproduces the Answer Key's split exactly.
If adding significance had silenced a real leak or let a trap through, that
partition would break here before it broke on stage.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pipeline.evidence import build_evidence_pack
from pipeline.ingest import ingest
from pipeline.monthly import monthly_table
from pipeline.significance import (
    DIMENSIONS,
    SIGNIFICANCE_LEVEL,
    is_material,
    permutation_test,
    significance_tests,
)

from datasets import MERIDIAN_CSV, thin_transactions

FLAG_ACCOUNTS = ["ACC-101", "ACC-104", "ACC-106", "ACC-107", "ACC-108", "ACC-112"]
NO_FLAG_ACCOUNTS = [
    "ACC-102", "ACC-103", "ACC-105", "ACC-110", "ACC-111",
    "ACC-113", "ACC-114", "ACC-115", "ACC-116", "ACC-117", "ACC-118",
]

# Which statuses count as "big" per dimension, and which significance result
# qualifies them. Defection has no permutation test: ten consecutive months at
# zero is not a noise question.
MATERIAL = {
    "revenue_decline": (("material_decline",), "revenue"),
    "margin": (("erosion_detected",), "margin"),
    "discount": (("creep_detected",), "discount"),
    "tier_mix": (("downgrade_detected",), "tier_mix"),
}


@pytest.fixture(scope="module")
def df():
    clean, _ = ingest(str(MERIDIAN_CSV))
    return clean


@pytest.fixture(scope="module")
def packs(df):
    return {a: build_evidence_pack(df, a) for a in sorted(df["account_id"].unique())}


def material_and_significant(pack: dict) -> list[str]:
    sig = pack["significance"]
    found = [
        name for name, (statuses, key) in MATERIAL.items()
        if is_material((pack.get(name) or {}).get("status"), sig.get(key), statuses)
    ]
    order = pack["order_pattern"]
    if order["status"] == "fragmentation_detected" and (
        sig["order_frequency"].get("significant") or sig["basket_width"].get("significant")
    ):
        found.append("order_pattern")
    if any(c["defected"] for c in pack["category_changes"]):
        found.append("category_defection")
    return found


# --- The test itself -------------------------------------------------------

def test_flat_noisy_series_is_not_significant():
    rng = np.random.default_rng(1)
    values = 100 + rng.normal(0, 15, size=24)
    result = permutation_test(values, recent_count=6, kind="relative")
    assert result["status"] == "tested"
    assert result["p_value"] > 0.2, result


def test_clear_step_down_is_significant():
    values = np.concatenate([np.full(18, 100.0), np.full(6, 50.0)])
    values += np.random.default_rng(2).normal(0, 3, size=24)
    result = permutation_test(values, recent_count=6, kind="relative")
    assert result["status"] == "tested"
    assert result["p_value"] < 0.01, result
    assert result["significant"] is True
    assert result["statistic"] < -0.4


def test_absolute_kind_measures_rate_shift_in_points():
    values = np.concatenate([np.full(18, 0.30), np.full(6, 0.18)])
    result = permutation_test(values, recent_count=6, kind="absolute")
    assert result["significant"] is True
    assert abs(result["statistic"] - (-0.12)) < 1e-6


def test_thin_history_refuses_to_answer():
    result = permutation_test(np.array([100.0, 90, 80, 70, 60]), recent_count=5, kind="relative")
    assert result["status"] == "insufficient_history"
    assert result["p_value"] is None


def test_p_value_is_never_exactly_zero():
    """A permutation p of 0 would claim certainty the sample cannot support."""
    values = np.concatenate([np.full(18, 100.0), np.full(6, 0.0)])
    result = permutation_test(values, recent_count=6, kind="relative")
    assert result["p_value"] > 0


def test_same_input_same_p_value():
    """Verdicts are cached by data fingerprint; a p-value that changed between
    runs on identical data would make the cache lie."""
    values = 100 + np.random.default_rng(3).normal(0, 20, size=24)
    a = permutation_test(values, 6, "relative")
    b = permutation_test(values, 6, "relative")
    assert a == b


# --- On the reference dataset ----------------------------------------------

def test_every_dimension_is_tested_on_the_full_file(packs):
    sig = packs["ACC-101"]["significance"]
    for name in DIMENSIONS:
        assert sig[name]["status"] == "tested", (name, sig[name])
        assert 0 < sig[name]["p_value"] <= 1


def test_p_values_sit_beside_the_statuses_they_qualify(packs):
    pack = packs["ACC-101"]
    for block in ("revenue_decline", "margin", "discount", "tier_mix"):
        assert "p_value" in pack[block], block
        assert "significant" in pack[block], block
    assert "p_value_frequency" in pack["order_pattern"]
    assert "p_value_basket_width" in pack["order_pattern"]


def test_the_two_hidden_leaks_are_significant_where_it_matters(packs):
    """ACC-101 leaks through margin and tier; ACC-107 through discount and
    margin. Those exact dimensions must be significant — and revenue, flat on
    both, must not be MATERIAL.

    Not "must not be significant": ACC-107's revenue dipped 2.6% on an account
    whose month-to-month noise is about 2%. That dip is statistically real, and
    the test is right to say so. It is also far below the materiality
    threshold, so the detector says `stable` and nothing flags. Significance
    answers "did it happen"; the threshold answers "does it matter"; a
    material finding needs both.
    """
    a = packs["ACC-101"]
    assert a["significance"]["margin"]["significant"] and a["significance"]["tier_mix"]["significant"]
    assert a["revenue_decline"]["status"] != "material_decline"

    b = packs["ACC-107"]
    assert b["significance"]["margin"]["significant"] and b["significance"]["discount"]["significant"]
    assert b["revenue_decline"]["status"] != "material_decline"


def test_collapse_to_zero_on_a_lumpy_series_is_significant():
    """The asymmetry trap. Relative change is bounded at -100% but unbounded
    above, so a zero-heavy series shuffles into huge positive 'shifts' by
    chance. A test that compared absolute values would count those against a
    genuine collapse and miss a customer who has stopped ordering. Real
    wholesale data exposed exactly this."""
    rng = np.random.default_rng(4)
    baseline = np.where(rng.random(18) < 0.4, 0.0, rng.uniform(80, 120, 18))  # lumpy, ~40% empty
    values = np.concatenate([baseline, np.zeros(6)])                           # then gone
    result = permutation_test(values, recent_count=6, kind="relative")
    assert result["statistic"] == -1.0
    assert result["significant"] is True, result


def test_sparse_decline_is_caught_by_the_halves_window():
    """Real wholesale customers skip ~44% of months, so 'the last six months'
    is often two or three observations — a window with no statistical power.
    The second (half-vs-half) window exists for exactly this: a customer whose
    revenue fell 70% over the second year, but whose last six months contain
    only two orders, must still register. Measured before this fix: the
    six-month window alone caught 2 of 18 such real customers; with halves, 5.

    Deterministic fixture — no seed — so the assertion is about the method,
    not about a lucky draw.
    """
    first_year = [100, 0, 110, 95, 0, 105, 100, 0, 90, 105, 100, 0]     # 9 orders
    second_year = [30, 0, 35, 0, 28, 32, 0, 0, 30, 0, 0, 27]            # 6 orders, last six hold 2
    values = np.array(first_year + second_year, dtype=float)
    result = significance_tests_for_series(values)

    assert result["p_value_halves"] < 0.025, result
    assert result["significant"] is True, result
    # The six-month window sees almost nothing here — that is the point.
    assert result["p_value"] > result["p_value_halves"], result


def significance_tests_for_series(values: np.ndarray) -> dict:
    """Drive the two-window test directly on a revenue-shaped series."""
    from pipeline.significance import _two_window_test
    recent_count = min(6, len(values))
    return _two_window_test(np.asarray(values, dtype=float), recent_count, "relative")


def test_gradual_bleed_is_caught_by_the_trend_test():
    """A 2%-a-month decline over two years with 15% noise. Both step tests
    are conservative here — shuffling a trending series gives a wide null —
    while a rank correlation with time sees the monotone slide immediately.
    This is the slow-bleed archetype (ACC-112) on a noisier account."""
    from pipeline.significance import trend_test
    rng = np.random.default_rng(7)
    values = 100 * (0.98 ** np.arange(24)) * np.exp(rng.normal(0, 0.15, 24))
    result = trend_test(values)
    assert result["status"] == "tested"
    assert result["statistic"] < -0.5, result
    assert result["p_value"] < 0.01, result
    combined = significance_tests_for_series(values)
    assert combined["p_value_trend"] < 0.01
    assert combined["significant"] is True


def test_trend_test_is_quiet_on_a_flat_noisy_series():
    from pipeline.significance import trend_test
    rng = np.random.default_rng(8)
    values = 100 + rng.normal(0, 15, 24)
    result = trend_test(values)
    assert result["p_value"] > 0.1, result


def test_lumpy_series_with_no_trend_is_not_significant():
    """The other half of the same trap: half-empty months with no trend must
    NOT read as a real shift just because individual months swing wildly."""
    rng = np.random.default_rng(5)
    values = np.where(rng.random(24) < 0.5, 0.0, rng.uniform(80, 120, 24))
    result = permutation_test(values, recent_count=6, kind="relative")
    assert result["status"] == "tested"
    assert result["p_value"] > 0.1, result


def test_premiumisation_is_significant_but_direction_says_favourable(packs):
    """Significance answers 'is it real', never 'is it bad'. ACC-111's mix
    shift is as real as ACC-101's — the status carries the direction."""
    pack = packs["ACC-111"]
    assert pack["significance"]["tier_mix"]["significant"]
    assert pack["tier_mix"]["status"] == "premiumisation_detected"
    assert material_and_significant(pack) == []


def test_thin_history_account_has_no_p_values(packs):
    sig = packs["ACC-109"]["significance"]
    for name in DIMENSIONS:
        assert sig[name]["status"] == "insufficient_history", name


@pytest.mark.parametrize("account_id", FLAG_ACCOUNTS)
def test_every_flag_account_keeps_a_material_significant_signal(packs, account_id):
    """Adding significance must not silence a real leak."""
    assert material_and_significant(packs[account_id]), account_id


@pytest.mark.parametrize("account_id", NO_FLAG_ACCOUNTS)
def test_no_cleared_account_gains_a_material_significant_signal(packs, account_id):
    signals = material_and_significant(packs[account_id])
    assert signals == [], f"{account_id}: {signals}"


def test_partition_matches_the_answer_key(packs):
    flagged = {a for a, p in packs.items() if material_and_significant(p)}
    assert flagged == set(FLAG_ACCOUNTS)


# --- Degraded input and the shared table -----------------------------------

def test_unavailable_dimensions_report_unavailable_not_a_p_value():
    clean, _ = ingest(thin_transactions())
    sig = build_evidence_pack(clean, "ACC-101")["significance"]
    assert sig["discount"]["status"] == "unavailable"
    assert sig["tier_mix"]["status"] == "unavailable"
    assert sig["margin"]["status"] == "tested"
    assert sig["revenue"]["status"] == "tested"


def test_monthly_table_keeps_gap_months_as_zero_revenue_rows(df):
    table = monthly_table(df[df["account_id"] == "ACC-117"])
    assert len(table) == 24
    assert (table["revenue"] == 0).sum() == 1
    # A month with no orders has no margin RATE — NaN, not zero.
    assert table.loc[table["orders"] == 0, "margin_pct"].isna().all()


def test_monthly_table_leaves_absent_margin_empty_not_zero():
    clean, _ = ingest(thin_transactions(with_margin=False))
    table = monthly_table(clean[clean["account_id"] == "ACC-101"])
    assert table["margin"].isna().all()
    assert table["margin_pct"].isna().all()


def test_dashboard_frame_reads_the_same_table(df):
    """The UI's monthly frame must be the pipeline table plus display
    columns — the chart, the table view and the p-value share one source."""
    from ui.metrics import monthly_frame
    ui_frame = monthly_frame(df, "ACC-101")
    table = monthly_table(df[df["account_id"] == "ACC-101"])
    for column in ("revenue", "orders", "lines", "margin_pct", "discount_pct", "high_tier_share"):
        np.testing.assert_allclose(
            ui_frame[column].to_numpy(dtype=float), table[column].to_numpy(dtype=float),
            equal_nan=True, err_msg=column,
        )
    assert set(ui_frame["window"]) == {"Baseline", "Recent"}
