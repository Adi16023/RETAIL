"""
Noise-relative significance for every detection dimension.

The problem this solves, measured rather than assumed
----------------------------------------------------
Every threshold in detect.py was calibrated against the reference workbook,
whose accounts wobble by about 6% month to month. Real wholesale customers
(UCI Online Retail II, 1,569 customers with a year or more of history) wobble
by about 138% — twenty-three times more — and skip half their months
entirely. Run on 150 of those real customers, the fixed thresholds called one
in three a "material decline". A permutation test on the same customers found
that two-thirds of those flags were indistinguishable from the customer's own
ordinary lumpiness.

A fixed threshold asks "is the drop bigger than X?". The right question is
"is the drop bigger than what THIS account's noise produces by chance?" —
and that question needs no recalibration to transfer to a new dataset,
because it measures each account against itself.

The test
--------
For each dimension, take the account's raw monthly values, split them into
the trailing RECENT_MONTHS window and everything before it (the same split
every detector uses), and compute recent-minus-baseline. Then shuffle the
months thousands of times and recompute. The p-value is the share of shuffles
that produced a shift at least as large in magnitude as the observed one.
Two-sided, because direction is already reported by the detector status —
this test only answers "is it real", never "is it bad".

Statuses stay. A signal is material when the threshold says it is BIG and the
p-value says it is REAL; either alone is not enough.

Two windows, not one
--------------------
The detectors compare the trailing RECENT_MONTHS (6) against everything
before. On the workbook that is six observations, because no month is empty.
On real wholesale data 44% of months are empty, so "the last six months" is
often three data points, and no permutation test has power at n=3. Measured:
of 18 real customers whose revenue halved, the six-month window caught 2.

So every dimension is tested three ways — at the detector's recent-6 split
(so the p-value describes the same comparison the status does), at
half-vs-half of the observed months (which has power on sparse data), and as
a rank trend against time (which has power on GRADUAL declines, where both
step tests are conservative). `significant` combines them with Holm–Bonferroni,
so running three tests does not inflate the false-positive rate. All three
p-values are reported and all three are model features.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .changepoint import RECENT_MONTHS
from .monthly import monthly_table

SIGNIFICANCE_LEVEL = 0.05
N_PERMUTATIONS = 2000
SEED = 0  # fixed: the same account must always get the same p-value

# Fewer observed months than this on either side and a permutation test has
# too few distinct orderings to say anything — the p-value floor alone would
# exceed the significance level.
MIN_RECENT_OBSERVED = 2
MIN_BASELINE_OBSERVED = 3

# Dimension -> (numerator column, denominator column or None, kind).
#
# Volumes (denominator None) are compared as relative change of the window
# mean. RATES are compared as a RATIO OF SUMS over each window — total margin
# over total revenue, High-tier revenue over all revenue — which is exactly the
# quantity the detector reports as the status. The first version used the
# mean of monthly ratios instead; on a lumpy account that lets a month with
# one small order weigh the same as a month with twenty, and produced p-values
# near 1.0 for tier shifts the detector measured at -37 points. The two
# numbers now describe the same thing.
DIMENSIONS = {
    "revenue": ("revenue", None, "relative"),
    "margin": ("margin", "revenue", "ratio"),
    "discount": ("discount_weighted", "discount_weight", "ratio"),
    "tier_mix": ("tier_High", "tier_total", "ratio"),
    "order_frequency": ("orders", None, "relative"),
    # Basket width is a ratio of sums too, but expressed as RELATIVE change:
    # "-0.47 lines per order" means nothing across accounts whose baskets run
    # from 4 lines to 60, and made the feature incomparable to real data.
    "basket_width": ("lines", "orders", "ratio_relative"),
}


def _shift(recent: np.ndarray, baseline: np.ndarray, kind: str) -> np.ndarray:
    """recent-minus-baseline along axis 1; NaN where a relative change has no
    baseline to be relative to."""
    r, b = recent.mean(axis=1), baseline.mean(axis=1)
    if kind == "relative":
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(b != 0, (r - b) / np.abs(b), np.nan)
    return r - b


def _ratio_shift(num: np.ndarray, den: np.ndarray, recent_count: int, relative: bool = False) -> np.ndarray:
    """Ratio of sums, recent minus baseline, along axis 1 — as an absolute
    difference (share points) or relative to the baseline ratio."""
    with np.errstate(divide="ignore", invalid="ignore"):
        recent = num[:, -recent_count:].sum(axis=1) / den[:, -recent_count:].sum(axis=1)
        baseline = num[:, :-recent_count].sum(axis=1) / den[:, :-recent_count].sum(axis=1)
        if relative:
            return np.where(baseline != 0, (recent - baseline) / np.abs(baseline), np.nan)
    return recent - baseline


def permutation_test(values: np.ndarray, recent_count: int, kind: str,
                     n_permutations: int = N_PERMUTATIONS, seed: int = SEED,
                     denominator: np.ndarray | None = None) -> dict:
    """Two-sided permutation test of recent-vs-baseline shift.

    `values` are the observed months in time order (NaN months already
    removed); the last `recent_count` of them are the recent window. With a
    `denominator`, the statistic is the ratio of sums (kind "ratio"); months
    are permuted jointly so numerator and denominator stay paired.
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    baseline_count = n - recent_count
    insufficient = {"status": "insufficient_history", "statistic": None, "p_value": None,
                    "significant": None, "n_months": int(n), "recent_months": int(recent_count)}
    if recent_count < MIN_RECENT_OBSERVED or baseline_count < MIN_BASELINE_OBSERVED:
        return insufficient

    # A permutation test can only be as fine-grained as the number of distinct
    # ways to choose the recent window. With 5 months split 2/3 there are 10
    # arrangements, so the smallest two-sided p it can ever produce is ~0.18 —
    # it is structurally unable to find anything, and reporting a p-value
    # would dress up "cannot tell" as "tested, not significant".
    arrangements = math.comb(n, recent_count)
    if 2.0 / (arrangements + 1) >= SIGNIFICANCE_LEVEL:
        return insufficient

    rng = np.random.default_rng(seed)
    if kind in ("ratio", "ratio_relative"):
        relative = kind == "ratio_relative"
        den = np.asarray(denominator, dtype=float)
        observed = float(_ratio_shift(values[None, :], den[None, :], recent_count, relative)[0])
        if not np.isfinite(observed):
            return insufficient
        order = rng.permuted(np.broadcast_to(np.arange(n), (n_permutations, n)).copy(), axis=1)
        shifts = _ratio_shift(values[order], den[order], recent_count, relative)
    else:
        observed = float(_shift(values[None, -recent_count:], values[None, :-recent_count], kind)[0])
        if not np.isfinite(observed):
            return insufficient
        permuted = rng.permuted(np.broadcast_to(values, (n_permutations, n)).copy(), axis=1)
        shifts = _shift(permuted[:, -recent_count:], permuted[:, :-recent_count], kind)
    shifts = shifts[np.isfinite(shifts)]
    if len(shifts) == 0:
        return insufficient

    # Two-sided p as twice the smaller tail — NOT a comparison of absolute
    # values. A relative change is asymmetric: it is bounded at -100% below and
    # unbounded above, so on a lumpy, zero-heavy series a random shuffle easily
    # produces a +300% "shift". Testing |permuted| >= |observed| would count
    # every one of those against a genuine -100% collapse and declare the
    # collapse unremarkable. Measured on real wholesale data, that version
    # missed customers who had stopped ordering entirely.
    #
    # +1 in numerator and denominator: a permutation p-value of exactly zero
    # would claim certainty the sample size cannot support.
    total = len(shifts) + 1
    lower_tail = (int((shifts <= observed).sum()) + 1) / total
    upper_tail = (int((shifts >= observed).sum()) + 1) / total
    p = min(1.0, 2.0 * min(lower_tail, upper_tail))

    return {
        "status": "tested",
        "statistic": round(observed, 4),
        "p_value": round(float(p), 4),
        "significant": bool(p < SIGNIFICANCE_LEVEL),
        "n_months": int(n),
        "recent_months": int(recent_count),
    }


def significance_tests(acc_df: pd.DataFrame) -> dict:
    """Run the permutation test on every dimension the file supports.

    A dimension whose column is absent reports `unavailable` — never a
    p-value — so "could not test" stays distinguishable from "tested, not
    significant". Months with no orders are dropped from RATE series (there
    is no margin rate for a month with no sales) but kept as zeros in VOLUME
    series (a month with no orders is a real zero for revenue and order count).
    """
    table = monthly_table(acc_df)
    results: dict = {"level": SIGNIFICANCE_LEVEL, "n_permutations": N_PERMUTATIONS}
    if table.empty:
        for name in DIMENSIONS:
            results[name] = {"status": "unavailable", "statistic": None, "p_value": None,
                             "significant": None, "n_months": 0, "recent_months": 0}
        return results

    n_months = len(table)
    recent_mask = np.zeros(n_months, dtype=bool)
    recent_mask[-min(RECENT_MONTHS, n_months):] = True

    for name, (column, den_column, kind) in DIMENSIONS.items():
        unavailable = {"status": "unavailable", "statistic": None, "p_value": None,
                       "statistic_halves": None, "p_value_halves": None,
                       "statistic_trend": None, "p_value_trend": None,
                       "significant": None, "n_months": 0, "recent_months": 0}
        if column not in table.columns or table[column].isna().all():
            results[name] = unavailable
            continue
        series = table[column].to_numpy(dtype=float)
        if den_column is None:
            observed = ~np.isnan(series)
            values, den = series[observed], None
        else:
            if den_column not in table.columns:
                results[name] = unavailable
                continue
            denominator = table[den_column].to_numpy(dtype=float)
            # A rate exists only in months with a denominator: no margin rate
            # for a month with no sales, no basket width for a month with no
            # orders. Those months are dropped, not counted as zero.
            observed = ~np.isnan(series) & ~np.isnan(denominator) & (denominator != 0)
            values, den = series[observed], denominator[observed]
        recent_count = int((observed & recent_mask).sum())
        results[name] = _two_window_test(values, recent_count, kind, den)

    return results


MIN_MONTHS_FOR_TREND = 8


def _rank(a: np.ndarray) -> np.ndarray:
    return a.argsort(axis=-1).argsort(axis=-1).astype(float)


def trend_test(values: np.ndarray, n_permutations: int = N_PERMUTATIONS, seed: int = SEED,
               denominator: np.ndarray | None = None) -> dict:
    """Is there a monotonic trend? Spearman rank correlation of value against
    time, with the permutation null obtained by shuffling the values.

    A step test (recent vs baseline) is the wrong instrument for a gradual
    decline: shuffling the months of a trending series produces a null with a
    WIDE spread — the trend itself is what makes the values range widely — so
    a real 2%-a-month bleed reads as unremarkable. Measured on synthetic
    accounts: a 50% fall over two years, noise 18%, no seasonality, p = 0.24
    under the step test. Rank correlation with time asks the question the
    bleed archetype actually poses, and ranks make it indifferent to the odd
    bulk month.
    """
    values = np.asarray(values, dtype=float)
    if denominator is not None:
        den = np.asarray(denominator, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            values = np.where(den != 0, values / den, np.nan)
        values = values[np.isfinite(values)]
    n = len(values)
    insufficient = {"status": "insufficient_history", "statistic": None, "p_value": None,
                    "significant": None, "n_months": int(n)}
    if n < MIN_MONTHS_FOR_TREND:
        return insufficient

    time_rank = _rank(np.arange(n))
    time_rank = (time_rank - time_rank.mean()) / time_rank.std()

    def rho(v: np.ndarray) -> np.ndarray:
        r = _rank(v)
        r = (r - r.mean(axis=-1, keepdims=True)) / r.std(axis=-1, keepdims=True)
        return (r * time_rank).mean(axis=-1)

    observed = float(rho(values))
    rng = np.random.default_rng(seed)
    permuted = rng.permuted(np.broadcast_to(values, (n_permutations, n)).copy(), axis=1)
    shifts = rho(permuted)
    total = len(shifts) + 1
    lower = (int((shifts <= observed).sum()) + 1) / total
    upper = (int((shifts >= observed).sum()) + 1) / total
    p = min(1.0, 2.0 * min(lower, upper))
    return {"status": "tested", "statistic": round(observed, 4), "p_value": round(float(p), 4),
            "significant": bool(p < SIGNIFICANCE_LEVEL), "n_months": int(n)}


def _holm_significant(p_values: list[float], level: float = SIGNIFICANCE_LEVEL) -> bool:
    """Holm–Bonferroni: is at least one of these p-values significant with the
    family-wise error rate held at `level`? Uniformly more powerful than plain
    Bonferroni and controls the same error rate."""
    ordered = sorted(p_values)
    m = len(ordered)
    return any(p <= level / (m - i) for i, p in enumerate(ordered))


def _two_window_test(values: np.ndarray, recent_count: int, kind: str,
                     denominator: np.ndarray | None = None) -> dict:
    """Recent-6 split, half-vs-half, and a rank trend test, Holm-combined.
    See module doc for why there are three."""
    recent = permutation_test(values, recent_count, kind, denominator=denominator)
    halves = permutation_test(values, len(values) // 2, kind, denominator=denominator)
    trend = trend_test(values, denominator=denominator)

    tested = [r for r in (recent, halves, trend) if r["status"] == "tested"]
    if not tested:
        return {**recent, "statistic_halves": None, "p_value_halves": None,
                "statistic_trend": None, "p_value_trend": None}

    significant = _holm_significant([r["p_value"] for r in tested])

    primary = recent if recent["status"] == "tested" else (halves if halves["status"] == "tested" else trend)
    return {
        "status": "tested",
        "statistic": primary["statistic"],
        "p_value": primary["p_value"],
        "statistic_halves": halves["statistic"] if halves["status"] == "tested" else None,
        "p_value_halves": halves["p_value"] if halves["status"] == "tested" else None,
        "statistic_trend": trend["statistic"] if trend["status"] == "tested" else None,
        "p_value_trend": trend["p_value"] if trend["status"] == "tested" else None,
        "significant": bool(significant),
        "n_months": recent["n_months"],
        "recent_months": recent["recent_months"],
    }


def is_material(status: str | None, significance: dict | None,
                material_statuses: tuple[str, ...]) -> bool:
    """A detector finding counts as material only when the threshold says it
    is big AND the permutation test says it is real. If the test could not be
    run (thin history, missing column) the threshold alone decides — the
    honest fallback, since 'untestable' must not silently veto a finding."""
    if status not in material_statuses:
        return False
    if not significance or significance.get("p_value") is None:
        return True
    return bool(significance["significant"])
