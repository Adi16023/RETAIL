"""
Synthetic B2B accounts, labelled by construction.

Every account is built in three steps:

  1. A HEALTHY BASELINE — a plausible customer who does nothing interesting.
     Ordering rhythm from real wholesale data (lumpy, gappy), products and
     prices from the workbook's catalogue.
  2. A SCENARIO injected on top — the modification IS the label, because we
     know what we did. A discount-creep account is a healthy account whose
     negotiated discount drifts upward from a known month.
  3. MESS — returns, an extra skipped month, a rep change, and columns
     dropped from the file, so the model learns to work with what is there.

Domain randomisation
--------------------
Every parameter is drawn per account across a range that spans the reference
workbook at one end (smooth, 1.25 orders every month, nothing skipped) and
real wholesale data at the other (half the months empty, orders in bursts).
A model trained on the workbook's shape alone learns the workbook; one
trained across the spread learns "a leak is a shift that is large relative to
THIS account's own behaviour" — the rule that transfers.

Realism constraints the scenarios respect
-----------------------------------------
- Mix collapse and premiumisation hold total revenue flat BY CONSTRUCTION
  (quantities are rescaled so expected revenue is unchanged). That is the
  whole point of those archetypes: nothing shows at the topline.
- Discount creep drifts by fractions of a point per month; it is never a
  jump. Real negotiated discounts move slowly.
- A seasonal dip repeats every year at the same calendar months, so the
  prior-year echo the detectors look for actually exists.
- A recovered dip ends at least six months before the history does, so it
  sits entirely in the baseline window — the trap is that it LOOKS bad in a
  plain chart and is entirely over.
- Thin-history accounts are 3–8 months long. Some of them dip in the last
  two months; the label is DEFER either way, because nothing can be
  concluded from that little.

Output is in the workbook's raw column vocabulary (order_date, product,
line_revenue, ...) so the ingestion layer is exercised exactly as it would be
by a real file. Dropped columns are NaN for that account, which ingestion
treats per account as "not present" — the same path a thin real file takes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from .calibration import Calibration

TIERS = ("High", "Mid", "Low")

# Scenario -> (label, leak dimensions). Dimensions name where the value leaks,
# in the evidence pack's vocabulary, so labels line up with verdicts.
SCENARIOS = {
    # FLAG
    "mix_collapse":        ("FLAG", ("margin", "tier_mix")),
    "discount_creep":      ("FLAG", ("discount", "margin")),
    "category_defection":  ("FLAG", ("category_mix", "revenue")),
    "broad_decline":       ("FLAG", ("revenue",)),
    "slow_bleed":          ("FLAG", ("revenue",)),
    "fragmentation":       ("FLAG", ("order_pattern",)),
    # NO FLAG
    "healthy":             ("NO_FLAG", ()),
    "seasonal_dip":        ("NO_FLAG", ()),
    "recovered_dip":       ("NO_FLAG", ()),
    "premiumisation":      ("NO_FLAG", ()),
    "growth":              ("NO_FLAG", ()),
    "bulk_month":          ("NO_FLAG", ()),
    "skipped_months":      ("NO_FLAG", ()),
    "returns_heavy":       ("NO_FLAG", ()),
    "borderline_drift":    ("NO_FLAG", ()),
    # DEFER
    "thin_history":        ("DEFER", ()),
}

# Sampling weights. Leaks are a minority of any real book — the workbook's
# own ratio is 6 of 18 — but the model needs enough of each leak type to
# learn it, so FLAG is over-represented relative to a real book and the
# classifier corrects with class weights at training time.
SCENARIO_WEIGHTS = {
    # Mix collapse and premiumisation are mirror images that differ only in
    # the SIGN of the tier shift — the one distinction the workbook says the
    # jury will test live — so both are represented a little more heavily.
    "mix_collapse": 8, "discount_creep": 7, "category_defection": 7,
    "broad_decline": 6, "slow_bleed": 6, "fragmentation": 7,
    "healthy": 12, "seasonal_dip": 6, "recovered_dip": 5, "premiumisation": 9,
    "growth": 5, "bulk_month": 4, "skipped_months": 4, "returns_heavy": 3,
    "borderline_drift": 5,
    "thin_history": 10,
}

# Which columns a file carries. "full" is the workbook; the rest are the thin
# files the Reality Test may hand over. Probabilities sum to 1.
COLUMN_VARIANTS = {
    "full":         (0.50, ()),
    "no_tier":      (0.15, ("tier",)),
    "no_pricing":   (0.12, ("list_price", "discount_pct")),
    "no_margin":    (0.11, ("unit_cost", "line_margin")),
    "revenue_only": (0.12, ("tier", "list_price", "discount_pct", "unit_cost", "line_margin")),
}

RAW_COLUMNS = [
    "order_id", "order_date", "account_id", "account_name", "region", "account_manager",
    "product", "category", "tier", "quantity", "list_price", "discount_pct",
    "unit_net_price", "unit_cost", "line_revenue", "line_margin", "is_return",
]


@dataclass
class AccountSpec:
    """Everything that determines one synthetic account. Written to labels.csv
    so any training row can be traced back to exactly how it was made."""
    account_id: str
    scenario: str
    label: str
    leak_dimensions: tuple
    history_months: int
    active_share: float           # P(a month has at least one order)
    orders_per_active: float
    lines_per_order: float
    units_scale: float            # median units per line
    price_scale: float            # business size multiplier on the catalogue
    noise_sigma: float            # lognormal sigma on monthly volume
    base_discount: float
    high_tier_target: float       # baseline share of revenue in High tier
    seasonal_amp: float           # 0 = none; else depth of the recurring dip
    seasonal_months: tuple        # calendar months (1-12) that dip
    basket_stability: float       # 1 = staples every order, tiny tail (workbook); 0 = loose (real)
    n_categories: int             # catalogue breadth this account sees (workbook = 8)
    # Benign structural variation every real account has and a stationary
    # generator lacks. Without these, a healthy synthetic account's p-values
    # are uniform and its order shape is flat, and a model learns that
    # "significant" means "leak" — then flags the workbook's healthy control.
    drift_step_pct: float         # one small permanent level shift (+-6% max)
    drift_step_month: int | None
    drift_walk_sigma: float       # slow random walk on monthly volume
    shape_drift: tuple            # (orders/active-month, lines/order) end-of-history multipliers
    tier_drift: float             # gentle linear drift of High-tier quantities (+-10% max)
    columns_variant: str
    onset_month: int | None       # first month the scenario acts (0-based)
    severity: float               # scenario-specific, 0-1
    manager_changed: bool
    return_rate: float
    extra_gaps: int
    seed: int
    # filled after generation, for diagnostics
    months_since_onset: int | None = None
    n_orders: int = 0
    n_lines: int = 0

    def as_row(self) -> dict:
        row = asdict(self)
        row["leak_dimensions"] = ";".join(self.leak_dimensions)
        row["seasonal_months"] = ";".join(str(m) for m in self.seasonal_months)
        return row


# --- Drawing an account spec -------------------------------------------------

def _pick(rng: np.random.Generator, weights: dict) -> str:
    names = list(weights)
    p = np.array([weights[n] for n in names], dtype=float)
    return str(rng.choice(names, p=p / p.sum()))


def draw_spec(index: int, rng: np.random.Generator, calibration: Calibration,
              scenario: str | None = None) -> AccountSpec:
    """Randomise one account's parameters across the workbook↔real spread."""
    b, e = calibration.behaviour, calibration.economics
    scenario = scenario or _pick(rng, SCENARIO_WEIGHTS)
    label, dims = SCENARIOS[scenario]

    # History: thin for DEFER, otherwise a wide spread. Seasonal and
    # recovered-dip scenarios need room for a prior year / a full recovery.
    if scenario == "thin_history":
        history = int(rng.integers(3, 9))
    elif scenario in ("seasonal_dip", "recovered_dip"):
        history = int(rng.integers(20, 40))
    else:
        history = int(np.clip(round(rng.normal(24, 9)), 9, 48))

    # Behaviour: a blend between the workbook's smooth regime and real
    # lumpiness. `realism` = 0 is the workbook, 1 is Online Retail II.
    #
    # A MIXTURE, not one smooth prior: some books are clean ERP exports where
    # every account orders every month (the workbook, and any file a judge
    # builds by hand), most are lumpy. A single beta put ~2% of accounts near
    # the smooth end, so the workbook's own regime — the one the model is
    # scored on — was barely represented in training.
    if rng.random() < 0.30:
        realism = float(rng.uniform(0.0, 0.15))
    else:
        realism = float(rng.beta(2, 1.5))
    active_share = float(np.interp(realism, [0, 1], [1.0, b.active_month_share.sample(rng)]))
    orders_per_active = float(np.interp(realism, [0, 1],
                                        [e.orders_per_month.sample(rng), b.orders_per_active_month.sample(rng)]))
    lines_per_order = float(np.interp(realism, [0, 1],
                                      [e.lines_per_order.sample(rng), b.lines_per_order.sample(rng)]))
    noise_sigma = float(np.interp(realism, [0, 1], [0.03, rng.uniform(0.3, 0.7)]))

    # Occasionally a much bigger customer — many orders a month.
    if rng.random() < 0.08:
        orders_per_active *= float(rng.uniform(3, 15))

    seasonal = scenario == "seasonal_dip" or rng.random() < 0.20
    seasonal_amp = float(rng.uniform(0.4, 0.7)) if seasonal else 0.0
    start = int(rng.integers(1, 13))
    seasonal_months = tuple(((start - 1 + k) % 12) + 1 for k in range(int(rng.integers(2, 4)))) if seasonal else ()

    columns_variant = _pick(rng, {k: v[0] for k, v in COLUMN_VARIANTS.items()})
    # A file that lacks EVERY column a leak lives in cannot show that leak, so
    # "FLAG" would be a label the data cannot support — noise, not a hard
    # example. Keep at least one carrying column (margin, tier or discount)
    # for the two scenarios that are invisible at revenue level.
    if scenario in ("mix_collapse", "discount_creep") and columns_variant == "revenue_only":
        columns_variant = _pick(rng, {k: v[0] for k, v in COLUMN_VARIANTS.items() if k != "revenue_only"})

    # Onset: leaks and one-off events need a baseline before them and time
    # after them. Early-warning cases (2–4 months after onset) are deliberate.
    onset = None
    if scenario == "slow_bleed":
        # A bleed is slow by definition — it needs most of the history to
        # become visible, so it starts early.
        onset = int(rng.integers(0, min(7, max(1, history - 8))))
    elif label == "FLAG":
        earliest, latest = 6, max(7, history - 2)
        onset = int(rng.integers(earliest, latest + 1)) if latest > earliest else earliest
    elif scenario == "recovered_dip":
        onset = int(rng.integers(3, max(4, history - 10)))
    elif scenario == "bulk_month":
        onset = int(rng.integers(6, max(7, history - 3)))
    elif scenario in ("growth", "borderline_drift"):
        onset = 0
    elif scenario == "thin_history" and rng.random() < 0.5:
        onset = max(1, history - 2)          # a dip in the last two months

    return AccountSpec(
        account_id=f"SYN-{index:05d}",
        scenario=scenario, label=label, leak_dimensions=dims,
        history_months=history,
        active_share=active_share,
        orders_per_active=max(1.0, orders_per_active),
        lines_per_order=max(2.0, lines_per_order),
        units_scale=float(np.interp(realism, [0, 1], [8.0, b.units_per_line.sample(rng)])),
        price_scale=float(np.exp(rng.uniform(np.log(0.4), np.log(3.0)))),
        noise_sigma=noise_sigma,
        base_discount=float(np.clip(e.discount_pct.sample(rng), 0.0, 0.3)),
        high_tier_target=float(np.clip(e.high_tier_revenue_share.sample(rng), 0.1, 0.85)),
        seasonal_amp=seasonal_amp, seasonal_months=seasonal_months,
        # Real accounts vary in WHEN they order far more than in WHAT they
        # order; stability stays high even at the lumpy end.
        basket_stability=float(np.clip(1.0 - 0.6 * realism + rng.normal(0, 0.05), 0.3, 1.0)),
        n_categories=int(np.clip(round(rng.normal(8, 3)), 3, 16)),
        drift_step_pct=float(rng.uniform(-0.06, 0.06)) if rng.random() < 0.6 else 0.0,
        drift_step_month=int(rng.integers(3, max(4, history - 3))) if history > 6 else None,
        drift_walk_sigma=float(rng.uniform(0.002, 0.010)),
        shape_drift=(float(rng.uniform(0.8, 1.25)), float(rng.uniform(0.8, 1.25))),
        tier_drift=float(rng.uniform(-0.10, 0.10)),
        columns_variant=columns_variant,
        onset_month=onset,
        severity=float(rng.uniform(0.35, 1.0)),
        manager_changed=bool(rng.random() < 0.15),
        # The workbook has 3 credit lines in 3,289 (0.09%); real wholesale
        # data has 2.3%. Blend with realism so both ends are represented.
        return_rate=(float(rng.uniform(0.05, 0.10)) if scenario == "returns_heavy"
                     else float(np.interp(realism, [0, 1], [0.001, b.return_line_rate]))),
        extra_gaps=int(rng.integers(1, 4)) if scenario == "skipped_months" else 0,
        seed=int(rng.integers(0, 2**31 - 1)),
    )


# --- Generating the transactions ------------------------------------------------

def _account_catalogue(catalogue: pd.DataFrame, n_categories: int, rng: np.random.Generator) -> pd.DataFrame:
    """The products this account can see.

    Real files range from four categories to dozens; the workbook has eight.
    A narrower catalogue drops whole categories; a wider one clones existing
    products into new categories with jittered prices, keeping the workbook's
    tier economics but not its exact breadth — so `category_count` is a real
    feature with a real spread, not a constant the model could key on.
    """
    catalogue = catalogue.reset_index(drop=True)
    categories = list(catalogue["category"].unique())
    if n_categories < len(categories):
        keep = set(rng.choice(categories, size=n_categories, replace=False))
        return catalogue[catalogue["category"].isin(keep)].reset_index(drop=True)
    extra = []
    for k in range(len(categories), n_categories):
        source_cat = str(rng.choice(categories))
        templates = catalogue[catalogue["category"] == source_cat]
        for _, row in templates.sample(n=min(len(templates), int(rng.integers(2, 4))),
                                       random_state=int(rng.integers(0, 2**31 - 1))).iterrows():
            jitter = float(np.exp(rng.normal(0.0, 0.3)))
            extra.append({"product": f"{row['product']}-{k}", "category": f"{source_cat} {k}",
                          "tier": row["tier"], "list_price": row["list_price"] * jitter,
                          "unit_cost": row["unit_cost"] * jitter})
    return pd.concat([catalogue, pd.DataFrame(extra)], ignore_index=True)

def _basket_weights(products: pd.DataFrame, high_share_target: float, rng: np.random.Generator) -> np.ndarray:
    """Per-account preference over the catalogue: Pareto-concentrated, then
    tilted so the High tier's expected revenue share hits the target."""
    weights = rng.dirichlet(np.full(len(products), 0.6))
    expected = weights * products["list_price"].to_numpy()
    high = products["tier"].to_numpy() == "High"
    current = expected[high].sum() / expected.sum() if expected.sum() > 0 else 0.5
    if 0 < current < 1:
        tilt = (high_share_target / current) * ((1 - current) / (1 - high_share_target))
        weights = np.where(high, weights * tilt, weights)
    return weights / weights.sum()


def _tier_multipliers(spec: AccountSpec, month: int) -> dict:
    """How much each tier's basket weight is scaled in `month` by the scenario."""
    mult = {t: 1.0 for t in TIERS}
    if spec.onset_month is None or month < spec.onset_month:
        return mult
    since = month - spec.onset_month
    ramp = min(1.0, since / max(1, 6 + 6 * (1 - spec.severity)))   # 6–12 months to full effect
    if spec.scenario == "mix_collapse":
        mult["High"] = 1.0 - 0.85 * spec.severity * ramp             # High fades toward 15%
        mult["Low"] = 1.0 + 2.5 * spec.severity * ramp
    elif spec.scenario == "premiumisation":
        mult["Low"] = 1.0 - 0.7 * spec.severity * ramp
        mult["High"] = 1.0 + 1.2 * spec.severity * ramp
    return mult


def _volume_multiplier(spec: AccountSpec, month: int, calendar_month: int, rng: np.random.Generator) -> float:
    """Overall quantity multiplier for a month: trend, seasonality, one-offs, noise."""
    m = 1.0
    if spec.seasonal_amp and calendar_month in spec.seasonal_months:
        m *= (1.0 - spec.seasonal_amp)
    if spec.onset_month is not None and month >= spec.onset_month:
        since = month - spec.onset_month
        if spec.scenario == "broad_decline":
            m *= (1.0 - 0.05 - 0.07 * spec.severity) ** since           # 5–12%/month compounding
        elif spec.scenario == "slow_bleed":
            m *= (1.0 - 0.015 - 0.02 * spec.severity) ** since          # 1.5–3.5%/month (ACC-112 is ~2%)
        elif spec.scenario == "borderline_drift":
            m *= (1.0 - 0.005) ** since                                  # ~ -11% over 24 months
        elif spec.scenario == "growth":
            m *= (1.0 + 0.01 + 0.02 * spec.severity) ** since
        elif spec.scenario == "recovered_dip" and since < 4:
            m *= 0.4 + 0.2 * (1 - spec.severity)
        elif spec.scenario == "bulk_month" and since == 0:
            m *= 2.0 + 2.0 * spec.severity
        elif spec.scenario == "thin_history" and spec.scenario and since >= 0:
            m *= 0.5                                                     # dip in the final months
    m *= float(np.exp(rng.normal(0.0, spec.noise_sigma)))
    return m


def _discount_for(spec: AccountSpec, month: int) -> float:
    if spec.scenario == "discount_creep" and spec.onset_month is not None and month >= spec.onset_month:
        since = month - spec.onset_month
        drift = (0.004 + 0.011 * spec.severity) * since                 # 0.4–1.5 pp per month
        return min(spec.base_discount + drift, spec.base_discount + 0.10 + 0.15 * spec.severity)
    return spec.base_discount


def _order_shape(spec: AccountSpec, month: int) -> tuple[float, float]:
    """(orders per active month, lines per order) after fragmentation."""
    if spec.scenario == "fragmentation" and spec.onset_month is not None and month >= spec.onset_month:
        since = month - spec.onset_month
        ramp = min(1.0, since / 4)
        factor = 1.0 + (0.5 + 1.5 * spec.severity) * ramp                # 1.5×–2.5× at full effect
        return spec.orders_per_active * factor, spec.lines_per_order / factor
    return spec.orders_per_active, spec.lines_per_order


def generate_account(spec: AccountSpec, calibration: Calibration,
                     start_month: pd.Period = pd.Period("2024-09", freq="M")) -> pd.DataFrame:
    """Transactions for one account, in the workbook's raw column vocabulary.

    The product model is a STABLE CORE BASKET, not a per-line lottery. A real
    account re-buys the same staples nearly every order and adds the odd extra;
    that consistency is why the workbook's accounts vary only ~6% month to
    month even though their catalogue spans a 50x price range. Sampling
    products independently per line — the obvious implementation — makes each
    month's revenue hinge on whether the ₹12,500 item happened to be drawn,
    and produced synthetic "healthy" accounts swinging 5k to 874k. Nothing
    trained on that would resemble any customer.

    Monthly volume is a BUDGET split across that month's orders, so the number
    of orders does not change how much is bought — only how it is packaged.
    That is what lets order fragmentation be its own signal: more orders,
    fewer lines each, the same total.
    """
    rng = np.random.default_rng(spec.seed)
    e = calibration.economics
    products = _account_catalogue(pd.DataFrame(e.products), spec.n_categories, rng)
    n_products = len(products)
    prices = products["list_price"].to_numpy(dtype=float) * spec.price_scale
    costs = products["unit_cost"].to_numpy(dtype=float) * spec.price_scale
    tier_of = products["tier"].to_numpy()
    category_of = products["category"].to_numpy()

    weights = _basket_weights(products, spec.high_tier_target, rng)

    # Core basket: the account's staples, bought nearly every order in a
    # quantity characteristic of this account. How "nearly" is the account's
    # basket stability: at the workbook end the most habitual items are in
    # every order (include-prob 1.0 down to 0.95 across the core) and the tail
    # is tiny; at the real end habits are looser. Stability is what keeps a
    # ₹12,500 staple from turning each month into a coin flip — without it,
    # "healthy" synthetic accounts swung 5k to 874k.
    stability = float(spec.basket_stability)
    core_size = int(np.clip(round(spec.lines_per_order * (0.6 + 0.3 * stability)), 2, n_products))
    core_rank = list(np.argsort(-weights)[:core_size])
    # The core must actually contain the tiers the account is supposed to
    # buy. With five staples drawn from twenty products, "top by weight" can
    # land all-Low or all-High by chance — and then a mix shift acts on
    # nothing while the expensive items sit in the tail as a lottery.
    if 0.15 < spec.high_tier_target < 0.85:
        for tier in ("High", "Low"):
            if not any(tier_of[i] == tier for i in core_rank):
                candidates = [i for i in np.argsort(-weights) if tier_of[i] == tier and i not in core_rank]
                if candidates:
                    core_rank[-1] = candidates[0]
    core_rank = np.array(core_rank)
    is_core = np.zeros(n_products, dtype=bool)
    is_core[core_rank] = True
    # Include-probability falls off by rank of expected REVENUE, not weight:
    # the staple that carries the account's money is the one bought every
    # single order. Ranking by weight let a cheap item be the certain one and
    # the expensive one the occasional one — and a month that happened to
    # skip it fell 30%, on a "workbook-smooth" account. At full stability the
    # spread is zero: every staple, every order, as in the workbook.
    by_revenue = core_rank[np.argsort(-(weights[core_rank] * prices[core_rank]))]
    include_prob = np.zeros(n_products)
    spread = 0.5 * (1.0 - stability)                             # 0 (workbook) .. 0.5 (real)
    for rank, idx in enumerate(by_revenue):
        include_prob[idx] = 1.0 - spread * (rank / max(1, core_size - 1))
    typical_units = rng.lognormal(np.log(spec.units_scale), 0.4, n_products)
    unit_sigma = 0.08 + 0.42 * (1.0 - stability)               # per-line quantity wobble: 8% (workbook) .. 50% (real)

    # Scale High-tier staple quantities so the core's expected High-tier
    # revenue share hits the account's target — that share is what the
    # mix-collapse and premiumisation recipes then move, and what the
    # tier-mix detector measures.
    core_value = include_prob * typical_units * prices
    high_core = is_core & (tier_of == "High")
    other_core = is_core & ~(tier_of == "High")
    if high_core.any() and other_core.any():
        current_high = core_value[high_core].sum()
        target_high = spec.high_tier_target / (1 - spec.high_tier_target) * core_value[other_core].sum()
        typical_units = np.where(high_core, typical_units * np.clip(target_high / current_high, 0.1, 10.0),
                                 typical_units)

    # Tail: everything else, bought occasionally, as SMALL add-ons — each tail
    # line is worth a fixed small fraction of a typical staple line whatever
    # its tier, so an expensive product in the tail cannot swing a month.
    tail_rate = max(0.0, spec.lines_per_order - include_prob.sum())
    tail_weights = np.where(is_core, 0.0, weights)
    staple_line_value = float(np.median((typical_units * prices)[is_core]))
    tail_units = (0.10 + 0.15 * (1.0 - stability)) * staple_line_value / prices

    # A defection removes one whole category — one the account actually buys,
    # and not one so large that its loss is an account-wide collapse (that is
    # a different scenario).
    defected_category = None
    if spec.scenario == "category_defection":
        # The category that leaves must be one a defection detector could
        # see leave: a staple (bought most orders) worth a real share of the
        # account. A 3% line bought now and then going quiet is not a
        # defection by anyone's definition, and labelling it one teaches the
        # model to find leaks in noise.
        expected_by_cat = pd.Series(include_prob * typical_units * prices).groupby(category_of).sum()
        share = expected_by_cat / expected_by_cat.sum()
        habitual = pd.Series(include_prob).groupby(category_of).max()
        candidates = share[(share >= 0.06) & (share <= 0.6) & (habitual.reindex(share.index) >= 0.7)]
        if candidates.empty:
            candidates = share[(share > 0.02) & (share <= 0.6)]
        if candidates.empty:
            candidates = share[share > 0]
        defected_category = str(rng.choice(candidates.index, p=(candidates / candidates.sum()).to_numpy()))

    def expected_revenue(unit_mult: np.ndarray, active: np.ndarray) -> float:
        core_part = (include_prob * typical_units * unit_mult * prices * active).sum()
        pool = tail_weights * active
        tail_part = (tail_rate * (pool / pool.sum() * tail_units * unit_mult * prices).sum()
                     if pool.sum() > 0 else 0.0)
        return float(core_part + tail_part)

    ones = np.ones(n_products)
    baseline_expected = expected_revenue(ones, ones)

    region = str(rng.choice(e.regions))
    managers = list(rng.choice(e.managers, size=2, replace=False))
    manager_switch = int(rng.integers(spec.history_months // 3, max(spec.history_months // 3 + 1,
                                                                    2 * spec.history_months // 3)))

    forced_gaps = set(rng.choice(np.arange(3, max(4, spec.history_months - 2)),
                                 size=min(spec.extra_gaps, max(1, spec.history_months - 5)), replace=False)) \
        if spec.extra_gaps else set()

    # Negotiated discounts drift a little over time even when nothing is
    # happening — the workbook's healthy accounts move about a point between
    # windows. A slow random walk, distinct from the deliberate creep recipe.
    discount_walk = np.cumsum(rng.normal(0.0, 0.0015, spec.history_months))
    volume_walk = np.exp(np.cumsum(rng.normal(0.0, spec.drift_walk_sigma, spec.history_months)))
    progress = np.linspace(0.0, 1.0, max(2, spec.history_months))

    rows, order_counter = [], 0
    for month in range(spec.history_months):
        period = start_month + month
        if month in forced_gaps or rng.random() > spec.active_share:
            continue

        tier_mult = _tier_multipliers(spec, month)
        unit_mult = np.array([tier_mult[t] for t in tier_of])
        unit_mult = unit_mult * np.where(tier_of == "High", 1.0 + spec.tier_drift * progress[month], 1.0)
        active = ones.copy()
        if defected_category and spec.onset_month is not None and month >= spec.onset_month:
            active = np.where(category_of == defected_category, 0.0, 1.0)

        # Hold expected revenue flat under a pure mix shift (collapse or
        # premiumisation); let it fall under a defection.
        flat_topline = spec.scenario in ("mix_collapse", "premiumisation")
        expected_now = expected_revenue(unit_mult, active)
        mix_rescale = (baseline_expected / expected_now) if (flat_topline and expected_now > 0) else 1.0

        volume = _volume_multiplier(spec, month, period.month, rng) * mix_rescale
        volume *= float(volume_walk[month])
        if spec.drift_step_month is not None and month >= spec.drift_step_month:
            volume *= 1.0 + spec.drift_step_pct
        discount = max(0.0, _discount_for(spec, month) + float(discount_walk[month]))
        orders_rate, lines_rate = _order_shape(spec, month)
        orders_rate *= 1.0 + (spec.shape_drift[0] - 1.0) * progress[month]
        lines_rate *= 1.0 + (spec.shape_drift[1] - 1.0) * progress[month]
        # How many orders this month: fixed at the account's rate for a
        # workbook-like customer, Poisson-scattered for a real-like one. The
        # workbook's accounts place 1.25 orders a month with 6% revenue noise —
        # that regularity is impossible if the count itself is a random draw.
        poisson_orders = rng.poisson(max(0.0, orders_rate - 1)) + 1
        n_orders = max(1, int(round(orders_rate + (1.0 - stability) * (poisson_orders - orders_rate))))
        # Monthly volume is a budget: dividing by realised orders means a
        # month that happened to get two orders does not buy twice as much.
        # Normalising against the EXPECTED order rate (not the base rate)
        # keeps units per line steady under fragmentation, so more orders
        # with fewer lines each still add up to the same month.
        per_line_scale = volume * (orders_rate / n_orders)
        line_factor = lines_rate / spec.lines_per_order        # < 1 under fragmentation
        manager = managers[1] if (spec.manager_changed and month >= manager_switch) else managers[0]

        for _ in range(n_orders):
            order_counter += 1
            order_id = f"{spec.account_id}-O{order_counter:05d}"
            order_date = period.to_timestamp() + pd.Timedelta(days=int(rng.integers(1, 28)))

            # Real orders vary in size far more than a fixed basket implies —
            # a top-up order one week, a full restock the next. The mood is
            # per order, and only meaningful at the loose end of stability.
            size_mood = float(np.exp(rng.normal(0.0, 0.5 * (1.0 - stability))))
            core_picks = np.flatnonzero(
                (rng.random(n_products) < np.minimum(1.0, include_prob * line_factor * size_mood)) & (active > 0)
            )
            tail_pool = tail_weights * active
            n_tail = int(rng.poisson(tail_rate * line_factor * size_mood))
            tail_picks = np.array([], dtype=int)
            if n_tail > 0 and tail_pool.sum() > 0:
                available = int((tail_pool > 0).sum())
                tail_picks = rng.choice(n_products, size=min(n_tail, available), replace=False,
                                        p=tail_pool / tail_pool.sum())
            if len(core_picks) == 0 and len(tail_picks) == 0:
                # An order always has at least its most habitual active line.
                habitual = np.flatnonzero(active > 0)
                core_picks = habitual[np.argmax(include_prob[habitual])][None]

            lines = [(i, typical_units[i]) for i in core_picks] + [(i, tail_units[i]) for i in tail_picks]
            order_rows = []
            for idx, base_units in lines:
                units = int(round(base_units * unit_mult[idx] * per_line_scale
                                  * float(np.exp(rng.normal(0.0, unit_sigma)))))
                # A tier faded to a fraction of a unit is simply not bought
                # that order — the fade acts on quantity, and quantity reaching
                # zero is how a line drops out. Never through a probability
                # that collapses whole orders to a single fallback line.
                if units < 1:
                    continue
                order_rows.append((idx, units))
            if not order_rows:
                idx = lines[0][0]
                order_rows.append((idx, 1))

            for idx, units in order_rows:
                line_discount = float(np.clip(discount + rng.normal(0, 0.008), 0.0, 0.6))
                unit_net = round(prices[idx] * (1 - line_discount), 2)
                revenue = round(unit_net * units, 2)
                margin = round(revenue - costs[idx] * units, 2)
                rows.append((order_id, order_date.date().isoformat(), spec.account_id,
                             f"Synthetic {spec.account_id[-5:]}", region, manager,
                             products.at[idx, "product"], category_of[idx], tier_of[idx], units,
                             round(prices[idx], 2), round(line_discount, 4), unit_net,
                             round(costs[idx], 2), revenue, margin, 0))

    frame = pd.DataFrame(rows, columns=RAW_COLUMNS)
    if frame.empty:
        return frame

    # Returns: some lines come back later as standalone credit orders, the
    # way the workbook records them.
    returns = frame.sample(frac=spec.return_rate, random_state=int(spec.seed % (2**32 - 1)))
    if not returns.empty:
        credits = returns.copy()
        credits["order_id"] = [f"{spec.account_id}-C{i:05d}" for i in range(len(credits))]
        credits["order_date"] = (pd.to_datetime(credits["order_date"])
                                 + pd.to_timedelta(rng.integers(5, 60, size=len(credits)), unit="D")
                                 ).dt.date.astype(str)
        for column in ("quantity", "line_revenue", "line_margin"):
            credits[column] = -credits[column]
        credits["is_return"] = 1
        frame = pd.concat([frame, credits], ignore_index=True)

    # Drop columns the file variant does not carry — NaN, which ingestion
    # treats per account as "not present".
    for column in COLUMN_VARIANTS[spec.columns_variant][1]:
        frame[column] = np.nan

    spec.n_orders = int(frame["order_id"].nunique())
    spec.n_lines = int(len(frame))
    spec.months_since_onset = (spec.history_months - 1 - spec.onset_month) if spec.onset_month is not None else None
    return frame.sort_values(["order_date", "order_id"]).reset_index(drop=True)


def generate_dataset(n_accounts: int, seed: int, calibration: Calibration | None = None,
                     scenario: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """`n_accounts` synthetic accounts and their labels. Fully determined by
    `seed`; per-account seeds are derived so adding accounts never changes
    the ones already generated."""
    calibration = calibration or Calibration.load()
    master = np.random.default_rng(seed)
    frames, labels = [], []
    for index in range(1, n_accounts + 1):
        account_rng = np.random.default_rng(master.integers(0, 2**31 - 1))
        spec = draw_spec(index, account_rng, calibration, scenario=scenario)
        frame = generate_account(spec, calibration)
        if frame.empty:
            continue
        frames.append(frame)
        labels.append(spec.as_row())
    return pd.concat(frames, ignore_index=True), pd.DataFrame(labels)
