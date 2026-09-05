"""
What the synthetic generator is calibrated to, and where each number comes from.

Two sources, kept separate on purpose:

  BEHAVIOUR  — how B2B customers order: how often, how many lines, how many
               units, how many months they skip, how often they return things,
               how long they stay. Measured from UCI Online Retail II, which
               is REAL wholesale ordering. This is the property a generator
               cannot invent convincingly and the property the reference
               workbook lacks (its accounts never skip a month; real ones skip
               half of them).

  ECONOMICS  — what they buy and what it is worth: the product catalogue,
               list prices, margin per tier, discount levels, category mix.
               Taken from the reference workbook, because no free real dataset
               carries margin or discount at line level.

Distributions are stored as quantiles (5/25/50/75/95). The generator samples
by piecewise-linear inverse CDF, which reproduces the measured spread without
assuming a parametric family — real order rates are not Poisson and real
basket sizes are not normal, and pretending otherwise is how a generator ends
up teaching the model a shape that does not exist.

If the external file is absent, the defaults below apply. They are the values
measured on 2026-09-05; `scripts/calibrate_generator.py` refreshes them.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
CALIBRATION_PATH = REPO_ROOT / "data" / "calibration" / "generator.json"
MERIDIAN_DIR = REPO_ROOT / "data" / "meridian"
REAL_CSV = REPO_ROOT / "data" / "external" / "online_retail_II.csv"

QUANTILES = (0.05, 0.25, 0.50, 0.75, 0.95)


@dataclass
class Quantiles:
    """A distribution summarised at five points; sampled by interpolation."""
    q05: float
    q25: float
    q50: float
    q75: float
    q95: float

    def as_array(self) -> np.ndarray:
        return np.array([self.q05, self.q25, self.q50, self.q75, self.q95])

    def sample(self, rng: np.random.Generator, size: int | None = None):
        u = rng.uniform(0.05, 0.95, size=size)
        return np.interp(u, QUANTILES, self.as_array())

    @classmethod
    def of(cls, values) -> "Quantiles":
        q = np.nanquantile(np.asarray(values, dtype=float), QUANTILES)
        return cls(*[float(v) for v in q])


@dataclass
class Behaviour:
    """Real wholesale ordering, per customer with >= 12 months of history."""
    # share of months in a customer's history that contain at least one order
    active_month_share: Quantiles = field(default_factory=lambda: Quantiles(0.20, 0.33, 0.48, 0.67, 0.95))
    # orders placed in a month, given that the month is active
    orders_per_active_month: Quantiles = field(default_factory=lambda: Quantiles(1.0, 1.0, 1.2, 1.6, 3.0))
    lines_per_order: Quantiles = field(default_factory=lambda: Quantiles(4.0, 10.0, 18.0, 28.0, 60.0))
    units_per_line: Quantiles = field(default_factory=lambda: Quantiles(1.0, 2.0, 6.0, 12.0, 48.0))
    history_months: Quantiles = field(default_factory=lambda: Quantiles(12.0, 18.0, 21.0, 24.0, 25.0))
    return_line_rate: float = 0.023
    # month-to-month coefficient of variation of revenue, for the sanity gate
    monthly_revenue_cv: Quantiles = field(default_factory=lambda: Quantiles(0.6, 1.08, 1.38, 1.67, 2.6))
    source: str = "defaults measured 2026-09-05 from UCI Online Retail II"


@dataclass
class Economics:
    """The workbook's catalogue and pricing behaviour."""
    products: list = field(default_factory=list)          # dicts: product, category, tier, list_price, unit_cost
    regions: list = field(default_factory=lambda: ["North", "South", "East", "West"])
    managers: list = field(default_factory=lambda: ["Priya Nair", "Fatima Sheikh", "Arjun Mehta", "Rohan Das"])
    discount_pct: Quantiles = field(default_factory=lambda: Quantiles(0.02, 0.04, 0.05, 0.06, 0.08))
    high_tier_revenue_share: Quantiles = field(default_factory=lambda: Quantiles(0.24, 0.47, 0.50, 0.66, 0.73))
    lines_per_order: Quantiles = field(default_factory=lambda: Quantiles(4.0, 5.5, 6.3, 7.0, 8.0))
    orders_per_month: Quantiles = field(default_factory=lambda: Quantiles(1.0, 1.2, 1.25, 1.35, 1.7))
    source: str = "defaults from Quessathon workbook"


@dataclass
class Calibration:
    behaviour: Behaviour = field(default_factory=Behaviour)
    economics: Economics = field(default_factory=Economics)

    def save(self, path: Path = CALIBRATION_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path = CALIBRATION_PATH) -> "Calibration":
        """Measured calibration if present, otherwise the built-in defaults —
        so generation works on a fresh clone without the 45 MB download."""
        if not path.exists():
            calibration = cls()
            calibration.economics.products = _default_products()
            return calibration
        raw = json.loads(path.read_text(encoding="utf-8"))
        behaviour = Behaviour(**{
            k: (Quantiles(**v) if isinstance(v, dict) else v) for k, v in raw["behaviour"].items()
        })
        economics = Economics(**{
            k: (Quantiles(**v) if isinstance(v, dict) and set(v) == {"q05", "q25", "q50", "q75", "q95"} else v)
            for k, v in raw["economics"].items()
        })
        if not economics.products:
            economics.products = _default_products()
        return cls(behaviour=behaviour, economics=economics)


def _default_products() -> list[dict]:
    """The workbook's catalogue if extracted, else a representative one."""
    path = MERIDIAN_DIR / "products.csv"
    if path.exists():
        products = pd.read_csv(path)
        return products[["product", "category", "tier", "list_price", "unit_cost"]].to_dict("records")
    return [
        {"product": "PH-Precision Torque Wrench", "category": "Precision Hardware", "tier": "High", "list_price": 7300, "unit_cost": 4380},
        {"product": "PH-Bearing Puller Kit", "category": "Precision Hardware", "tier": "High", "list_price": 5600, "unit_cost": 3360},
        {"product": "DE-Thermal Imager", "category": "Diagnostic Equipment", "tier": "High", "list_price": 12500, "unit_cost": 7500},
        {"product": "PT-Cordless Drill", "category": "Power Tools", "tier": "High", "list_price": 6800, "unit_cost": 4420},
        {"product": "EA-Cable Gland Set", "category": "Electrical Accessories", "tier": "Mid", "list_price": 1100, "unit_cost": 770},
        {"product": "FF-Hex Bolt Assortment", "category": "Fasteners & Fittings", "tier": "Mid", "list_price": 900, "unit_cost": 630},
        {"product": "CN-Cutting Blade 10pk", "category": "Consumables", "tier": "Low", "list_price": 420, "unit_cost": 357},
        {"product": "SB-Safety Gloves 12pr", "category": "Safety Basics", "tier": "Low", "list_price": 360, "unit_cost": 306},
        {"product": "CM-Degreaser 5L", "category": "Cleaning & Maintenance", "tier": "Low", "list_price": 480, "unit_cost": 413},
    ]


# --- Measuring ---------------------------------------------------------------

def measure_behaviour(real_csv: Path = REAL_CSV, min_history_months: int = 12) -> Behaviour:
    """Fit the behaviour distributions from Online Retail II."""
    raw = pd.read_csv(real_csv).dropna(subset=["Customer ID"])
    raw["is_return"] = raw["Invoice"].astype(str).str.startswith("C")
    raw["revenue"] = raw["Quantity"] * raw["Price"]
    raw["month"] = pd.to_datetime(raw["InvoiceDate"]).dt.to_period("M")
    sales = raw[~raw["is_return"]]

    per_customer = sales.groupby("Customer ID").agg(
        first=("month", "min"), last=("month", "max"),
        orders=("Invoice", "nunique"), lines=("Invoice", "size"),
        active_months=("month", "nunique"),
    )
    per_customer["history"] = (per_customer["last"] - per_customer["first"]).apply(lambda d: d.n) + 1
    core = per_customer[per_customer["history"] >= min_history_months]

    active_share = core["active_months"] / core["history"]
    orders_per_active = core["orders"] / core["active_months"]
    lines_per_order = core["lines"] / core["orders"]

    core_sales = sales[sales["Customer ID"].isin(core.index)]
    units = core_sales["Quantity"].clip(lower=1)

    cvs = []
    monthly = core_sales.groupby(["Customer ID", "month"])["revenue"].sum()
    for _, s in monthly.groupby(level=0):
        s = s.droplevel(0)
        s = s.reindex(pd.period_range(s.index.min(), s.index.max(), freq="M"), fill_value=0.0)
        if s.mean() > 0:
            cvs.append(s.std() / s.mean())

    return Behaviour(
        active_month_share=Quantiles.of(active_share),
        orders_per_active_month=Quantiles.of(orders_per_active),
        lines_per_order=Quantiles.of(lines_per_order),
        units_per_line=Quantiles.of(units),
        history_months=Quantiles.of(core["history"]),
        return_line_rate=float(raw["is_return"].mean()),
        monthly_revenue_cv=Quantiles.of(cvs),
        source=f"measured from {real_csv.name}: {len(core):,} customers with >= {min_history_months} months",
    )


def measure_economics(meridian_dir: Path = MERIDIAN_DIR) -> Economics:
    """Fit pricing behaviour from the workbook's healthy-looking accounts."""
    transactions = pd.read_csv(meridian_dir / "transactions.csv")
    products = pd.read_csv(meridian_dir / "products.csv")
    transactions["month"] = pd.to_datetime(transactions["order_date"]).dt.to_period("M")
    sales = transactions[transactions["is_return"] == 0]

    per_account = sales.groupby("account_id").agg(
        orders=("order_id", "nunique"), lines=("order_id", "size"), months=("month", "nunique"))
    high = sales[sales["tier"] == "High"].groupby("account_id")["line_revenue"].sum()
    total = sales.groupby("account_id")["line_revenue"].sum()

    return Economics(
        products=products[["product", "category", "tier", "list_price", "unit_cost"]].to_dict("records"),
        regions=sorted(transactions["region"].dropna().unique().tolist()),
        managers=sorted(transactions["account_manager"].dropna().unique().tolist()),
        discount_pct=Quantiles.of(sales["discount_pct"]),
        high_tier_revenue_share=Quantiles.of((high / total).fillna(0)),
        lines_per_order=Quantiles.of(per_account["lines"] / per_account["orders"]),
        orders_per_month=Quantiles.of(per_account["orders"] / per_account["months"]),
        source=f"measured from {meridian_dir.name}: {len(per_account)} accounts",
    )
