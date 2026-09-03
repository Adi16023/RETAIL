"""
Stage 1: Data Processing Layer — ingestion, column mapping, validation,
and data-sufficiency scoring.

This is the only stage that ever touches a raw, possibly-unfamiliar CSV
(the Reality Test file included). It must never crash on an unexpected
schema — it either resolves the input to the canonical schema below, or
raises IngestionError naming exactly what's missing. The LLM layer never
sees a raw file; it only sees the validation report and the evidence pack
built downstream from clean_df.

Spec: data.md, "Handling unseen data (Reality Test ingestion)".

The canonical schema is modelled on the official Quessathon workbook
(`Quessathon_Revenue_Leakage_Dataset.xlsx` -> data/meridian/transactions.csv),
whose 17 columns are the richest input the pipeline is expected to see.
Everything beyond REQUIRED_FIELDS is *enrichment*: if it is present the
downstream stages light up extra detection dimensions (margin erosion,
discount creep, tier mix, returns), and if it is absent they degrade to the
revenue-only analysis the earlier synthetic data supported. That asymmetry
is deliberate — the Reality Test file may carry fewer columns, and missing
enrichment must narrow the analysis, never break it.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

CANONICAL_COLUMNS = [
    "order_id", "account_id", "account_name", "region", "account_manager",
    "date", "product_id", "category", "tier",
    "quantity", "list_price", "discount_pct", "unit_price", "unit_cost",
    "revenue", "margin", "is_return",
]

# Fields the pipeline cannot function without, once revenue is resolvable
# (revenue itself is required but may be *derived* from unit_price*quantity).
REQUIRED_FIELDS = ["account_id", "date", "product_id", "category"]

# Never required. Each unlocks a detection dimension when present — see
# _derive_missing_fields for what is reconstructed from what, and
# ValidationReport.analysis_dimensions for what the caller actually gets.
OPTIONAL_FIELDS = [
    "order_id", "account_name", "region", "account_manager", "tier",
    "list_price", "discount_pct", "unit_cost", "margin", "is_return",
]

# Ordered: earlier entries claim a raw column first, so a more specific
# field (list_price) is resolved before a more generic one (unit_price)
# could fuzzily absorb it.
SYNONYMS = {
    "account_id": ["account_id", "customer_id", "client_id", "customer", "client", "cust_id"],
    "account_name": ["account_name", "customer_name", "client_name", "account", "customer_account"],
    "region": ["region", "zone", "territory", "geo", "area"],
    "account_manager": ["account_manager", "sales_rep", "rep", "owner", "salesperson",
                        "relationship_manager", "am"],
    "date": ["date", "order_date", "txn_date", "transaction_date", "purchase_date", "invoice_date"],
    "product_id": ["product_id", "product", "sku", "item_id", "item_code", "product_code",
                   "product_name", "item"],
    "category": ["category", "product_category", "category_name", "segment", "product_group"],
    "tier": ["tier", "value_tier", "product_tier", "price_tier", "value_band", "band"],
    "quantity": ["quantity", "qty", "units", "unit_count", "no_of_units"],
    "list_price": ["list_price", "mrp", "gross_price", "catalog_price", "catalogue_price",
                   "list_rate"],
    "discount_pct": ["discount_pct", "discount", "discount_percent", "discount_percentage",
                     "disc_pct", "discount_rate"],
    "unit_price": ["unit_price", "unit_net_price", "net_unit_price", "net_price", "price",
                   "unitprice", "price_per_unit", "rate", "selling_price"],
    "unit_cost": ["unit_cost", "cost", "cost_price", "unit_cogs", "cogs", "buy_price"],
    "revenue": ["revenue", "line_revenue", "amount", "net_sales", "total", "line_total",
                "sales_amount", "total_amount", "net_amount", "net_revenue"],
    "margin": ["margin", "line_margin", "profit", "gross_margin", "margin_amount",
               "gross_profit", "line_profit"],
    "is_return": ["is_return", "return_flag", "returned", "is_credit", "credit_note",
                  "is_refund"],
    "order_id": ["order_id", "order_number", "order_no", "invoice_id", "invoice_number"],
}

FUZZY_CUTOFF = 0.75

# Canonical value-tier labels, highest-value first. The workbook uses
# High/Mid/Low; other files may say Premium/Standard/Commodity.
TIER_ORDER = ["High", "Mid", "Low"]

TIER_SYNONYMS = {
    "High": {"high", "high value", "high-value", "premium", "a", "top", "gold", "tier 1", "t1"},
    "Mid": {"mid", "medium", "mid value", "mid-value", "standard", "b", "middle", "silver",
            "tier 2", "t2"},
    "Low": {"low", "low value", "low-value", "commodity", "basic", "c", "bottom", "bronze",
            "tier 3", "t3"},
}


class IngestionError(Exception):
    """Raised when the input cannot be resolved to a usable transaction schema.

    Always carries a message naming exactly what's missing — a clear error
    beats a stack trace when this runs live on stage.
    """


@dataclass
class ValidationReport:
    column_mapping: dict = field(default_factory=dict)
    fields_found: list = field(default_factory=list)
    fields_inferred: list = field(default_factory=list)
    fields_missing_optional: list = field(default_factory=list)
    rows_total_raw: int = 0
    rows_dropped_invalid: int = 0
    row_drop_reasons: dict = field(default_factory=dict)
    date_range: tuple = (None, None)
    n_accounts: int = 0
    n_orders: int = 0
    n_categories: int = 0
    order_id_inferred: bool = False
    margin_present: bool = False
    unit_economics_present: bool = False
    tier_present: bool = False
    tier_values: list = field(default_factory=list)
    tier_values_unrecognized: list = field(default_factory=list)
    discount_present: bool = False
    return_lines: int = 0
    return_value: float = 0.0
    # Which detection dimensions the resolved columns can actually support.
    # Stage 3 reads this rather than probing for columns itself, and the
    # evidence pack passes it to Stage 4 so the model knows what it is NOT
    # being shown (e.g. "margin unavailable" is very different from
    # "margin stable").
    analysis_dimensions: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["date_range"] = [
            None if self.date_range[0] is None else str(self.date_range[0].date()),
            None if self.date_range[1] is None else str(self.date_range[1].date()),
        ]
        return _json_safe(d)


def _json_safe(value):
    """Recursively convert numpy scalar types to native Python types so the
    report can be safely json.dumps'd as part of the evidence pack."""
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _normalize_name(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_").replace("-", "_")


def _map_columns(columns: list[str]) -> dict[str, str | None]:
    """Map raw input columns to canonical field names.

    Exact synonym match first (case/whitespace-insensitive), then fuzzy
    fallback against whatever raw columns remain unmapped.
    """
    normalized_to_raw = {_normalize_name(c): c for c in columns}
    mapping: dict[str, str | None] = {}
    used_raw = set()

    for canonical, synonyms in SYNONYMS.items():
        match = None
        for syn in synonyms:
            if syn in normalized_to_raw and normalized_to_raw[syn] not in used_raw:
                match = normalized_to_raw[syn]
                break
        mapping[canonical] = match
        if match:
            used_raw.add(match)

    # Fuzzy fallback for canonical fields still unmapped.
    remaining_raw = [c for c in columns if c not in used_raw]
    remaining_normalized = {_normalize_name(c): c for c in remaining_raw}
    for canonical in mapping:
        if mapping[canonical] is not None:
            continue
        close = difflib.get_close_matches(
            canonical, remaining_normalized.keys(), n=1, cutoff=FUZZY_CUTOFF
        )
        if close:
            raw_col = remaining_normalized.pop(close[0])
            mapping[canonical] = raw_col
            remaining_raw.remove(raw_col)

    return mapping


def _normalize_tier(value) -> str | None:
    """Map a raw tier label onto the canonical High/Mid/Low vocabulary.
    Returns None for anything unrecognized — the caller records those rather
    than guessing, since a mis-mapped tier would silently corrupt the
    high-tier-share signal that the mix-collapse archetype turns on."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    key = str(value).strip().lower()
    if not key:
        return None
    for canonical, aliases in TIER_SYNONYMS.items():
        if key in aliases:
            return canonical
    return None


def _derive_missing_fields(df: pd.DataFrame, report: ValidationReport) -> pd.DataFrame:
    if "revenue" not in df.columns and {"unit_price", "quantity"}.issubset(df.columns):
        df["revenue"] = df["unit_price"] * df["quantity"]
        report.fields_inferred.append("revenue (unit_price * quantity)")
    elif "unit_price" not in df.columns and {"revenue", "quantity"}.issubset(df.columns):
        with np.errstate(divide="ignore", invalid="ignore"):
            df["unit_price"] = np.where(df["quantity"] != 0, df["revenue"] / df["quantity"], np.nan)
        report.fields_inferred.append("unit_price (revenue / quantity)")

    # Margin is the single most valuable enrichment column: two of the
    # workbook's archetypes (hidden mix collapse, discount creep) are
    # invisible in revenue and only show up in margin. Reconstruct it from
    # unit economics wherever the file gives us the pieces.
    if "margin" not in df.columns and {"unit_cost", "quantity", "revenue"}.issubset(df.columns):
        df["margin"] = df["revenue"] - (df["unit_cost"] * df["quantity"])
        report.fields_inferred.append("margin (revenue - unit_cost * quantity)")

    if "discount_pct" not in df.columns and {"list_price", "unit_price"}.issubset(df.columns):
        with np.errstate(divide="ignore", invalid="ignore"):
            df["discount_pct"] = np.where(
                df["list_price"] > 0, 1.0 - (df["unit_price"] / df["list_price"]), np.nan
            )
        report.fields_inferred.append("discount_pct (1 - unit_price / list_price)")

    # A return/credit line is identified by its sign. Returns must be NETTED
    # into revenue, not dropped — the workbook's returns archetype exists
    # precisely to catch a pipeline that reads a credit note as leakage.
    if "is_return" not in df.columns:
        negative = pd.Series(False, index=df.index)
        if "revenue" in df.columns:
            negative |= df["revenue"] < 0
        if "quantity" in df.columns:
            negative |= df["quantity"] < 0
        df["is_return"] = negative.astype(int)
        if negative.any():
            report.fields_inferred.append("is_return (negative quantity or revenue)")
    else:
        df["is_return"] = (
            pd.to_numeric(df["is_return"], errors="coerce").fillna(0).astype(int).clip(0, 1)
        )

    if "tier" in df.columns:
        raw_tiers = df["tier"]
        normalized = raw_tiers.map(_normalize_tier)
        # pandas turns the None returned for an unrecognized label into NaN,
        # so the "did this map?" test has to be pd.isna, not `is None` — with
        # the latter, an unknown tier vocabulary is silently discarded and
        # nothing in the report says so.
        unrecognized = sorted({
            str(value) for value, mapped in zip(raw_tiers, normalized)
            if pd.isna(mapped) and not pd.isna(value)
        })
        if unrecognized:
            report.tier_values_unrecognized = unrecognized
            report.fields_inferred.append(
                f"tier: {len(unrecognized)} unrecognized label(s) {unrecognized} left unmapped — "
                "tier-mix analysis falls back to revenue-derived category value ranking"
            )
        df["tier"] = normalized

    if "order_id" not in df.columns:
        df["order_id"] = [f"ROW-{i}" for i in range(len(df))]
        report.order_id_inferred = True
        report.fields_inferred.append("order_id (synthetic, one order per row — order-size analysis degraded)")

    for optional_col in ("account_name", "region", "account_manager", "tier", "list_price",
                         "discount_pct", "unit_price", "quantity", "unit_cost", "margin"):
        if optional_col not in df.columns:
            df[optional_col] = np.nan
            report.fields_missing_optional.append(optional_col)

    return df


EXCEL_SUFFIXES = (".xlsx", ".xlsm", ".xls")

# When an Excel workbook is handed over without a sheet name, prefer a tab
# that looks like transaction lines. The official workbook ships five tabs,
# and picking the first one ("READ ME") would be an unhelpful crash.
TRANSACTION_SHEET_HINTS = ("transaction", "line", "order", "sales", "data")


def _looks_like_excel(source) -> bool:
    name = getattr(source, "name", None) or (source if isinstance(source, str) else "")
    return str(name).lower().endswith(EXCEL_SUFFIXES)


def _pick_transaction_sheet(sheet_names: list[str]) -> str:
    for hint in TRANSACTION_SHEET_HINTS:
        for name in sheet_names:
            if hint in name.strip().lower():
                return name
    return sheet_names[0]


def _read_source(source, sheet_name: str | None = None) -> pd.DataFrame:
    if isinstance(source, pd.DataFrame):
        return source.copy()

    if sheet_name is not None or _looks_like_excel(source):
        try:
            book = pd.ExcelFile(source)
            chosen = sheet_name if sheet_name is not None else _pick_transaction_sheet(book.sheet_names)
            if chosen not in book.sheet_names:
                raise IngestionError(
                    f"Sheet {chosen!r} not found in workbook. Available sheets: {book.sheet_names}."
                )
            return book.parse(chosen)
        except IngestionError:
            raise
        except Exception as e:
            raise IngestionError(f"Could not read '{source}' as an Excel workbook: {e}") from e

    try:
        return pd.read_csv(source)
    except Exception as e:
        raise IngestionError(f"Could not read '{source}' as CSV: {e}") from e


def ingest(source: str | pd.DataFrame, sheet_name: str | None = None) -> tuple[pd.DataFrame, dict]:
    """Ingest a raw transaction file (or DataFrame) and resolve it to the
    canonical schema.

    `source` may be a CSV path, an Excel workbook path/handle (a sheet is
    picked by name, or guessed from the tab names), or a DataFrame.

    Returns (clean_df, validation_report_dict). Raises IngestionError with
    a specific message if the input cannot be resolved.
    """
    raw = _read_source(source, sheet_name=sheet_name)

    if raw.empty or len(raw.columns) == 0:
        raise IngestionError("Input file has no columns / is empty.")

    report = ValidationReport(rows_total_raw=len(raw))

    mapping = _map_columns(list(raw.columns))
    report.column_mapping = {k: v for k, v in mapping.items()}

    df = pd.DataFrame()
    for canonical, raw_col in mapping.items():
        if raw_col is not None:
            df[canonical] = raw[raw_col]
            report.fields_found.append(canonical)

    missing_required = [f for f in REQUIRED_FIELDS if f not in df.columns]
    has_revenue_path = "revenue" in df.columns or {"unit_price", "quantity"}.issubset(df.columns)
    if missing_required or not has_revenue_path:
        problems = list(missing_required)
        if not has_revenue_path:
            problems.append("revenue (and no unit_price+quantity pair to derive it from)")
        raise IngestionError(
            "Input is missing required field(s) that could not be matched or derived: "
            f"{', '.join(problems)}. Detected columns: {list(raw.columns)}. "
            f"Column mapping found so far: {report.column_mapping}."
        )

    df = _derive_missing_fields(df, report)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in ("unit_price", "quantity", "revenue", "list_price", "discount_pct",
                "unit_cost", "margin"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    before = len(df)
    bad_date = df["date"].isna()
    bad_revenue = df["revenue"].isna()
    bad_account = df["account_id"].isna() | (df["account_id"].astype(str).str.strip() == "")
    invalid = bad_date | bad_revenue | bad_account
    if invalid.any():
        report.row_drop_reasons = {
            "unparseable_date": int(bad_date.sum()),
            "missing_or_non_numeric_revenue": int(bad_revenue.sum()),
            "missing_account_id": int(bad_account.sum()),
        }
    df = df.loc[~invalid].reset_index(drop=True)
    report.rows_dropped_invalid = before - len(df)

    if df.empty:
        raise IngestionError(
            "All rows were dropped during validation (unparseable dates, "
            f"non-numeric price/quantity/revenue, or missing account_id). "
            f"Drop reasons: {report.row_drop_reasons}."
        )

    report.date_range = (df["date"].min(), df["date"].max())
    report.n_accounts = int(df["account_id"].nunique())
    report.n_orders = int(df["order_id"].nunique())
    report.n_categories = int(df["category"].nunique())
    report.margin_present = bool(df["margin"].notna().any())
    report.unit_economics_present = bool(df["unit_price"].notna().any() and df["quantity"].notna().any())
    report.tier_present = bool(df["tier"].notna().any())
    report.tier_values = [t for t in TIER_ORDER if (df["tier"] == t).any()]
    report.discount_present = bool(df["discount_pct"].notna().any())
    report.return_lines = int((df["is_return"] == 1).sum())
    report.return_value = round(float(df.loc[df["is_return"] == 1, "revenue"].sum()), 2)
    report.analysis_dimensions = analysis_dimensions(df)

    df = df[[c for c in CANONICAL_COLUMNS if c in df.columns]]
    return df, report.to_dict()


def analysis_dimensions(df: pd.DataFrame) -> dict:
    """Which detection dimensions the resolved columns can support.

    Stage 3 and the evidence pack consult this instead of testing for
    columns themselves, so that "we could not look" is always distinguishable
    downstream from "we looked and found nothing" — the difference between an
    honest partial answer and a false all-clear.
    """
    # bool(...) on every value: pandas `.any()` returns numpy.bool_, which
    # json.dumps refuses — and this dict is serialized straight into the
    # Stage 4 prompt.
    def has(col: str) -> bool:
        return bool(col in df.columns and df[col].notna().any())

    return {
        "revenue": has("revenue"),
        "margin": has("margin"),
        "discount": has("discount_pct"),
        "tier_mix": has("tier"),
        "category_mix": has("category"),
        "order_pattern": bool("order_id" in df.columns and df["order_id"].nunique() < len(df)),
        "returns": bool("is_return" in df.columns and (df["is_return"] == 1).any()),
    }


# A baseline needs enough months to be a baseline at all. Below this, the
# "recent window" and the "baseline window" are the same handful of orders
# and no trend statement is meaningful.
MIN_MONTHS_FOR_BASELINE = 6

# Seasonality can only be ruled out by looking at the same calendar window a
# year earlier. Under 12 months that comparison does not exist, so a decline
# is permanently ambiguous between "seasonal" and "structural" — the account
# can be described, but not classified with confidence.
MIN_MONTHS_FOR_SEASONAL_CHECK = 12


def account_sufficiency(df: pd.DataFrame) -> dict:
    """Per-account data-sufficiency score, used downstream to drive the
    defer path when history is too thin to support a confident verdict.

    Score is a weighted average of three normalized subscores (history
    length, order count, category coverage), then capped by two hard
    history floors — deterministic, no LLM involved.

    The caps exist because the weighted score alone can be talked up by
    breadth: an account onboarded five months ago that buys across every
    category scores ~0.4 ("marginal") purely on category coverage, when the
    honest answer is that five months cannot distinguish a temporary dip
    from a structural one at all. The floors make short history dominate,
    because for this question it does.
    """
    results = {}
    for account_id, g in df.groupby("account_id"):
        history_days = (g["date"].max() - g["date"].min()).days
        history_months = history_days / 30.44
        order_count = g["order_id"].nunique()
        category_count = g["category"].nunique()

        history_score = min(1.0, history_months / 12)
        order_score = min(1.0, order_count / 20)
        category_score = min(1.0, category_count / 3)
        # History and order count are the primary drivers of whether a trend
        # can be trusted; category breadth alone must not compensate for a
        # short or sparse history (e.g. many categories, four total orders).
        overall = round(0.45 * history_score + 0.45 * order_score + 0.10 * category_score, 2)

        flags = []
        if history_score < 0.5:
            flags.append("short_history")
        if order_score < 0.5:
            flags.append("few_orders")
        if category_score < 0.5:
            flags.append("narrow_category_coverage")

        if overall >= 0.7:
            label = "sufficient"
        elif overall >= 0.4:
            label = "marginal"
        else:
            label = "insufficient"

        if history_months < MIN_MONTHS_FOR_SEASONAL_CHECK:
            flags.append("cannot_check_prior_year_seasonality")
            if label == "sufficient":
                label = "marginal"
        if history_months < MIN_MONTHS_FOR_BASELINE:
            flags.append("history_too_short_for_baseline")
            label = "insufficient"

        results[account_id] = {
            "score": overall,
            "label": label,
            "flags": flags,
            "history_months": round(history_months, 1),
            "order_count": int(order_count),
            "category_count": int(category_count),
        }
    return results
