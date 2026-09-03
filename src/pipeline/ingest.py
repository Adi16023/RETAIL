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
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

CANONICAL_COLUMNS = [
    "order_id", "account_id", "date", "product_id",
    "category", "unit_price", "quantity", "revenue", "margin",
]

# Fields the pipeline cannot function without, once revenue is resolvable
# (revenue itself is required but may be *derived* from unit_price*quantity).
REQUIRED_FIELDS = ["account_id", "date", "product_id", "category"]

# order_id and margin are never required — see derive_missing_fields.
OPTIONAL_FIELDS = ["order_id", "margin"]

SYNONYMS = {
    "account_id": ["account_id", "customer_id", "client_id", "customer", "client", "cust_id"],
    "date": ["date", "order_date", "txn_date", "transaction_date", "purchase_date", "invoice_date"],
    "product_id": ["product_id", "sku", "item_id", "item_code", "product_code"],
    "category": ["category", "product_category", "category_name", "segment", "product_group"],
    "unit_price": ["unit_price", "price", "unitprice", "price_per_unit", "rate"],
    "quantity": ["quantity", "qty", "units", "unit_count", "no_of_units"],
    "revenue": ["revenue", "amount", "net_sales", "total", "line_total", "sales_amount",
                "total_amount", "net_amount"],
    "margin": ["margin", "profit", "gross_margin", "margin_amount"],
    "order_id": ["order_id", "order_number", "order_no", "invoice_id", "invoice_number"],
}

FUZZY_CUTOFF = 0.75


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


def _derive_missing_fields(df: pd.DataFrame, report: ValidationReport) -> pd.DataFrame:
    if "revenue" not in df.columns and {"unit_price", "quantity"}.issubset(df.columns):
        df["revenue"] = df["unit_price"] * df["quantity"]
        report.fields_inferred.append("revenue (unit_price * quantity)")
    elif "unit_price" not in df.columns and {"revenue", "quantity"}.issubset(df.columns):
        with np.errstate(divide="ignore", invalid="ignore"):
            df["unit_price"] = np.where(df["quantity"] != 0, df["revenue"] / df["quantity"], np.nan)
        report.fields_inferred.append("unit_price (revenue / quantity)")

    if "order_id" not in df.columns:
        df["order_id"] = [f"ROW-{i}" for i in range(len(df))]
        report.order_id_inferred = True
        report.fields_inferred.append("order_id (synthetic, one order per row — order-size analysis degraded)")

    for optional_col in ("unit_price", "quantity", "margin"):
        if optional_col not in df.columns:
            df[optional_col] = np.nan
            report.fields_missing_optional.append(optional_col)

    return df


def ingest(source: str | pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Ingest a raw transaction file (or DataFrame) and resolve it to the
    canonical schema.

    Returns (clean_df, validation_report_dict). Raises IngestionError with
    a specific message if the input cannot be resolved.
    """
    if isinstance(source, pd.DataFrame):
        raw = source.copy()
    else:
        try:
            raw = pd.read_csv(source)
        except Exception as e:
            raise IngestionError(f"Could not read '{source}' as CSV: {e}") from e

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
    for col in ("unit_price", "quantity", "revenue"):
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

    df = df[[c for c in CANONICAL_COLUMNS if c in df.columns]]
    return df, report.to_dict()


def account_sufficiency(df: pd.DataFrame) -> dict:
    """Per-account data-sufficiency score, used downstream to drive the
    defer path when history is too thin to support a confident verdict.

    Score is a simple average of three normalized subscores (history length,
    order count, category coverage) — deterministic, no LLM involved.
    """
    results = {}
    now = df["date"].max()
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

        results[account_id] = {
            "score": overall,
            "label": label,
            "flags": flags,
            "history_months": round(history_months, 1),
            "order_count": int(order_count),
            "category_count": int(category_count),
        }
    return results
