import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pipeline.ingest import IngestionError, account_sufficiency, ingest

REPO_ROOT = Path(__file__).resolve().parents[1]
GENERATED_CSV = REPO_ROOT / "synthetic_data" / "transactions.csv"


def test_ingest_generated_data_clean():
    df, report = ingest(str(GENERATED_CSV))
    assert list(df.columns) == [
        "order_id", "account_id", "date", "product_id",
        "category", "unit_price", "quantity", "revenue", "margin",
    ]
    assert report["rows_dropped_invalid"] == 0
    assert report["order_id_inferred"] is False
    assert report["margin_present"] is True
    assert report["n_accounts"] == 35


def test_ingest_no_margin_variant():
    df = pd.read_csv(GENERATED_CSV).drop(columns=["margin"])
    clean, report = ingest(df)
    assert "margin" in clean.columns  # present but all-NaN
    assert clean["margin"].isna().all()
    assert report["margin_present"] is False
    assert "margin" in report["fields_missing_optional"]


def test_ingest_hostile_schema_with_synonyms_and_derivation():
    """Simulate an unseen Reality Test file: different column names,
    no order_id, no revenue column (must derive from unit_price*qty)."""
    hostile = pd.DataFrame({
        "Customer ID": ["A1", "A1", "A2"],
        "Txn Date": ["2025-01-05", "2025-02-10", "2025-01-20"],
        "SKU": ["P1", "P2", "P3"],
        "Product Category": ["Widgets", "Widgets", "Gadgets"],
        "Rate": [100.0, 50.0, 200.0],
        "Qty": [2, 4, 1],
    })
    clean, report = ingest(hostile)
    assert len(clean) == 3
    assert clean["revenue"].tolist() == [200.0, 200.0, 200.0]
    assert report["order_id_inferred"] is True
    assert report["n_accounts"] == 2
    assert report["column_mapping"]["account_id"] == "Customer ID"
    assert report["column_mapping"]["date"] == "Txn Date"


def test_ingest_missing_required_field_raises_clear_error():
    no_date = pd.DataFrame({
        "account_id": ["A1"], "product_id": ["P1"], "category": ["Widgets"],
        "revenue": [100.0],
    })
    with pytest.raises(IngestionError, match="date"):
        ingest(no_date)


def test_ingest_no_revenue_path_raises_clear_error():
    no_revenue_path = pd.DataFrame({
        "account_id": ["A1"], "date": ["2025-01-01"],
        "product_id": ["P1"], "category": ["Widgets"],
    })
    with pytest.raises(IngestionError, match="revenue"):
        ingest(no_revenue_path)


def test_ingest_empty_file_raises():
    with pytest.raises(IngestionError):
        ingest(pd.DataFrame())


def test_ingest_drops_invalid_rows_not_whole_file():
    df = pd.DataFrame({
        "account_id": ["A1", "A1", None],
        "date": ["2025-01-01", "not-a-date", "2025-01-03"],
        "product_id": ["P1", "P2", "P3"],
        "category": ["Widgets", "Widgets", "Widgets"],
        "revenue": [100.0, 100.0, 100.0],
    })
    clean, report = ingest(df)
    assert len(clean) == 1
    assert report["rows_dropped_invalid"] == 2
    assert report["row_drop_reasons"]["unparseable_date"] == 1
    assert report["row_drop_reasons"]["missing_account_id"] == 1


def test_account_sufficiency_flags_sparse_history():
    df, _ = ingest(str(GENERATED_CSV))
    import json
    gt = json.load(open(REPO_ROOT / "synthetic_data" / "ground_truth.json"))
    sparse_accounts = [g["account_id"] for g in gt if g["archetype"] == "sparse_short_history"]
    normal_accounts = [g["account_id"] for g in gt if g["archetype"] == "healthy"]

    scores = account_sufficiency(df)

    for acc in sparse_accounts:
        assert scores[acc]["label"] == "insufficient", f"{acc}: {scores[acc]}"
        assert "short_history" in scores[acc]["flags"]

    for acc in normal_accounts:
        assert scores[acc]["label"] in ("sufficient", "marginal"), f"{acc}: {scores[acc]}"
