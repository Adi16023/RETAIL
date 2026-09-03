import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pipeline.ingest import CANONICAL_COLUMNS, IngestionError, account_sufficiency, ingest

from datasets import MERIDIAN_CSV, THIN_HISTORY, thin_transactions

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKBOOK = REPO_ROOT / "Quessathon_Revenue_Leakage_Dataset.xlsx"


def test_ingest_the_reference_dataset_resolves_every_column():
    """All 17 workbook columns must map, with nothing derived and nothing
    dropped. If a single column stops resolving, a whole detection dimension
    silently goes dark — so this asserts the mapping itself, not just that
    ingestion succeeded."""
    df, report = ingest(str(MERIDIAN_CSV))

    assert list(df.columns) == CANONICAL_COLUMNS
    assert report["rows_total_raw"] == 3289
    assert report["rows_dropped_invalid"] == 0
    assert report["fields_inferred"] == [], "the workbook supplies everything directly"
    assert report["fields_missing_optional"] == []
    assert report["n_accounts"] == 18
    assert report["n_orders"] == 551

    assert report["column_mapping"]["date"] == "order_date"
    assert report["column_mapping"]["product_id"] == "product"
    assert report["column_mapping"]["revenue"] == "line_revenue"
    assert report["column_mapping"]["margin"] == "line_margin"
    assert report["column_mapping"]["unit_price"] == "unit_net_price"
    assert report["column_mapping"]["list_price"] == "list_price", "must not be taken by unit_price"

    assert report["tier_values"] == ["High", "Mid", "Low"]
    assert report["tier_values_unrecognized"] == []
    assert report["return_lines"] == 3
    assert report["return_value"] < 0
    assert all(report["analysis_dimensions"].values()), report["analysis_dimensions"]


def test_ingest_reads_the_workbook_directly():
    """The .xlsx is the artefact the organisers hand over. Reading it must
    not depend on having run the extraction script first, and the right tab
    has to be picked out of six without being told which."""
    df, report = ingest(str(WORKBOOK))
    assert report["n_accounts"] == 18
    assert report["rows_total_raw"] == 3289
    assert list(df.columns) == CANONICAL_COLUMNS


def test_ingest_thin_file_degrades_instead_of_breaking():
    """A file with no tier, list price, discount or unit cost is the shape
    the Reality Test may arrive in. Ingestion must still produce the full
    canonical frame — missing columns present-but-empty, so every downstream
    stage can index them unconditionally — and must report which dimensions
    it therefore could not analyse."""
    clean, report = ingest(thin_transactions())

    assert list(clean.columns) == CANONICAL_COLUMNS
    assert report["rows_dropped_invalid"] == 0
    assert report["n_accounts"] == 18

    for absent in ("tier", "list_price", "discount_pct", "unit_cost"):
        assert absent in report["fields_missing_optional"]
        assert clean[absent].isna().all()

    dimensions = report["analysis_dimensions"]
    assert dimensions["revenue"] is True and dimensions["margin"] is True
    # The distinction the whole design rests on: not analysed, not "fine".
    assert dimensions["tier_mix"] is False
    assert dimensions["discount"] is False

    # Returns survive the column loss: a credit note is identifiable by its
    # sign alone, so netting still happens on a file with no is_return flag.
    assert dimensions["returns"] is True
    assert report["return_lines"] == 3
    assert any("is_return" in f for f in report["fields_inferred"])


def test_ingest_no_margin_variant():
    """The narrowest realistic file: revenue only. Margin must come back
    present-but-empty and be reported as unavailable, never as zero."""
    clean, report = ingest(thin_transactions(with_margin=False))

    assert "margin" in clean.columns
    assert clean["margin"].isna().all()
    assert report["margin_present"] is False
    assert "margin" in report["fields_missing_optional"]
    assert report["analysis_dimensions"]["margin"] is False


def test_margin_is_reconstructed_from_unit_cost_when_absent():
    """Margin drives two of the six real leak detections, so it is worth
    rebuilding wherever the file gives us the pieces rather than declaring
    the dimension unavailable."""
    raw = pd.read_csv(MERIDIAN_CSV).drop(columns=["line_margin"])
    clean, report = ingest(raw)

    assert report["margin_present"] is True
    assert any("margin" in f for f in report["fields_inferred"])
    assert report["analysis_dimensions"]["margin"] is True

    expected = pd.read_csv(MERIDIAN_CSV)["line_margin"]
    assert (clean["margin"] - expected).abs().max() < 0.01


def test_unrecognized_tier_labels_are_reported_not_guessed():
    """A mis-mapped tier would silently corrupt the high-tier-share signal
    that the flagship leak turns on, so an unknown label must be surfaced
    and left unmapped rather than bucketed on a guess."""
    raw = pd.read_csv(MERIDIAN_CSV)
    raw.loc[raw["tier"] == "Mid", "tier"] = "Tier Q"
    clean, report = ingest(raw)

    assert report["tier_values_unrecognized"] == ["Tier Q"]
    assert set(clean["tier"].dropna().unique()) == {"High", "Low"}
    assert any("unrecognized" in f for f in report["fields_inferred"])


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


def test_account_sufficiency_isolates_the_one_account_that_must_defer():
    """Exactly one account in the reference book has too little history to
    call. Both halves matter: it must be flagged insufficient, and the other
    seventeen must not be — an over-eager sufficiency check would push the
    agent to defer on accounts it can and should judge."""
    df, _ = ingest(str(MERIDIAN_CSV))
    scores = account_sufficiency(df)

    insufficient = {a for a, s in scores.items() if s["label"] == "insufficient"}
    assert insufficient == {THIN_HISTORY}

    thin = scores[THIN_HISTORY]
    assert thin["history_months"] < 6
    assert "history_too_short_for_baseline" in thin["flags"]
    # Under a year there is no prior year to compare against, so seasonality
    # can never be ruled out — that is why this account is uncallable.
    assert "cannot_check_prior_year_seasonality" in thin["flags"]

    for account_id, score in scores.items():
        if account_id != THIN_HISTORY:
            assert score["label"] == "sufficient", f"{account_id}: {score}"


def test_category_breadth_cannot_mask_a_short_history():
    """The account that must defer buys across all eight categories, which
    is enough to carry the weighted score into 'marginal' on breadth alone.
    The history floor exists so that cannot happen."""
    df, _ = ingest(str(MERIDIAN_CSV))
    thin = account_sufficiency(df)[THIN_HISTORY]

    assert thin["category_count"] == 8, "breadth is genuinely high here"
    assert thin["score"] >= 0.4, "the weighted score alone would say 'marginal'"
    assert thin["label"] == "insufficient", "the history floor must override it"
