"""
The account book is a view of the evidence packs, not a second analysis.

Search and filters only slice an already-built table. A filter that
re-ran the pipeline would make Detect expensive again, which is the
thing this screen exists to avoid.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pipeline.evidence import build_evidence_pack
from pipeline.ingest import ingest
from ui.accounts import (
    DISPLAY_COLUMNS,
    account_row,
    build_account_catalogue,
    filter_catalogue,
    page_window,
    paginate_catalogue,
    _book_table_html,
    _cell_text,
)
from ui.palette import COLORS, status_style

from datasets import FLAGSHIP_LEAK, MERIDIAN_CSV, THIN_HISTORY, thin_transactions

FLAG_ACCOUNTS = ["ACC-101", "ACC-104", "ACC-106", "ACC-107", "ACC-108", "ACC-112"]


@pytest.fixture(scope="module")
def df():
    clean, _ = ingest(str(MERIDIAN_CSV))
    return clean


@pytest.fixture(scope="module")
def catalogue(df):
    return build_account_catalogue(df)


def test_catalogue_has_one_row_per_account(df, catalogue):
    assert set(catalogue["account_id"]) == set(df["account_id"].unique())
    assert len(catalogue) == df["account_id"].nunique()


def test_catalogue_carries_identity_and_every_signal(catalogue):
    for source, _ in DISPLAY_COLUMNS:
        assert source in catalogue.columns, source
    row = catalogue.set_index("account_id").loc[FLAGSHIP_LEAK]
    assert row["account_name"] == "Northgate Traders"
    assert row["region"]
    assert row["account_manager"]
    assert row["needs_attention"]


def test_catalogue_row_matches_the_pack_the_dashboard_reads(df, catalogue):
    pack = build_evidence_pack(df, FLAGSHIP_LEAK)
    expected = account_row(pack)
    got = catalogue.set_index("account_id").loc[FLAGSHIP_LEAK]
    assert got["revenue_recent"] == expected["revenue_recent"]
    assert got["margin_label"] == expected["margin_label"]
    assert got["tier_label"] == expected["tier_label"]


def test_search_is_client_side_and_does_not_drop_columns(catalogue):
    filtered = filter_catalogue(catalogue, search="northgate")
    assert list(filtered["account_id"]) == [FLAGSHIP_LEAK]
    assert set(filtered.columns) == set(catalogue.columns)


def test_region_and_manager_filters_slice_the_same_frame(catalogue):
    region = catalogue.set_index("account_id").loc[FLAGSHIP_LEAK]["region"]
    by_region = filter_catalogue(catalogue, regions=[region])
    assert FLAGSHIP_LEAK in set(by_region["account_id"])
    assert by_region["region"].eq(region).all()

    manager = catalogue.set_index("account_id").loc[FLAGSHIP_LEAK]["account_manager"]
    by_manager = filter_catalogue(catalogue, managers=[manager])
    assert FLAGSHIP_LEAK in set(by_manager["account_id"])
    assert by_manager["account_manager"].eq(manager).all()


def test_attention_filter_keeps_the_flag_accounts(catalogue):
    flagged = filter_catalogue(catalogue, attention="attention")
    assert set(FLAG_ACCOUNTS) <= set(flagged["account_id"])
    assert flagged["needs_attention"].all()

    clear = filter_catalogue(catalogue, attention="clear")
    assert FLAGSHIP_LEAK not in set(clear["account_id"])
    assert not clear["needs_attention"].any()


def test_thin_history_stays_on_the_book(catalogue):
    row = catalogue.set_index("account_id").loc[THIN_HISTORY]
    assert row["history_label"] == "insufficient"


def test_signal_filter_matches_any_status_column(catalogue):
    label = catalogue.set_index("account_id").loc[FLAGSHIP_LEAK]["margin_label"]
    filtered = filter_catalogue(catalogue, signals=[label])
    assert FLAGSHIP_LEAK in set(filtered["account_id"])


def test_thin_file_still_builds_a_book():
    clean, _ = ingest(thin_transactions())
    catalogue = build_account_catalogue(clean)
    assert len(catalogue) == clean["account_id"].nunique()
    assert catalogue["region"].eq("").all()
    assert catalogue["account_manager"].eq("").all()
    assert catalogue["revenue_recent"].notna().any()


def test_cell_text_formats_money_and_rates():
    assert _cell_text("revenue_recent", 283313.4) == "₹283,313"
    assert _cell_text("revenue_change_pct", -0.126) == "-12.6%"
    assert _cell_text("orders", 30) == "30"
    assert _cell_text("account_name", None) == "—"


def test_book_table_is_rows_not_checkboxes(catalogue):
    html = _book_table_html(catalogue.head(2))
    assert 'href="?account=ACC-101"' in html
    assert "checkbox" not in html.lower()
    assert html.count("<tr>") == 3  # header + two accounts


def test_book_table_colors_revenue_status(catalogue):
    html = _book_table_html(catalogue)
    assert "rl-book-status" in html

    color, icon, text = status_style("stable", COLORS)
    assert color == COLORS["good"]
    assert catalogue["revenue_status"].eq("stable").any()
    assert f"--sig:{color}" in html
    assert text in html
    assert icon in html

    leak = catalogue.set_index("account_id").loc["ACC-104"]
    color, icon, text = status_style(leak["revenue_status"], COLORS)
    assert leak["revenue_status"] == "material_decline"
    assert color == COLORS["critical"]
    assert f"--sig:{color}" in html
    assert text in html
    assert icon in html


def test_paginate_clamps_page_and_slices_client_side(catalogue):
    page_rows, page, total_pages, start, end = paginate_catalogue(catalogue, 1, 10)
    assert list(page_rows["account_id"]) == list(catalogue["account_id"].iloc[:10])
    assert (page, total_pages, start, end) == (1, 2, 0, 10)

    page_rows, page, total_pages, start, end = paginate_catalogue(catalogue, 99, 10)
    assert page == 2
    assert list(page_rows["account_id"]) == list(catalogue["account_id"].iloc[10:])
    assert end == len(catalogue)


def test_page_window_stays_short():
    assert page_window(1, 2) == [1, 2]
    assert page_window(1, 12) == [1, 2, 3, 4, 5]
    assert page_window(12, 12) == [8, 9, 10, 11, 12]
    assert page_window(6, 12) == [4, 5, 6, 7, 8]
