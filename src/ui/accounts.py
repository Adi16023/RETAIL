"""
The account book — every account as a row, the dashboard as a click-through.

Detect opens on this table so a manager can scan the book before opening
anyone. Every cell is a figure the evidence pack already computed; search
and filters only slice this frame, they never re-run the pipeline.
"""

from __future__ import annotations

import math
from html import escape
from urllib.parse import quote

import pandas as pd
import streamlit as st

from pipeline.evidence import build_evidence_pack

from .palette import STATUS_MEANING, active, status_style

# Detector statuses whose role is a real problem, not a warning or an all-clear.
_CONCERN_ROLES = frozenset({"critical", "serious"})

# Catalogue field -> the column a manager reads. Order is the table order.
DISPLAY_COLUMNS = [
    ("account_id", "Account"),
    ("account_name", "Name"),
    ("region", "Region"),
    ("revenue_label", "Revenue"),
    ("account_manager", "Manager"),
    ("months", "Months"),
    ("orders", "Orders"),
    ("revenue_recent", "Revenue / mo"),
    ("revenue_change_pct", "Change"),
    ("margin_recent_pct", "Margin %"),
    ("discount_recent_pct", "Discount %"),
    ("high_tier_recent_pct", "High-tier %"),
    ("margin_label", "Margin"),
    ("discount_label", "Discount"),
    ("tier_label", "Value mix"),
    ("order_label", "Order shape"),
    ("category_label", "Categories"),
    ("history_label", "History"),
]

SEARCH_COLUMNS = ("account_id", "account_name", "region", "account_manager")

ROWS_PER_PAGE_OPTIONS = (10, 25, 50, 100)
DEFAULT_ROWS_PER_PAGE = 25


def _as_text(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    return str(value)


def _status_label(status: str | None) -> str:
    return STATUS_MEANING.get(status or "unavailable", ("warning", "—", str(status)))[2]


def is_concern(status: str | None) -> bool:
    role, _, _ = STATUS_MEANING.get(status or "", ("warning", "", ""))
    return role in _CONCERN_ROLES


def account_row(pack: dict) -> dict:
    """One catalogue row from an evidence pack. Magnitudes stay numeric."""
    revenue = pack.get("overall_revenue") or {}
    margin = pack.get("margin_profile") or {}
    discount = pack.get("discount") or {}
    tier = pack.get("tier_mix") or {}
    history = pack.get("history") or {}

    revenue_status = (pack.get("revenue_decline") or {}).get("status")
    margin_status = (pack.get("margin") or {}).get("status")
    discount_status = (pack.get("discount") or {}).get("status")
    tier_status = (pack.get("tier_mix") or {}).get("status")
    order_status = (pack.get("order_pattern") or {}).get("status")

    defected = [
        change["category"]
        for change in pack.get("category_changes") or []
        if change.get("defected")
    ]
    category_label = f"Lost: {', '.join(defected)}" if defected else _status_label("stable")
    history_label = (pack.get("data_sufficiency") or {}).get("label") or "—"

    needs_attention = bool(defected) or any(
        is_concern(status)
        for status in (revenue_status, margin_status, discount_status, tier_status, order_status)
    )

    return {
        "account_id": pack["account_id"],
        "account_name": _as_text(pack.get("account_name")),
        "region": _as_text(pack.get("region")),
        "account_manager": _as_text(pack.get("account_manager")),
        "months": history.get("months_of_history"),
        "orders": history.get("order_count"),
        "revenue_recent": revenue.get("recent_monthly_rate"),
        "revenue_change_pct": revenue.get("pct_change"),
        "margin_recent_pct": margin.get("recent_margin_pct"),
        "discount_recent_pct": discount.get("recent_avg_discount_pct"),
        "high_tier_recent_pct": (tier.get("recent_share_by_tier") or {}).get("High"),
        "revenue_status": revenue_status,
        "margin_status": margin_status,
        "discount_status": discount_status,
        "tier_status": tier_status,
        "order_status": order_status,
        "revenue_label": _status_label(revenue_status),
        "margin_label": _status_label(margin_status),
        "discount_label": _status_label(discount_status),
        "tier_label": _status_label(tier_status),
        "order_label": _status_label(order_status),
        "category_label": category_label,
        "history_label": history_label,
        "needs_attention": needs_attention,
    }


def build_account_catalogue(df: pd.DataFrame) -> pd.DataFrame:
    """One row per account in `df`, from the same packs the dashboard reads."""
    rows = [
        account_row(build_evidence_pack(df, account_id))
        for account_id in sorted(df["account_id"].unique())
    ]
    return pd.DataFrame(rows)


def filter_catalogue(
    catalogue: pd.DataFrame,
    *,
    search: str = "",
    regions: list[str] | tuple[str, ...] = (),
    managers: list[str] | tuple[str, ...] = (),
    history: list[str] | tuple[str, ...] = (),
    signals: list[str] | tuple[str, ...] = (),
    attention: str = "",
) -> pd.DataFrame:
    """Slice an already-built catalogue. No pipeline work."""
    frame = catalogue
    query = (search or "").strip().casefold()
    if query:
        haystack = frame.reindex(columns=list(SEARCH_COLUMNS)).fillna("").astype(str)
        matched = haystack.apply(
            lambda column: column.str.casefold().str.contains(query, regex=False),
            axis=0,
        ).any(axis=1)
        frame = frame[matched]

    if regions:
        frame = frame[frame["region"].isin(regions)]
    if managers:
        frame = frame[frame["account_manager"].isin(managers)]
    if history:
        frame = frame[frame["history_label"].isin(history)]
    if signals:
        signal_columns = [
            "revenue_label", "margin_label", "discount_label",
            "tier_label", "order_label", "category_label",
        ]
        present = [column for column in signal_columns if column in frame.columns]
        frame = frame[frame[present].isin(signals).any(axis=1)]
    if attention == "attention":
        frame = frame[frame["needs_attention"]]
    elif attention == "clear":
        frame = frame[~frame["needs_attention"]]
    return frame.reset_index(drop=True)


def paginate_catalogue(
    catalogue: pd.DataFrame,
    page: int,
    rows_per_page: int,
) -> tuple[pd.DataFrame, int, int, int, int]:
    """Return (page rows, clamped page, total pages, start index, end index)."""
    count = len(catalogue)
    size = max(1, int(rows_per_page))
    total_pages = max(1, math.ceil(count / size)) if count else 1
    page = min(max(1, int(page)), total_pages)
    start = 0 if count == 0 else (page - 1) * size
    end = min(start + size, count)
    return catalogue.iloc[start:end].reset_index(drop=True), page, total_pages, start, end


def page_window(current: int, total: int, span: int = 5) -> list[int]:
    """The page numbers youkti shows — a short window around the current page."""
    if total <= span:
        return list(range(1, total + 1))
    if current <= 3:
        return list(range(1, span + 1))
    if current >= total - 2:
        return list(range(total - span + 1, total + 1))
    return list(range(current - 2, current + 3))


def _unique_values(catalogue: pd.DataFrame, column: str) -> list[str]:
    values = catalogue[column].dropna().astype(str)
    return sorted(value for value in values.unique() if value and value != "—")


_MONEY_COLUMNS = frozenset({"revenue_recent"})
_RATIO_COLUMNS = frozenset({
    "revenue_change_pct", "margin_recent_pct",
    "discount_recent_pct", "high_tier_recent_pct",
})


def _cell_text(source: str, value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    if source in _MONEY_COLUMNS:
        return f"₹{value:,.0f}"
    if source in _RATIO_COLUMNS:
        return f"{value * 100:.1f}%"
    if source == "months":
        return f"{value:.1f}"
    if source == "orders":
        return f"{int(value)}"
    return str(value) or "—"


def _status_key(value) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    return str(value)


def _revenue_status_html(row) -> str:
    """Icon + word in the palette color. Hue is never the only signal."""
    color, icon, text = status_style(_status_key(row.get("revenue_status")), active())
    return (
        f'<span class="rl-book-status" style="--sig:{escape(color, quote=True)}">'
        f'<span class="rl-book-status-icon" aria-hidden="true">{escape(icon)}</span>'
        f"{escape(text)}</span>"
    )


def _opened_account(catalogue: pd.DataFrame) -> str | None:
    """A row click lands as `?account=ACC-101`. Consume it once."""
    requested = st.query_params.get("account")
    if not requested:
        return None
    known = set(catalogue["account_id"].astype(str))
    if "account" in st.query_params:
        del st.query_params["account"]
    return requested if requested in known else None


def _book_table_html(catalogue: pd.DataFrame) -> str:
    """A real table: the row is the hit target, no checkbox column."""
    headers = "".join(
        f"<th>{escape(label)}</th>"
        for source, label in DISPLAY_COLUMNS
        if source in catalogue.columns
    )
    rows = []
    for _, row in catalogue.iterrows():
        account_id = str(row["account_id"])
        href = f"?account={quote(account_id, safe='')}"
        cells = []
        for source, _label in DISPLAY_COLUMNS:
            if source not in catalogue.columns:
                continue
            inner = (
                _revenue_status_html(row)
                if source == "revenue_label"
                else escape(_cell_text(source, row[source]))
            )
            cells.append(
                f'<td><a class="rl-book-hit" href="{escape(href, quote=True)}">{inner}</a></td>'
            )
        rows.append(f"<tr>{''.join(cells)}</tr>")
    return (
        '<div class="rl-book-wrap">'
        f'<table class="rl-book"><thead><tr>{headers}</tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table></div>'
    )


def render_account_book(
    catalogue: pd.DataFrame,
    *,
    trailing=None,
) -> str | None:
    """Search, filter and table. Returns an account id when a row is opened.

    `trailing` is an optional callable rendered on the search row — used for
    the file-info control so it sits with the filters, not above the table.
    """
    options_region = _unique_values(catalogue, "region")
    options_manager = _unique_values(catalogue, "account_manager")
    options_history = _unique_values(catalogue, "history_label")
    options_signal = sorted({
        label
        for column in (
            "revenue_label", "margin_label", "discount_label",
            "tier_label", "order_label", "category_label",
        )
        for label in _unique_values(catalogue, column)
    })

    # Search and Filters share the left side. A spacer column keeps the
    # file-info control on the right without stretching the two apart.
    if trailing is not None:
        search_col, filter_col, _, info_col = st.columns(
            [2.4, 0.9, 3.2, 1.5], vertical_alignment="bottom", gap="small",
        )
    else:
        search_col, filter_col, _ = st.columns(
            [2.4, 0.9, 4.7], vertical_alignment="bottom", gap="small",
        )
        info_col = None
    with search_col:
        search = st.text_input(
            "Search accounts",
            placeholder="Search by id, name, region or manager",
            label_visibility="collapsed",
            key="account_search",
        )
    with filter_col:
        with st.popover(
            "Filters",
            icon=":material/filter_alt:",
            use_container_width=True,
            key="account_filters",
        ):
            st.caption("These only hide rows. The figures are already computed.")
            attention = st.selectbox(
                "Attention",
                options=["", "attention", "clear"],
                format_func=lambda value: {
                    "": "All accounts",
                    "attention": "Needs attention",
                    "clear": "No material signal",
                }[value],
                key="account_filter_attention",
            )
            regions = st.multiselect(
                "Region", options_region,
                key="account_filter_region",
            ) if options_region else []
            managers = st.multiselect(
                "Manager", options_manager,
                key="account_filter_manager",
            ) if options_manager else []
            history = st.multiselect(
                "History", options_history,
                key="account_filter_history",
            ) if options_history else []
            signals = st.multiselect(
                "Signal", options_signal,
                key="account_filter_signal",
            )
    if trailing is not None and info_col is not None:
        with info_col:
            trailing()

    filtered = filter_catalogue(
        catalogue,
        search=search or "",
        regions=regions,
        managers=managers,
        history=history,
        signals=signals,
        attention=attention or "",
    )

    chips = []
    if attention == "attention":
        chips.append("Needs attention")
    elif attention == "clear":
        chips.append("No material signal")
    chips.extend(f"Region: {value}" for value in regions)
    chips.extend(f"Manager: {value}" for value in managers)
    chips.extend(f"History: {value}" for value in history)
    chips.extend(f"Signal: {value}" for value in signals)
    if (search or "").strip():
        chips.append(f"Search: {(search or '').strip()}")

    if chips:
        chip_row, clear_row = st.columns([4, 1], vertical_alignment="center")
        with chip_row:
            st.caption(" · ".join(chips))
        with clear_row:
            if st.button("Clear filters", type="tertiary", use_container_width=True):
                for key in (
                    "account_search", "account_filter_attention", "account_filter_region",
                    "account_filter_manager", "account_filter_history", "account_filter_signal",
                    "account_page",
                ):
                    st.session_state.pop(key, None)
                st.rerun()

    filter_fp = (
        search or "", attention or "",
        tuple(regions), tuple(managers), tuple(history), tuple(signals),
    )
    if st.session_state.get("_account_filter_fp") != filter_fp:
        st.session_state["_account_filter_fp"] = filter_fp
        st.session_state["account_page"] = 1

    if "account_rows_per_page" not in st.session_state:
        st.session_state["account_rows_per_page"] = DEFAULT_ROWS_PER_PAGE
    if "account_page" not in st.session_state:
        st.session_state["account_page"] = 1

    page_rows, page, total_pages, start, end = paginate_catalogue(
        filtered,
        st.session_state["account_page"],
        st.session_state["account_rows_per_page"],
    )
    st.session_state["account_page"] = page

    if filtered.empty:
        st.info("No accounts match these filters.")
        opened = None
    else:
        opened = _opened_account(catalogue)
        st.markdown(_book_table_html(page_rows), unsafe_allow_html=True)

    _render_pagination(len(filtered), page, total_pages, start, end)
    return opened


def _set_account_page(page: int) -> None:
    st.session_state["account_page"] = page


def _reset_account_page() -> None:
    st.session_state["account_page"] = 1


def _render_pagination(total: int, page: int, total_pages: int, start: int, end: int) -> None:
    """Left: rows-per-page + range. Right: compact page buttons."""
    shown_from, shown_to = (start + 1, end) if total else (0, 0)
    window = page_window(page, total_pages)
    with st.container(key="account_pagination"):
        label, picker, shown, _, nav = st.columns(
            [0.85, 0.72, 1.7, 4.1, 1.4], vertical_alignment="center", gap="small",
        )
        label.caption("Rows per page")
        picker.selectbox(
            "Rows per page",
            list(ROWS_PER_PAGE_OPTIONS),
            key="account_rows_per_page",
            on_change=_reset_account_page,
            label_visibility="collapsed",
        )
        shown.caption(f"Showing **{shown_from}** to **{shown_to}** of **{total}** accounts")
        with nav:
            disabled = total == 0 or total_pages <= 1
            slots = [
                ("first", ":material/first_page:", 1, disabled or page <= 1, "First page"),
                ("prev", ":material/chevron_left:", page - 1, disabled or page <= 1, "Previous page"),
                *((f"{number}", None, number, disabled, f"Page {number}") for number in window),
                ("next", ":material/chevron_right:", page + 1, disabled or page >= total_pages, "Next page"),
                ("last", ":material/last_page:", total_pages, disabled or page >= total_pages, "Last page"),
            ]
            buttons = st.columns(len(slots), gap="small")
            for column, (key, icon, target, is_disabled, help_text) in zip(buttons, slots):
                current = key.isdigit() and int(key) == page
                column.button(
                    key if key.isdigit() else "",
                    icon=icon,
                    type="primary" if current else "tertiary",
                    disabled=is_disabled,
                    on_click=_set_account_page, args=(target,),
                    key=f"account-page-{key}",
                    help=help_text,
                )
