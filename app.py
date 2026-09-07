"""
Revenue Leakage Investigator — the demo surface.

A home sidebar with four pages. Detect opens on the account book; click a
row to open that account. The other three pages then work on the one you
opened:

  Detect       the book, then the deterministic evidence
  Investigate  the single LLM call, on demand
  Attribute    scored against the Answer Key
  Prioritise   several finished accounts, side by side

Headings and captions are written for an account manager, not an engineer:
they say what the section answers, not how it is computed. Where an
implementation detail earns its place ("every number it quotes comes from the
analysis above"), it is there because it tells the reader how much to trust
what they are looking at.

The pages still build on each other: the evidence is what the agent reasons
over, and the verdict is what the Answer Key scores.

The evidence costs nothing — clicking through eighteen accounts spends no money and
waits on no model. The verdict is a paid API call, so its result is cached to disk
by account + model + data fingerprint and reused automatically; only a change
to one of those three re-runs it.

No hard-coded absolute paths — everything is relative to this file or comes
from the uploaded file object.
"""

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent
load_dotenv(REPO_ROOT / ".env")
sys.path.insert(0, str(REPO_ROOT / "src"))

from pipeline.agent import AgentError
from ml.predict import attach_model_opinion
from pipeline.compare import (
    MAX_ACCOUNTS,
    MIN_ACCOUNTS,
    ComparisonError,
    build_comparison_digest,
    compare_accounts,
)
from pipeline.decide import DecisionError, build_decision_input, price_options, recommend
from pipeline.evidence import build_evidence_pack
from pipeline.impact import compute_impact
from pipeline.ingest import IngestionError, ingest
from pipeline.prioritize import prioritize
from pipeline.report import assemble_report
from pipeline.timeline import PRESENCE_ONLY_DIMENSIONS, build_timeline
from ui import cache
from ui.accounts import apply_investigation_cache, build_account_catalogue, render_account_book
from ui.aryachat import render_aryachat
from ui.compare import render_comparison
from ui.decide import render_decision, render_early_warning, render_no_lever, render_options
from ui.palette import active
from ui.dashboard import render_account_dashboard
from ui import theme
from ui.verdict import render_verdict, render_verdict_header
from validation.answer_key import AnswerKeyUnavailable, load_answer_key, score_report

MERIDIAN_TRANSACTIONS = REPO_ROOT / "data" / "meridian" / "transactions.csv"

# The model choice is presented by what it MEANS to the business — an open
# model you can self-host versus a proprietary API — not by model id. A
# manager choosing between "gpt-oss-120b" and "claude-sonnet-5" is being asked
# a question they have no basis to answer; choosing between open source and
# proprietary is a real decision they own.
MODEL_CHOICES = {
    "Open source": {
        "provider": "Groq",
        "model": "openai/gpt-oss-120b",
        "note": "An open-weights model. No vendor lock-in, self-hostable.",
    },
    "Proprietary": {
        "provider": "Anthropic",
        "model": "claude-sonnet-5",
        "note": "A frontier commercial model, called over its vendor API.",
    },
}

# Every model call in the app — the investigation, the recommendation, the
# comparison and the chat — runs on this one, and no page shows a picker
# (removed Sept 7 at the user's request: the open-source model's free tier
# cannot carry a chat turn, and a manager should not be asked to choose).
# MODEL_CHOICES stays as the mapping; switching the whole app is this line.
DEFAULT_MODEL_CHOICE = "Proprietary"

st.set_page_config(
    page_title="Revenue Leakage Investigator",
    layout="wide",
    initial_sidebar_state="expanded",
)
theme.inject()

# Same four pages as the old numbered steps. Sidebar labels are the pipeline
# words; full titles and captions stay on the page itself.
PAGES = [
    {
        "key": "ask",
        "icon": ":material/forum:",
        "number": 0,
        "label": "AryaChat",
        "title": "Ask about your accounts",
        "caption": "Ask about any account, or the whole book, in plain words.",
    },
    {
        "key": "data",
        "icon": ":material/analytics:",
        "number": 1,
        "label": "Detect",
        "title": "What this account has been doing",
        "caption": (
            "Every order, month by month, across the six things that can quietly go wrong: "
            "spend, margin, discounting, what they buy, and how they order."
        ),
        "list_title": "Which accounts need a look?",
        "list_caption": (
            "Every account in this file, with the figures the analysis uses. Click a row to open it."
        ),
    },
    {
        "key": "verdict",
        "icon": ":material/psychology:",
        "number": 2,
        "label": "Investigate",
        "title": "Is anything actually going wrong?",
        "caption": (
            "A straight call from the AI — leakage, a temporary dip, healthy, or too thin to "
            "decide — with the figures behind it and what to do."
        ),
    },
    # Attribute — scores the verdict against the workbook's Answer Key. It is
    # a validation surface, not something a manager uses, so it is hidden for
    # now. `render_score_page` stays; restore this entry (and the "score"
    # branch at the bottom of the file) to bring it back.
    # {
    #     "key": "score",
    #     "icon": ":material/fact_check:",
    #     "number": 3,
    #     "label": "Attribute",
    #     "title": "How much should you trust that verdict?",
    #     "caption": (
    #         "The right answer for each demo account was written down in advance and kept away "
    #         "from the AI. This compares the two."
    #     ),
    # },
    # Prioritise — several accounts side by side. Hidden from the nav since
    # Sept 7 at the user's request: AryaChat answers "who do I call first" and
    # "compare A and B" from the same digest, so the page said the same thing
    # twice. `render_compare_page` stays; restore this entry (and the routing
    # branch at the bottom of the file) to bring it back.
    # {
    #     "key": "compare",
    #     "icon": ":material/compare_arrows:",
    #     "number": 3,
    #     "label": "Prioritise",
    #     "title": "How do these accounts compare?",
    #     "caption": (
    #         "Several accounts side by side: who to call first, and which are living the same story."
    #     ),
    # },
]


def _set_home_page(page_key: str) -> None:
    st.session_state["home_page"] = page_key


def _open_account(account_id: str) -> None:
    st.session_state["selected_account"] = account_id


def _back_to_account_book() -> None:
    st.session_state["selected_account"] = None
    st.session_state["home_page"] = "data"
    st.session_state.pop("account_book", None)
    if "account" in st.query_params:
        del st.query_params["account"]


# --- Live model access -----------------------------------------------------

def get_live_client(provider: str, groq_model: str | None = None):
    """Construct the real LLM client. Imported locally so the rest of the app
    still runs when a package or credential is missing.

    Nothing else in the pipeline is provider-aware — swapping providers is
    this function and nothing more.
    """
    if provider == "Groq":
        from pipeline.groq_client import GroqShimClient
        return GroqShimClient(model_override=groq_model)
    import anthropic
    return anthropic.Anthropic()


def run_agent_with_error_handling(client, df, account_id, evidence_pack, model, provider):
    from pipeline.agent import investigate

    error_module = __import__("groq") if provider == "Groq" else __import__("anthropic")
    key_hint = "GROQ_API_KEY" if provider == "Groq" else "ANTHROPIC_API_KEY"

    try:
        return investigate(client, df, account_id, evidence_pack, model=model), None
    except error_module.AuthenticationError:
        return None, f"No valid API credentials for this model — set {key_hint} in .env."
    except error_module.RateLimitError:
        return None, "Rate limited by the model provider. Wait a moment and retry."
    except error_module.APIConnectionError:
        return None, "Could not reach the model provider (network issue)."
    except error_module.APIStatusError as e:
        return None, f"Model provider error ({e.status_code}): {e.message}"
    except AgentError as e:
        return None, f"Agent did not reach a verdict: {e}"
    except Exception as e:
        return None, f"Unexpected error during investigation: {e}"


def investigate_account(df, account_id, pack, choice_label):
    """Run Stages 4-7 for one account. Returns (report, error)."""
    spec = MODEL_CHOICES[choice_label]
    provider, model = spec["provider"], spec["model"]

    try:
        client = get_live_client(provider, groq_model=model if provider == "Groq" else None)
    except Exception as e:
        key_hint = "GROQ_API_KEY" if provider == "Groq" else "ANTHROPIC_API_KEY"
        return None, f"Could not initialize the model client: {e}. Set {key_hint} in .env."

    verdict, error = run_agent_with_error_handling(client, df, account_id, pack, model, provider)
    if error:
        return None, error

    impact = compute_impact(pack, verdict)
    priority = prioritize(impact, verdict)
    report = assemble_report(account_id, pack, verdict, impact, priority)
    report["_run"] = {"choice": choice_label, "model": model, "provider": provider}
    return report, None


# --- PDF export ------------------------------------------------------------

@st.cache_data(show_spinner=False)
def build_pdf_cached(report_json: str, depth: str) -> bytes:
    """Render a report to PDF, memoised on (report, depth).

    Keyed by the serialized report rather than the dict so Streamlit can hash
    it, and so that re-rendering the page — which happens on every widget
    interaction — does not rebuild the document each time.
    """
    from reporting.pdf import build_pdf

    return build_pdf(json.loads(report_json), depth)


# Only the brief is offered here. `reporting.pdf` still renders a full
# dossier — the per-dimension evidence, the ruled-out explanations and the
# provenance appendix are all intact behind BRIEF's sibling depth — but this
# screen deliberately exposes one document, so there is no depth to choose
# and no wrong choice to make. Re-enabling it is a widget, not a rewrite.
PDF_DEPTH = "brief"


def render_pdf_download(report: dict) -> None:
    """A download button for the executive brief.

    Rendered from the report loaded off the disk cache, never from inside the
    "Investigate" button block: the download button triggers a Streamlit
    rerun, at which point that button reads False and an inline render would
    blank the page mid-demo.
    """
    st.markdown("**Take it away as a PDF**")
    st.caption("A two-page brief for the account owner — nothing in it that is not on this page.")

    try:
        with st.spinner("Building the PDF…"):
            from reporting.pdf import pdf_filename

            pdf_bytes = build_pdf_cached(json.dumps(report), PDF_DEPTH)
    except Exception as e:
        # The PDF is a convenience on top of a report the page has already
        # rendered in full. A failure here must not blank the analysis.
        st.warning(f"Could not build the PDF: {e}. The verdict above is unaffected.")
        return

    get, _ = st.columns([1, 2])
    get.download_button(
        "Download the brief (PDF)",
        data=pdf_bytes,
        file_name=pdf_filename(report, PDF_DEPTH),
        mime="application/pdf",
        type="primary",
        width="stretch",
    )
    get.caption(f"{len(pdf_bytes) / 1024:.0f} KB")


def compare_selected_accounts(digest, choice_label):
    """Run the cross-account comparison. Returns (result, error).

    Mirrors investigate_account: same client construction, same provider
    error handling, so a missing key or a rate limit reads the same here as
    it does on the single-account path.
    """
    spec = MODEL_CHOICES[choice_label]
    provider, model = spec["provider"], spec["model"]
    key_hint = "GROQ_API_KEY" if provider == "Groq" else "ANTHROPIC_API_KEY"

    try:
        client = get_live_client(provider, groq_model=model if provider == "Groq" else None)
    except Exception as e:
        return None, f"Could not initialize the model client: {e}. Set {key_hint} in .env."

    error_module = __import__("groq") if provider == "Groq" else __import__("anthropic")
    try:
        return compare_accounts(client, digest, model=model), None
    except error_module.AuthenticationError:
        return None, f"No valid API credentials for this model — set {key_hint} in .env."
    except error_module.RateLimitError:
        return None, "Rate limited by the model provider. Wait a moment and retry."
    except error_module.APIConnectionError:
        return None, "Could not reach the model provider (network issue)."
    except error_module.APIStatusError as e:
        return None, f"Model provider error ({e.status_code}): {e.message}"
    except ComparisonError as e:
        return None, f"The comparison did not come back in a usable shape: {e}"
    except Exception as e:
        return None, f"Unexpected error during comparison: {e}"


def recommend_decision(decision_input, choice_label):
    """Ask the adviser model what to do. Returns (decision, error).

    Same provider handling as the other two call sites, so a missing key or
    a rate limit reads identically wherever it happens.
    """
    spec = MODEL_CHOICES[choice_label]
    provider, model = spec["provider"], spec["model"]
    key_hint = "GROQ_API_KEY" if provider == "Groq" else "ANTHROPIC_API_KEY"

    try:
        client = get_live_client(provider, groq_model=model if provider == "Groq" else None)
    except Exception as e:
        return None, f"Could not initialize the model client: {e}. Set {key_hint} in .env."

    error_module = __import__("groq") if provider == "Groq" else __import__("anthropic")
    try:
        return recommend(client, decision_input, model=model), None
    except error_module.AuthenticationError:
        return None, f"No valid API credentials for this model — set {key_hint} in .env."
    except error_module.RateLimitError:
        return None, "Rate limited by the model provider. Wait a moment and retry."
    except error_module.APIConnectionError:
        return None, "Could not reach the model provider (network issue)."
    except error_module.APIStatusError as e:
        return None, f"Model provider error ({e.status_code}): {e.message}"
    except DecisionError as e:
        return None, f"The recommendation did not come back in a usable shape: {e}"
    except Exception as e:
        return None, f"Unexpected error while drafting the recommendation: {e}"


@st.cache_data(show_spinner=False)
def cached_price_options(report_json: str) -> list:
    """Priced interventions for a finished report. Free — deterministic
    arithmetic — but it runs on every rerun, so it is cached like the rest."""
    return price_options((json.loads(report_json)).get("evidence") or {})


def render_intervention(report: dict, fingerprint: str, account_id: str,
                        model_id: str, choice_label: str) -> None:
    """The priced options, then an optional written recommendation.

    The calculator renders first and never calls a model: a manager can size
    every option, and decide, with no key configured and no network. The
    recommendation is an explicit button on top, so a failure there cannot
    take the numbers down with it.

    The recommendation is offered even when nothing could be priced. A leak
    the pipeline cannot size honestly — an account splitting its orders, say,
    where nothing has actually been lost yet — still deserves a decision; it
    just does not get a rupee figure attached to it.
    """
    st.markdown("### What should we do about it?")

    # Nothing to decide on an account that is fine, or one nobody was
    # willing to call: acting on a deferred verdict is worse than waiting.
    if report.get("verdict") == "healthy" or report.get("defer"):
        render_no_lever(report)
        return

    options = cached_price_options(json.dumps(report))
    colors = active()

    if options:
        render_options(options, colors)
        lever = options[0]["lever"]
    else:
        render_early_warning(report)
        lever = "unpriced"

    # Keyed on the lever rather than on a chosen option: the agent now picks
    # the option itself, so there is one recommendation per account.
    decision = cache.load_decision(fingerprint, account_id, model_id, lever)

    act, note = st.columns([1, 3])
    with act:
        draft_clicked = st.button(
            "Recommend what to do",
            type="primary",
            use_container_width=True,
            disabled=decision is not None,
            help="Already recommended — the saved answer is shown below."
            if decision is not None
            else "One model call: picks the move, says what it gets you, and lists the steps.",
        )
    with note:
        st.write("")
        st.caption("The AI picks one of the options above, says what it is worth, and lists the steps.")

    if decision is None and draft_clicked:
        with st.spinner("Drafting the recommendation…"):
            decision_input = build_decision_input(report, options)
            decision, decision_error = recommend_decision(decision_input, choice_label)
        if decision_error:
            st.warning(f"{decision_error} Anything above is unaffected.")
            decision = None
        else:
            cache.save_decision(fingerprint, account_id, model_id, lever, decision)
            st.rerun()

    if decision:
        render_decision(decision, colors)


DIMENSION_LABELS = {
    "revenue": "Revenue", "margin": "Margin", "discount": "Discount",
    "tier_mix": "Value mix", "category_mix": "Category mix",
    "order_pattern": "Order pattern", "returns": "Returns",
}


def render_ingestion_report(report: dict) -> None:
    """How the raw file was resolved to the canonical schema.

    A mapping table rather than raw JSON, because on an unfamiliar file this
    is the first thing worth checking: which of your columns became which
    field, what had to be derived, and — most importantly — which analyses
    the file therefore cannot support. A dimension that could not be measured
    must never be mistaken for one that was measured and found fine.
    """
    rows, dropped = report.get("rows_total_raw", 0), report.get("rows_dropped_invalid", 0)
    left, right = st.columns(2)
    left.metric("Rows read", f"{rows:,}")
    right.metric("Rows dropped", f"{dropped:,}",
                 help="Unparseable date, non-numeric revenue, or missing account id.")
    if dropped:
        st.caption("Dropped for: " + ", ".join(
            f"{reason.replace('_', ' ')} ({count})"
            for reason, count in (report.get("row_drop_reasons") or {}).items() if count
        ))

    inferred = {note.split(" ")[0]: note for note in report.get("fields_inferred", [])}
    st.markdown("**Column mapping** — your columns → the fields the pipeline needs")
    st.dataframe(
        pd.DataFrame([
            {
                "Field": field,
                "Came from": (
                    source_column if source_column
                    else f"derived: {inferred[field].split('(', 1)[-1].rstrip(')')}"
                    if field in inferred else "not present"
                ),
                "Status": "matched" if source_column else "derived" if field in inferred else "missing",
            }
            for field, source_column in (report.get("column_mapping") or {}).items()
        ]),
        width="stretch", hide_index=True,
    )

    for note in report.get("fields_inferred", []):
        if note.split(" ")[0] not in (report.get("column_mapping") or {}):
            st.caption(f"Derived: {note}")

    # "returns" is excluded from the missing list: it flags whether the file
    # CONTAINS return lines, not whether returns could be analysed. A file
    # with no credit notes is not missing a column, and listing it under
    # "these columns are absent" is simply false
    # (see timeline.PRESENCE_ONLY_DIMENSIONS).
    dimensions = report.get("analysis_dimensions") or {}
    available = [DIMENSION_LABELS.get(k, k) for k, ok in dimensions.items() if ok]
    missing = [
        DIMENSION_LABELS.get(k, k) for k, ok in dimensions.items()
        if not ok and k not in PRESENCE_ONLY_DIMENSIONS
    ]
    st.markdown("**What this file can and cannot show**")
    if available:
        st.success("Analysable: " + ", ".join(available))
    if missing:
        st.warning(
            "Not analysable — these columns are absent: " + ", ".join(missing)
            + ". They were not measured, which is not the same as measured and fine."
        )

    with st.expander("Raw ingestion report (JSON)"):
        st.json(report)


@st.dialog("How this file was read", width="large")
def show_ingestion_modal(report: dict) -> None:
    """Same report as before — a modal, not an expander that pushes the page."""
    render_ingestion_report(report)


@st.dialog("Data source", width="small")
def show_data_source_modal() -> None:
    """Pick a file without a popover hanging off the sidebar."""
    current = st.session_state.get("upload_name")
    if current:
        st.caption(f"Using **{current}**. Drop a different file to replace it.")
    else:
        st.caption(
            "Leave this empty to use the reference dataset — the official Quessathon "
            "workbook, already extracted."
        )
    uploaded = st.file_uploader("Use your own transaction file", type=["csv", "xlsx"])
    if uploaded is not None and st.session_state.get("upload_name") != uploaded.name:
        st.session_state["upload_bytes"] = uploaded.getvalue()
        st.session_state["upload_name"] = uploaded.name
        st.rerun()
    if current and st.button("Use the reference dataset instead", type="tertiary"):
        st.session_state.pop("upload_bytes", None)
        st.session_state.pop("upload_name", None)
        st.rerun()


# --- Data (cached: every widget interaction re-runs the whole script) ------
#
# Without these, toggling the model radio re-read the CSV, re-ran ingestion
# and rebuilt the evidence pack before drawing a single pixel — which is what
# made switching models feel slow. None of that work depends on the widget
# that changed, so it is cached on its inputs.

@st.cache_data(show_spinner=False)
def load_reference_data():
    return ingest(str(MERIDIAN_TRANSACTIONS))


@st.cache_data(show_spinner=False)
def load_uploaded_data(file_bytes: bytes, filename: str):
    """Keyed on the file's CONTENT, not the upload widget's object identity —
    Streamlit hands back a new object every rerun, which would defeat the
    cache entirely."""
    import io
    buffer = io.BytesIO(file_bytes)
    buffer.name = filename
    return ingest(buffer)


@st.cache_data(show_spinner=False, hash_funcs={pd.DataFrame: cache.dataframe_identity})
def cached_evidence_pack(df: pd.DataFrame, account_id: str) -> dict:
    # The classifier's second opinion rides along in the pack: deterministic,
    # so it is computed once with the rest and shown both before and after
    # the AI step. Without a trained model file it is simply `available: false`.
    return attach_model_opinion(build_evidence_pack(df, account_id))


@st.cache_data(show_spinner=False, hash_funcs={pd.DataFrame: cache.dataframe_identity})
def cached_account_fingerprint(df: pd.DataFrame, account_id: str) -> str:
    return cache.account_fingerprint(df, account_id)


@st.cache_data(show_spinner=False, hash_funcs={pd.DataFrame: cache.dataframe_identity})
def cached_book_fingerprint(df: pd.DataFrame) -> str:
    """One hash for the whole file — what a book-level chat is keyed by, so
    a replaced upload starts a clean chat list rather than quoting figures
    that no longer exist."""
    return cache.selection_fingerprint(df, list(df["account_id"].unique()))


@st.cache_data(show_spinner=False, hash_funcs={pd.DataFrame: cache.dataframe_identity})
def cached_account_index(df: pd.DataFrame) -> tuple[dict, list]:
    names = (
        df.drop_duplicates("account_id").set_index("account_id")["account_name"].to_dict()
        if "account_name" in df.columns else {}
    )
    return names, sorted(df["account_id"].unique())


@st.cache_data(show_spinner=False, hash_funcs={pd.DataFrame: cache.dataframe_identity})
def cached_account_catalogue(df: pd.DataFrame) -> pd.DataFrame:
    """One row per account, from the same packs Detect already builds."""
    return build_account_catalogue(df)


@st.cache_data(show_spinner=False, hash_funcs={pd.DataFrame: cache.dataframe_identity})
def cached_comparison_digest(df: pd.DataFrame, account_ids: tuple[str, ...]) -> dict:
    """What the comparison model is shown. Free — it only re-reads the
    deterministic stages — but it runs on every rerun of the page, so it is
    cached on the selection like everything else here."""
    return build_comparison_digest(df, list(account_ids))


# --- Home sidebar + data ---------------------------------------------------

if st.session_state.get("home_page") not in {p["key"] for p in PAGES}:
    # Also catches a browser session still pointing at a page that has since
    # been hidden, which would otherwise fail the page_spec lookup below.
    st.session_state["home_page"] = "data"
if st.session_state.get("model_choice") not in MODEL_CHOICES:
    st.session_state["model_choice"] = DEFAULT_MODEL_CHOICE
if "selected_account" not in st.session_state:
    st.session_state["selected_account"] = None

with st.sidebar:
    theme.sidebar_brand("Revenue Leakage Investigator")
    current_page = st.session_state["home_page"]
    for page in PAGES:
        st.button(
            page["label"],
            key=f"navpage-{page['key']}",
            type="primary" if current_page == page["key"] else "secondary",
            use_container_width=True,
            on_click=_set_home_page,
            args=(page["key"],),
        )
    st.divider()
    open_data_source = st.button(
        "Choose your data source", icon=":material/folder_open:", use_container_width=True,
        key="sidebar-source",
    )

# Opened from the main script, not inside the sidebar — otherwise Streamlit
# parks the dialog over the left column instead of the centre of the page.
if open_data_source:
    show_data_source_modal()

if "upload_bytes" in st.session_state:
    using_reference = False
    try:
        df, ingestion_report = load_uploaded_data(
            st.session_state["upload_bytes"], st.session_state["upload_name"],
        )
    except IngestionError as e:
        st.error(f"Could not process this file: {e}")
        st.stop()
else:
    using_reference = True
    if not MERIDIAN_TRANSACTIONS.exists():
        st.error(
            f"Reference data not found at {MERIDIAN_TRANSACTIONS}. "
            "Run `python scripts/prepare_dataset.py` to extract it from the workbook."
        )
        st.stop()
    df, ingestion_report = load_reference_data()


# --- Account (the one control everything hangs off) ------------------------

names, accounts = cached_account_index(df)

# A leftover id from a previous file must not silently analyse the wrong book.
if st.session_state["selected_account"] not in set(accounts):
    st.session_state["selected_account"] = None

account_id = st.session_state["selected_account"]
choice_label = st.session_state["model_choice"]
model_id = MODEL_CHOICES[choice_label]["model"]
fingerprint = cached_account_fingerprint(df, account_id) if account_id else None
report = cache.load(fingerprint, account_id, model_id) if account_id else None
if report is not None:
    st.session_state.setdefault("investigation_reports", {})[account_id] = report

# Packs are only built for pages that show them. Attribute / Prioritise read
# the finished report; hashing and assembling the pack on those clicks was
# wasted work that showed up as lag. The account book is its own cached table.
needs_pack = account_id is not None and (
    current_page in ("data", "verdict", "ask")
    or (report is not None and "evidence_timeline" not in report)
)
pack = cached_evidence_pack(df, account_id) if needs_pack else None

# A report cached before the evidence timeline existed has a verdict but no
# case behind it, which would render a PDF full of blanks. Both keys are
# deterministic functions of the evidence pack, and the fingerprint that
# found this file already guarantees the pack matches the data the verdict
# was computed from — so they can be rebuilt here rather than paying for the
# investigation again.
if pack is not None and report is not None and "evidence_timeline" not in report:
    report["evidence"] = pack
    report["evidence_timeline"] = build_timeline(pack)

page_spec = next(p for p in PAGES if p["key"] == current_page)


def _file_info_button() -> None:
    if st.button(
        "How this file was read", type="tertiary", icon=":material/info:", key="file-info",
    ):
        show_ingestion_modal(ingestion_report)


def _account_identity_bar() -> None:
    """Selected account on pages that are not the book, plus a way back."""
    back, who, info = st.columns([1.2, 3, 1.4], vertical_alignment="center")
    with back:
        st.button(
            "All accounts",
            icon=":material/arrow_back:",
            type="tertiary",
            on_click=_back_to_account_book,
            help="Back to the account list",
        )
    with who:
        label = f"{account_id} — {names[account_id]}" if names.get(account_id) else account_id
        st.caption(label)
    with info:
        _file_info_button()


def _need_an_account() -> None:
    theme.page(page_spec["title"], page_spec["caption"])
    st.info("Pick an account from Detect — the table is the starting point.")
    if st.button("Open the account list", type="primary"):
        _back_to_account_book()
        st.rerun()


def render_data_page() -> None:
    if account_id is None:
        theme.page(page_spec["list_title"], page_spec["list_caption"])
        book = cached_account_catalogue(df)
        remembered = st.session_state.setdefault("investigation_reports", {})
        catalogue = apply_investigation_cache(
            book,
            {
                str(aid): (
                    remembered.get(str(aid))
                    or cache.load_for_account(
                        str(aid), cached_account_fingerprint(df, aid), model_id,
                    )
                )
                for aid in book["account_id"]
            },
        )
        opened = render_account_book(
            catalogue,
            trailing=_file_info_button,
        )
        if opened:
            _open_account(opened)
            st.rerun()
        return

    _account_identity_bar()
    theme.page(page_spec["title"], page_spec["caption"])
    _verdict_at_a_glance()
    render_account_dashboard(df, account_id, pack=pack)


def _verdict_at_a_glance() -> None:
    """The AI's call on the first screen.

    The verdict is a paid model call that lives on Investigate, but the
    manager opening an account wants the headline before the evidence. So:
    if this account has already been investigated with the chosen model,
    show the banner and the money here (the same header Investigate
    renders); if not, say so and offer to run it in place, so the banner
    appears without leaving the page. The reasoning and the PDF stay on
    Investigate.
    """
    if report is not None:
        render_verdict_header(report, pack=pack)
        return

    note, act = st.columns([3, 1], vertical_alignment="center")
    with note:
        st.caption(
            f"Not yet analysed by the AI — the evidence below is complete on its own. "
            "Run the AI to add the verdict and the rupees at risk here."
        )
    with act:
        run_here = st.button(
            "Analyse with AI", type="primary", width="stretch", key="detect-investigate",
            help="One model call; the result is saved and reused on every page.",
        )
    if run_here:
        with st.spinner(f"Investigating {account_id}…"):
            fresh, error = investigate_account(df, account_id, pack, choice_label)
        if error:
            st.error(error)
        else:
            cache.save(fingerprint, account_id, model_id, fresh)
            st.rerun()


def render_verdict_page() -> None:
    if account_id is None:
        _need_an_account()
        return
    _account_identity_bar()
    theme.page(page_spec["title"], page_spec["caption"])
    # The control row keeps the SAME two columns and the SAME single button in
    # every state — only the button's enabled-ness and the status text change.
    # Swapping the button's label and type between "Run" and "Re-run" made the row
    # reflow on every rerun, which read as the page flickering. There is no
    # model picker: the app runs on DEFAULT_MODEL_CHOICE.
    act, status = st.columns([1, 3])

    with act:
        investigate_clicked = st.button(
            "Investigate with AI",
            type="primary",
            width="stretch",
            disabled=report is not None,
            help="Already analysed with this model — the saved result is shown below."
            if report is not None else "Runs the single LLM call for this account.",
        )

    with status:
        if report is not None:
            st.caption("Already reviewed — this is the saved result.")

    if report is None and investigate_clicked:
        with st.spinner(f"Investigating {account_id}…"):
            fresh, error = investigate_account(df, account_id, pack, choice_label)
        if error:
            st.error(error)
            st.info("The evidence on Detect is unaffected — only the model step failed.")
        else:
            cache.save(fingerprint, account_id, model_id, fresh)
            st.session_state.setdefault("investigation_reports", {})[account_id] = fresh
            st.rerun()

    if report:
        render_verdict(report, pack=pack)
        render_intervention(report, fingerprint, account_id, model_id, choice_label)
        render_pdf_download(report)
    else:
        st.info(f"{account_id} has not been reviewed yet — click **Investigate with AI** above.")


def render_score_page() -> None:
    if account_id is None:
        _need_an_account()
        return
    _account_identity_bar()
    theme.page(page_spec["title"], page_spec["caption"])
    try:
        answer_key = load_answer_key()
    except AnswerKeyUnavailable:
        answer_key = None

    key_row = (answer_key or {}).get(account_id)

    if not using_reference or key_row is None:
        st.caption(
            "The Answer Key only covers the reference dataset's 18 accounts, so there is "
            "nothing to score this account against."
        )
    elif not report:
        st.caption(
            f"Run the investigation under Investigate and this will score it. "
            f"This account is the **{key_row['archetype']}** case."
        )
        with st.expander("What this account is designed to test (spoiler)"):
            st.markdown(f"**Tests:** {key_row['what_it_tests']}")
            st.markdown(f"**Really happening:** {key_row['what_is_really_happening']}")
    else:
        result = score_report(report, key_row)

        # Full-width verdict line, then the detail beneath it. The previous
        # two-column split put a short badge beside three long paragraphs, so the
        # columns had wildly different heights and nothing lined up.
        outcome_line = (
            f"Expected **{result['expected_outcome']}** · agent said **{result['actual_outcome']}**"
        )
        if result["outcome_match"]:
            st.success(f"**Correct.** {outcome_line}")
        else:
            st.error(f"**Missed.** {outcome_line}")

        facts = st.columns(2)
        facts[0].markdown(f"**Archetype**  \n{result['archetype']}")
        confidence_note = "not scored"
        if result["confidence_match"] is not None:
            mark = "matches" if result["confidence_match"] else "differs"
            confidence_note = (
                f"expected {result['expected_confidence']}, "
                f"got {result['actual_confidence']} — {mark}"
            )
        facts[1].markdown(f"**Confidence**  \n{confidence_note}")

        st.markdown(f"**What this account tests**  \n{result['what_it_tests']}")
        st.markdown(f"**What is really happening**  \n{result['what_is_really_happening']}")

        st.caption(
            "The Answer Key only ever scores a finished verdict — it is never shown to the "
            "agent. It exists for this dataset alone; an unseen file will have none."
        )


def render_compare_page() -> None:
    if account_id:
        _account_identity_bar()
    else:
        _, info = st.columns([4, 1])
        with info:
            _file_info_button()
    theme.page(page_spec["title"], page_spec["caption"])
    if account_id is None:
        st.caption("Pick accounts below, or open one from Detect first.")
    # Seeded from the AI verdict, not from the whole book: this page compares
    # the account you just investigated against others you choose, so it starts
    # with that one account and nothing else. With no investigation yet there
    # is nothing to compare from, and the picker stays empty.
    #
    # The widget key carries whether the account has been investigated, because
    # Streamlit applies `default` only when it first sees a key. Without that,
    # the picker rendered empty before the investigation would stay empty after
    # it, and the account you just ran would never appear.
    default_selection = [account_id] if report else []

    selected_accounts = st.multiselect(
        "Accounts to compare",
        accounts,
        default=default_selection,
        key=f"compare_selection_{account_id}_{'investigated' if report else 'new'}",
        format_func=lambda a: f"{a} — {names[a]}" if names.get(a) else a,
        help=(
            "Starts with the account you investigated. Add the accounts you want to read it "
            "against — up to "
            f"{MAX_ACCOUNTS}."
        ),
    )

    if not selected_accounts and not report:
        st.info(
            "Nothing to compare yet — investigate an account under Investigate "
            "and it will appear here, ready to read against any others you pick."
        )
    elif len(selected_accounts) < MIN_ACCOUNTS:
        st.info(
            f"Add at least one more account to compare {account_id} against."
            if selected_accounts
            else f"Pick at least {MIN_ACCOUNTS} accounts to compare."
        )
    elif len(selected_accounts) > MAX_ACCOUNTS:
        st.warning(
            f"{len(selected_accounts)} accounts selected — a comparison reads at most "
            f"{MAX_ACCOUNTS} at once. Remove {len(selected_accounts) - MAX_ACCOUNTS} to continue."
        )
    else:
        comparison_digest = cached_comparison_digest(df, tuple(selected_accounts))
        selection_fp = cache.selection_fingerprint(df, selected_accounts)

        # Reuse silently when this exact set of accounts, model and data has been
        # compared before — the same rule the single-account cache follows.
        comparison = cache.load_comparison(selection_fp, model_id)

        run_col, note_col = st.columns([1, 3])
        with run_col:
            compare_clicked = st.button(
                "Compare with AI",
                type="primary",
                width="stretch",
                disabled=comparison is not None,
                help="Already compared — the saved result is shown below."
                if comparison is not None
                else f"One model call reading all {len(selected_accounts)} accounts together.",
            )
        with note_col:
            st.write("")
            if comparison is not None:
                st.caption(
                    f"These {len(selected_accounts)} accounts have already been compared with the "
                    "AI — this is the saved result."
                )

        if comparison is None and compare_clicked:
            with st.spinner(
                f"Comparing {len(selected_accounts)} accounts…"
            ):
                comparison, compare_error = compare_selected_accounts(
                    comparison_digest, choice_label
                )
            if compare_error:
                st.error(compare_error)
                st.info("The per-account analysis is unaffected — only the comparison failed.")
                comparison = None
            else:
                cache.save_comparison(selection_fp, model_id, comparison)
                st.rerun()

        if comparison:
            render_comparison(comparison, comparison_digest)


def render_ask_page() -> None:
    """AryaChat: the analyst for the whole book. Needs no account picked —
    the manager names one in the question (or asks across the book) and
    the model resolves it and calls the tool that holds the answer. An
    account opened elsewhere in the app seeds a new chat's focus. The same
    model picker as Investigate; the open-source model is the practical
    default because a chat turn has to come back in seconds."""
    if account_id:
        _account_identity_bar()
    else:
        _, info = st.columns([4, 1])
        with info:
            _file_info_button()
    theme.page(page_spec["title"], page_spec["caption"])
    chosen = DEFAULT_MODEL_CHOICE
    spec = MODEL_CHOICES[chosen]
    model = spec["model"]

    def make_client():
        return get_live_client(spec["provider"], groq_model=model if spec["provider"] == "Groq" else None)

    # The chat reaches the pipeline through these two: packs are the same
    # cached ones Detect draws, verdicts are whatever this model has saved
    # for an account (so a brief says "not analysed" rather than borrowing
    # another model's call).
    def pack_for(acc: str) -> dict:
        return cached_evidence_pack(df, acc)

    def report_for(acc: str):
        return cache.load(cached_account_fingerprint(df, acc), acc, model)

    render_aryachat(
        df, cached_book_fingerprint(df), model, chosen, make_client,
        pack_for, report_for, names, initial_focus=account_id,
    )


if current_page == "data":
    render_data_page()
elif current_page == "verdict":
    render_verdict_page()
# elif current_page == "score":
#     render_score_page()
# elif current_page == "compare":
#     render_compare_page()
else:
    render_ask_page()
