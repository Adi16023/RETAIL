"""
Demo UI (Stage 8 surface, per plan.md): upload a transaction CSV, run the
full 7-stage pipeline, and render the verdict, evidence, financial impact,
and prioritization. Two modes:

  - "Analyze a CSV": the real pipeline, including a live Stage 4 LLM call.
    Deterministic results (Stages 1-3) are shown even if Stage 4 fails, so
    a live-failure degrades gracefully instead of blanking the whole page.
  - "View offline demo": loads a pre-computed cached report from
    demo_cache/ — the live-failure fallback required by plan.md Phase 4
    ("offline fallback plan (cached run of a known archetype)"). Never
    silently substituted for a real uploaded file's failed run — the user
    picks it explicitly.

No hard-coded absolute paths — everything is relative to this file or
comes from the uploaded file object.
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

from pipeline.agent import DEFAULT_MODEL, AgentError
from pipeline.evidence import build_evidence_pack
from pipeline.ingest import IngestionError, account_sufficiency, ingest
from pipeline.impact import compute_impact
from pipeline.prioritize import prioritize
from pipeline.report import assemble_report

DEMO_CACHE_DIR = REPO_ROOT / "demo_cache"

st.set_page_config(page_title="Revenue Leakage Investigator", layout="wide")


def get_live_client(provider: str, groq_model: str | None = None):
    """Construct the real LLM client for the chosen provider. Import is
    local so the app still runs (in offline-demo mode) even if a package
    or credentials aren't available.

    Groq is a temporary stand-in for Claude (see pipeline/groq_client.py)
    while the team doesn't yet have Anthropic API access — swapping back
    is just picking "Anthropic" in the sidebar once a key exists; nothing
    else in the pipeline is provider-aware."""
    if provider == "Groq":
        from pipeline.groq_client import GroqShimClient
        return GroqShimClient(model_override=groq_model)
    import anthropic
    return anthropic.Anthropic()


def run_agent_with_error_handling(client, df, account_id, evidence_pack, model, provider):
    from pipeline.agent import investigate

    error_module = __import__("groq") if provider == "Groq" else __import__("anthropic")
    provider_name = "Groq" if provider == "Groq" else "Anthropic"
    key_hint = "GROQ_API_KEY" if provider == "Groq" else "ANTHROPIC_API_KEY"

    try:
        return investigate(client, df, account_id, evidence_pack, model=model), None
    except error_module.AuthenticationError:
        return None, f"No valid {provider_name} API credentials — set {key_hint}, or use 'View offline demo' instead."
    except error_module.RateLimitError:
        return None, f"Rate limited by the {provider_name} API. Wait a moment and retry, or use 'View offline demo'."
    except error_module.APIConnectionError:
        return None, f"Could not reach the {provider_name} API (network issue). Use 'View offline demo' if this persists."
    except error_module.APIStatusError as e:
        return None, f"{provider_name} API error ({e.status_code}): {e.message}"
    except AgentError as e:
        return None, f"Agent did not reach a verdict: {e}"
    except Exception as e:
        return None, f"Unexpected error during investigation: {e}"


def render_verdict_badge(verdict: str, defer: bool):
    if defer or verdict == "insufficient_data":
        st.warning(f"**Verdict: {verdict.replace('_', ' ').title()}** — evidence was insufficient for a confident call.")
    elif verdict == "leakage_detected":
        st.error(f"**Verdict: {verdict.replace('_', ' ').title()}**")
    else:
        st.success(f"**Verdict: {verdict.replace('_', ' ').title()}**")


def render_report(report: dict, cached: bool = False):
    if cached:
        st.info(
            f"Cached offline demo run ({report.get('_cache_metadata', {}).get('archetype', 'unknown')} archetype) — "
            "not a live LLM call. Used when live API access isn't available."
        )

    col1, col2, col3 = st.columns(3)
    col1.metric("Account", report["account_id"])
    col2.metric("Confidence", report["confidence"].title())
    col3.metric("Temporary / Structural", report["temporary_or_structural"].replace("_", " ").title())

    render_verdict_badge(report["verdict"], report["defer"])
    st.write(report["narrative"])

    if report["defer"] and report["data_needed_if_deferring"]:
        st.write("**What data would resolve this:**")
        for item in report["data_needed_if_deferring"]:
            st.write(f"- {item}")

    if report["attributed_categories"]:
        st.subheader("Attributed categories")
        st.write(", ".join(report["attributed_categories"]))

    st.subheader("Cited evidence")
    for fact in report["cited_evidence"]:
        st.write(f"- {fact}")

    impact = report["financial_impact"]
    if impact["per_category"]:
        st.subheader("Financial impact")
        rows = [c for c in impact["per_category"] if c["quantifiable"]]
        if rows:
            st.dataframe(pd.DataFrame(rows), width="stretch")
        unquantified = [c for c in impact["per_category"] if not c["quantifiable"]]
        for c in unquantified:
            st.caption(f"{c['category']}: not quantifiable — {c['reason']}")
        st.metric("Total monthly revenue at risk", f"₹{impact['total_monthly_revenue_at_risk']:,.0f}")

    priority = report["prioritization"]
    st.subheader("Prioritization")
    st.write(f"**Priority: {priority['priority']}** — {priority.get('reason', '')}")
    if priority.get("churn_risk_projection"):
        crp = priority["churn_risk_projection"]
        st.write(
            f"If this trend persists: projected loss of ₹{crp['projected_12_month_loss_if_unaddressed']:,.0f} "
            f"over 12 months (₹{crp['monthly_run_rate_loss']:,.0f}/month)."
        )
        st.caption(crp["basis"])

    if report["recommended_actions"]:
        st.subheader("Recommended actions")
        for action in report["recommended_actions"]:
            st.write(f"- {action}")

    with st.expander("Data sufficiency"):
        st.json(report["data_sufficiency"])


st.title("Revenue Leakage Investigator")
st.caption("Detect → Investigate → Attribute → Prioritise — Quessathon Retail challenge")

mode = st.sidebar.radio("Mode", ["Analyze a CSV", "View offline demo"])

provider = st.sidebar.radio(
    "LLM provider", ["Groq", "Anthropic"],
    help="Groq is a temporary stand-in while the team doesn't have Claude API access yet "
         "(see pipeline/groq_client.py). Switch to Anthropic once a key is available — "
         "nothing else in the pipeline needs to change.",
)
if provider == "Groq":
    from pipeline.groq_client import DEFAULT_GROQ_MODEL
    model = st.sidebar.text_input("Groq model", value=DEFAULT_GROQ_MODEL)
else:
    model = st.sidebar.text_input("Model", value=DEFAULT_MODEL)

if mode == "View offline demo":
    cache_files = sorted(DEMO_CACHE_DIR.glob("*.json")) if DEMO_CACHE_DIR.exists() else []
    if not cache_files:
        st.error(f"No cached demo reports found in {DEMO_CACHE_DIR}. Run scripts/generate_demo_cache.py first.")
    else:
        choice = st.selectbox("Cached archetype", [f.stem for f in cache_files])
        report = json.loads((DEMO_CACHE_DIR / f"{choice}.json").read_text())
        render_report(report, cached=True)

else:
    uploaded = st.file_uploader("Upload a transaction CSV", type=["csv"])
    if uploaded is not None:
        try:
            df, ingestion_report = ingest(uploaded)
        except IngestionError as e:
            st.error(f"Could not process this file: {e}")
            st.stop()

        with st.expander("Ingestion report"):
            st.json(ingestion_report)

        account_ids = sorted(df["account_id"].unique())
        account_id = st.selectbox("Account to investigate", account_ids)

        sufficiency = account_sufficiency(df).get(account_id)
        if sufficiency:
            st.caption(
                f"Data sufficiency for {account_id}: **{sufficiency['label']}** "
                f"({sufficiency['history_months']} months, {sufficiency['order_count']} orders, "
                f"{sufficiency['category_count']} categories)"
            )

        if st.button("Run investigation", type="primary"):
            with st.spinner("Running deterministic analysis (Stages 1-3)..."):
                evidence_pack = build_evidence_pack(df, account_id)

            with st.expander("Evidence pack (what the agent sees)"):
                st.json(evidence_pack)

            with st.spinner(f"Running investigation agent (Stage 4, via {provider})..."):
                try:
                    client = get_live_client(provider, groq_model=model if provider == "Groq" else None)
                except Exception as e:
                    key_hint = "GROQ_API_KEY" if provider == "Groq" else "ANTHROPIC_API_KEY"
                    st.error(
                        f"Could not initialize the {provider} client: {e}. "
                        f"Set {key_hint}, or switch to 'View offline demo' in the sidebar."
                    )
                    st.stop()

                verdict, error = run_agent_with_error_handling(client, df, account_id, evidence_pack, model, provider)

            if error:
                st.error(error)
                st.info(
                    "Deterministic evidence (Stages 1-3) above is still valid and traceable — "
                    "only the LLM reasoning step failed. Retry, or use 'View offline demo' in the sidebar."
                )
                st.stop()

            impact = compute_impact(evidence_pack, verdict["attributed_categories"])
            priority = prioritize(impact, verdict)
            report = assemble_report(account_id, evidence_pack, verdict, impact, priority)
            render_report(report)
    else:
        st.write("Upload a CSV to begin, or switch to 'View offline demo' in the sidebar.")
