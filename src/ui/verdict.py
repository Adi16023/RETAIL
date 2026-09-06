"""
The investigation result, laid out for the person who has to act on it.

A verdict is only useful if a manager can answer three questions from it in
under a minute: what is the call, how much money, and why should I believe
it. The page is ordered exactly that way — banner, money, narrative, the
facts as plain sentences — with one collapsed dropdown holding the dated
timeline for anyone who wants to see when each thing moved.

Actions are deliberately absent: the decision step owns "what to do", and a
second list here made the screen say two things.

The statistical classifier has no section of its own. Its probability for
the outcome the AI reached is shown in brackets on the confidence line —
one number, honest either way: a high figure corroborates, a low one says
the numbers alone did not point here.

Every figure shown here was computed deterministically upstream. The model
chose which of them to cite; it never produced one.
"""

from __future__ import annotations

from html import escape

import pandas as pd
import streamlit as st

from pipeline.timeline import READS_AS_ORDER

from .palette import active
from . import theme

VERDICT_PRESENTATION = {
    "leakage_detected": ("critical", "🔴", "Leakage detected"),
    "healthy": ("good", "🟢", "Healthy"),
    "insufficient_data": ("warning", "🟡", "Not enough evidence to call"),
}

PRIORITY_PRESENTATION = {
    "High": ("critical", "Act now"),
    "Medium": ("serious", "Plan a response"),
    "Low": ("warning", "Monitor"),
    "None": ("good", "No action needed"),
    "Deferred": ("warning", "Send to a human"),
}

DIMENSION_LABELS = {
    "revenue": "Revenue",
    "margin": "Margin",
    "discount": "Discount",
    "category_mix": "Category mix",
    "tier_mix": "Value mix",
    "order_pattern": "Order pattern",
}

_OUTCOME_PROBABILITY = {"FLAG": "p_flag", "NO_FLAG": "p_no_flag", "DEFER": "p_defer"}


def _money(value) -> str:
    return "—" if value in (None, "") else f"₹{value:,.0f}"


def _probability(p) -> str:
    # Capped at 99: a classifier is never certain, and "100%" on a banner
    # would claim it is.
    return f"{min(round(p * 100), 99):.0f}%"


def _ml_confidence(report: dict) -> str | None:
    """The classifier's probability for the outcome the AI reached, or None
    when no model was available for this run."""
    opinion = report.get("model_opinion") or {}
    agreement = report.get("model_agreement")
    if not opinion.get("available") or not agreement:
        return None
    p = opinion.get(_OUTCOME_PROBABILITY.get(agreement.get("agent_outcome"), ""))
    return None if p is None else _probability(p)


def _banner(report: dict, colors: dict) -> None:
    role, icon, label = VERDICT_PRESENTATION.get(
        report["verdict"], ("warning", "⚪", report["verdict"])
    )
    if report.get("defer"):
        role, icon, label = "warning", "🟡", "Deferred — not enough evidence to call"

    priority = (report.get("prioritization") or {}).get("priority", "None")
    priority_role, priority_hint = PRIORITY_PRESENTATION.get(priority, ("warning", ""))

    shape = (report.get("temporary_or_structural") or "").replace("_", " ")
    title = f"{icon} {label}"
    if shape and shape != "not applicable":
        title += f" · {shape.capitalize()}"

    # One line per source, so the reader never has to work out whose number
    # is whose: the AI agent is the judge and states a confidence level; the
    # statistical model is a second witness and states a probability.
    lines = [
        f"AI agent — <strong>{escape(report['confidence'].capitalize())} confidence</strong>",
    ]
    ml = _ml_confidence(report)
    if ml:
        lines.append(f"Statistical model — <strong>{escape(ml)} confidence score</strong>")
    lines.append(
        f"Priority <strong style='color:{colors[priority_role]}'>{escape(str(priority))}</strong>"
        f" — {escape(priority_hint)}"
    )
    theme.banner(role, title, lines)

    if report.get("leak_dimensions"):
        chips = "".join(
            f"<span class='rl-chip'>{escape(DIMENSION_LABELS.get(d, d))}</span>"
            for d in report["leak_dimensions"]
        )
        st.markdown(
            f"<div class='rl-chip-row rl-rise'><div class='rl-chip-kicker'>"
            f"Value is leaving through</div>{chips}</div>",
            unsafe_allow_html=True,
        )


def _money_row(report: dict) -> None:
    impact = report.get("financial_impact") or {}
    margin_impact = impact.get("margin_impact") or {}
    projection = (report.get("prioritization") or {}).get("churn_risk_projection") or {}

    revenue_risk = impact.get("total_monthly_revenue_at_risk") or 0
    margin_risk = margin_impact.get("monthly_margin_at_risk") or 0
    if not revenue_risk and not margin_risk:
        return

    # Whichever exposure is larger is "the cost of doing nothing" — the two
    # are never added, because margin is a slice of revenue.
    twelve_month = max(
        projection.get("projected_12_month_loss_if_unaddressed") or 0,
        projection.get("projected_12_month_margin_loss_if_unaddressed") or 0,
    )
    severity = impact.get("overall_severity_pct_of_baseline")

    columns = st.columns(4)
    columns[0].metric("Revenue at risk / month", _money(revenue_risk))
    columns[1].metric("Margin at risk / month", _money(margin_risk),
                      help="A slice of the revenue figure, not a second pot. Never add the two.")
    columns[2].metric("Cost of doing nothing, 12 months", _money(twelve_month),
                      help="If today's rate simply continues. Not a forecast of further decline.")
    columns[3].metric("Share of baseline at risk",
                      "—" if severity is None else f"{severity * 100:.0f}%",
                      help="The worse of the revenue and margin ratios.")


def _join(items: list[str]) -> str:
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def _reasoning(report: dict) -> None:
    """The narrative, then the facts it rests on, as plain sentences."""
    st.markdown(report["narrative"])

    facts = report.get("cited_evidence") or []
    if not facts:
        st.info("No evidence was cited — that is a bug, not a valid verdict.")
    else:
        st.markdown("**Why**")
        st.markdown("\n".join(f"- {fact}" for fact in facts))

    if report.get("attributed_categories"):
        per_category = {
            c["category"]: c for c in (report.get("financial_impact") or {}).get("per_category", [])
        }
        parts = []
        for category in report["attributed_categories"]:
            entry = per_category.get(category, {})
            if entry.get("quantifiable"):
                detail = f"{_money(entry['monthly_revenue_at_risk'])} a month"
                if entry.get("defected"):
                    detail += ", stopped completely"
                parts.append(f"**{category}** ({detail})")
            else:
                parts.append(f"**{category}**")
        verb = "accounts" if len(parts) == 1 else "account"
        st.markdown(f"{_join(parts)} {verb} for the loss.")
    elif report["verdict"] == "leakage_detected" and not report.get("defer"):
        st.markdown("No single category is responsible: the decline is spread across the book.")

    if report.get("defer") and report.get("data_needed_if_deferring"):
        st.markdown("**What would settle it**")
        st.markdown("\n".join(f"- {item}" for item in report["data_needed_if_deferring"]))


def noise_verdict(significance: dict | None) -> str:
    """How a dimension's permutation test reads to a person: is the shift
    distinguishable from this account's own month-to-month noise? Used by
    the dashboard's status strip."""
    if not significance:
        return "—"
    p = significance.get("p_value")
    if p is None:
        return "too few months to test"
    verdict = "real" if significance.get("significant") else "within normal noise"
    return f"p = {p:.2f} · {verdict}"


_READS_AS_ICONS = {"concern": "🔴", "reassuring": "🟢", "checked": "🔍", "context": "⚪"}


def _timeline_section(report: dict) -> None:
    """The dated evidence log — the same rows the PDF prints, re-keyed by
    date so the reader can see that the discount stepped up two months
    before the margin gave way."""
    timeline = report.get("evidence_timeline") or []
    if not timeline:
        return
    with st.expander("What changed, and when"):
        counts = {kind: sum(1 for e in timeline if e["reads_as"] == kind) for kind in READS_AS_ORDER}
        st.caption(" · ".join(
            f"{_READS_AS_ICONS[kind]} {counts[kind]} {kind}" for kind in READS_AS_ORDER if counts[kind]
        ))
        st.dataframe(
            pd.DataFrame([
                {
                    "": _READS_AS_ICONS.get(event["reads_as"], ""),
                    "When": event["when"],
                    "What happened": event["headline"],
                    "Detail": event["detail"],
                }
                for event in timeline
            ]),
            width="stretch",
            hide_index=True,
        )


def render_verdict(report: dict, pack: dict | None = None, cached: bool = False) -> None:
    """Render one investigation result. `pack` supplies the classifier's
    opinion for a verdict saved before the classifier existed."""
    colors = active()

    # A verdict saved before the classifier existed carries no opinion of its
    # own. The pack's current opinion is still worth showing beside it.
    if report.get("model_opinion") is None and (pack or {}).get("model_opinion", {}).get("available"):
        from pipeline.report import model_agreement
        report = {**report, "model_opinion": pack["model_opinion"],
                  "model_agreement": model_agreement(pack, report)}

    if cached:
        metadata = report.get("_cache_metadata", {})
        st.info(
            f"Cached demo run — **{metadata.get('archetype', 'unknown')}**. The rupee figures and "
            "evidence are computed from the real data; only the model's judgement is scripted."
        )

    heading = report["account_id"]
    if report.get("account_name"):
        heading += f" · {report['account_name']}"
    st.subheader(heading)

    _banner(report, colors)
    _money_row(report)
    _reasoning(report)
    _timeline_section(report)
