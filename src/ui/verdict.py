"""
The investigation result, laid out for the person who has to act on it.

A verdict is only useful if a manager can answer four questions from it:
what is the call, how much money, why should I believe it, and what do I do.
So the page is ordered exactly that way — banner, money, reasoning, action —
rather than as a dump of the model's output fields.

The section that earns the most trust is "Ruled out". It is built from the
evidence pack, not from the model's text: it lists the innocent explanations
that were available for this account and says which ones were checked and
dismissed. Showing the reasoning that did NOT make the cut is what separates
a considered verdict from a confident-sounding one, and it is the difference
between a manager acting on this and ignoring it.

Every figure shown here was computed deterministically upstream. The model
chose which of them to cite; it never produced one.
"""

from __future__ import annotations

from html import escape

import altair as alt
import pandas as pd
import streamlit as st

from pipeline.timeline import PRESENCE_ONLY_DIMENSIONS, READS_AS_ORDER

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


def _money(value) -> str:
    return "—" if value in (None, "") else f"₹{value:,.0f}"


def _pct(value, points: bool = False) -> str:
    if value is None:
        return "—"
    return f"{value:+.1f}pp" if points else f"{value * 100:.1f}%"


def _banner(report: dict, colors: dict) -> None:
    role, icon, label = VERDICT_PRESENTATION.get(
        report["verdict"], ("warning", "⚪", report["verdict"])
    )
    if report.get("defer"):
        role, icon, label = "warning", "🟡", "Deferred — not enough evidence to call"

    priority = (report.get("prioritization") or {}).get("priority", "None")
    priority_role, priority_hint = PRIORITY_PRESENTATION.get(priority, ("warning", ""))

    shape = (report.get("temporary_or_structural") or "").replace("_", " ")
    subtitle = []
    if shape and shape != "not applicable":
        subtitle.append(shape.capitalize())
    subtitle.append(f"{report['confidence'].capitalize()} confidence")

    theme.banner(
        role,
        f"{icon} {label}",
        [
            escape(" · ".join(subtitle)),
            f"Priority <strong style='color:{colors[priority_role]}'>{escape(str(priority))}</strong>"
            f" — {escape(priority_hint)}",
        ],
    )

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

    columns = st.columns(4)
    columns[0].metric("Revenue at risk / month", _money(revenue_risk))
    columns[1].metric("Gross margin at risk / month", _money(margin_risk))

    # Whichever exposure is larger is the one worth quoting as "the cost of
    # doing nothing" — and the two must never be added, because margin is a
    # slice of revenue, not a second pot of money.
    twelve_month = max(
        projection.get("projected_12_month_loss_if_unaddressed") or 0,
        projection.get("projected_12_month_margin_loss_if_unaddressed") or 0,
    )
    columns[2].metric("Exposure over 12 months", _money(twelve_month),
                      help="If today's rate simply continues. Not a forecast of further decline.")

    severity = impact.get("overall_severity_pct_of_baseline")
    columns[3].metric(
        "Severity", "—" if severity is None else f"{severity * 100:.0f}%",
        help="Share of this account's baseline at risk — the worse of the revenue "
             "and margin ratios, never their sum.",
    )

    if revenue_risk and margin_risk:
        st.caption(
            "Revenue and margin at risk overlap — margin is a slice of revenue. "
            "Read whichever is larger; never add them."
        )


def _evidence_tab(report: dict, pack: dict | None, colors: dict) -> None:
    st.markdown("**What the agent based this on**")
    facts = report.get("cited_evidence") or []
    if not facts:
        st.info("No evidence was cited — that is a bug, not a valid verdict.")
    for index, fact in enumerate(facts, start=1):
        st.markdown(
            f"<div style='display:flex;gap:0.6rem;margin-bottom:0.5rem'>"
            f"<div style='flex:0 0 1.4rem;height:1.4rem;border-radius:50%;"
            f"background:{colors['series_1']}22;color:{colors['series_1']};font-size:0.78rem;"
            f"font-weight:700;display:flex;align-items:center;justify-content:center'>{index}</div>"
            f"<div style='flex:1;color:{colors['text_primary']}'>{fact}</div></div>",
            unsafe_allow_html=True,
        )

    if report.get("attributed_categories"):
        st.markdown("**Categories held responsible**")
        per_category = {
            c["category"]: c for c in (report.get("financial_impact") or {}).get("per_category", [])
        }
        for category in report["attributed_categories"]:
            entry = per_category.get(category, {})
            if entry.get("quantifiable"):
                st.markdown(
                    f"- **{category}** — {_money(entry['monthly_revenue_at_risk'])}/month, "
                    f"sustained {entry['persistence_months']} months"
                    + (", stopped completely" if entry.get("defected") else "")
                )
            else:
                st.markdown(f"- **{category}** — not separately quantifiable "
                            f"({entry.get('reason', 'no change-point')})")
    elif report["verdict"] == "leakage_detected" and not report.get("defer"):
        st.info(
            "No single category was named. The decline is spread across the whole book — "
            "naming a scapegoat category would send the account team after the wrong thing."
        )

    if pack:
        with st.expander("The underlying numbers these claims rest on"):
            _dimension_table(pack)
        st.caption(
            "Every figure here was calculated from this account's transactions before the AI "
            "saw them. The AI decided which ones mattered — it did not produce any of the "
            "numbers itself, so you can check each one against the charts above."
        )


def noise_verdict(significance: dict | None) -> str:
    """How a dimension's permutation test reads to a person: is the shift
    distinguishable from this account's own month-to-month noise?"""
    if not significance:
        return "—"
    p = significance.get("p_value")
    if p is None:
        return "too few months to test"
    verdict = "real" if significance.get("significant") else "within normal noise"
    return f"p = {p:.2f} · {verdict}"


def _dimension_table(pack: dict) -> None:
    rows = []
    significance = pack.get("significance") or {}

    revenue = pack.get("revenue_decline") or {}
    overall = pack.get("overall_revenue") or {}
    rows.append(("Revenue", revenue.get("status", "—"),
                 f"{_money(overall.get('baseline_monthly_median'))}/mo → "
                 f"{_money(overall.get('recent_monthly_rate'))}/mo",
                 noise_verdict(significance.get("revenue"))))

    margin = pack.get("margin")
    if margin:
        rows.append(("Margin", margin.get("status", "—"),
                     f"{_pct(margin.get('baseline_margin_pct'))} → "
                     f"{_pct(margin.get('recent_margin_pct'))} "
                     f"({_pct(margin.get('margin_pct_change_pp'), points=True)})",
                     noise_verdict(significance.get("margin"))))

    discount = pack.get("discount")
    if discount:
        rows.append(("Discount", discount.get("status", "—"),
                     f"{_pct(discount.get('baseline_avg_discount_pct'))} → "
                     f"{_pct(discount.get('recent_avg_discount_pct'))} "
                     f"({_pct(discount.get('discount_pct_change_pp'), points=True)})",
                     noise_verdict(significance.get("discount"))))

    tier = pack.get("tier_mix")
    if tier:
        baseline = (tier.get("baseline_share_by_tier") or {}).get("High")
        recent = (tier.get("recent_share_by_tier") or {}).get("High")
        rows.append(("Value mix", tier.get("status", "—"),
                     f"High tier {_pct(baseline)} → {_pct(recent)} "
                     f"({_pct(tier.get('high_tier_share_change_pp'), points=True)})",
                     noise_verdict(significance.get("tier_mix"))))

    pattern = pack.get("order_pattern") or {}
    frequency = pattern.get("order_frequency_pct_change")
    width = pattern.get("basket_width_pct_change")
    rows.append(("Order pattern", pattern.get("status", "—"),
                 "—" if frequency is None else
                 f"orders {frequency * 100:+.0f}%, basket width {width * 100:+.0f}%",
                 noise_verdict(significance.get("order_frequency"))))

    defected = [c["category"] for c in pack.get("category_changes", []) if c["defected"]]
    rows.append(("Categories", "defection" if defected else "none stopped",
                 ", ".join(defected) if defected else "every line still trading", "—"))

    st.dataframe(
        [{"Dimension": d, "Status": s.replace("_", " "), "Movement": m, "Real, or noise?": n}
         for d, s, m, n in rows],
        width="stretch", hide_index=True,
    )
    st.caption(
        "\"Real, or noise?\" shuffles this account's own months and asks how often chance alone "
        "produces a shift this large. A small p means the change is bigger than the account's "
        "ordinary lumpiness; a status can be over the threshold and still be noise."
    )


def _ruled_out_tab(pack: dict | None, report: dict) -> None:
    """The innocent explanations, and what happened to each.

    Built from the evidence pack rather than the model's prose, so this
    section is true even when the model forgets to mention something.
    """
    if not pack:
        st.info("The ruled-out checks are only available on a live run, not a cached demo.")
        return

    st.markdown("**Could this be something harmless?**")
    st.caption(
        "Most accounts that look like leaks are not. These are the ordinary explanations "
        "for a scary-looking number, and what the data says about each one here."
    )

    checks = []

    seasonality = pack.get("seasonality") or {}
    if seasonality.get("status") == "confirmed":
        checks.append(("✅", "Seasonality", "**This is seasonal.** The recent dip repeats the "
                       f"same calendar window a year earlier ({', '.join(seasonality['echo_months'])})."))
    elif seasonality.get("status") == "not_present":
        checks.append(("❌", "Seasonality", "Checked and ruled out — the same months a year "
                       "earlier were normal, so this is not a recurring seasonal dip."))
    elif seasonality.get("status") == "no_recent_dip":
        checks.append(("➖", "Seasonality", "Not applicable — there is no recent dip to explain."))
    else:
        checks.append(("⚠️", "Seasonality", "**Could not be checked** — under a year of history "
                       "means there is no prior year to compare against."))

    episodes = pack.get("dip_episodes") or []
    recovered = [e for e in episodes if e["recovered"]]
    ongoing = [e for e in episodes if e["is_ongoing"]]
    if ongoing:
        checks.append(("❌", "Already recovered?", f"No — the dip that began "
                       f"{ongoing[0]['start_month']} is still running after "
                       f"{ongoing[0]['months']} months."))
    elif recovered:
        checks.append(("✅", "Already recovered?", "**Yes.** The dip "
                       f"{recovered[-1]['start_month']} → {recovered[-1]['end_month']} recovered "
                       f"from {recovered[-1]['recovered_from_month']} — a resolved incident, "
                       "not a live problem."))
    else:
        checks.append(("➖", "Already recovered?", "No dip episodes in this history."))

    quality = pack.get("data_quality") or {}
    gaps = (quality.get("gaps") or {}).get("months_with_no_orders") or []
    if gaps:
        checks.append(("⚠️", "Missing data", f"**{len(gaps)} month(s) with no orders** "
                       f"({', '.join(gaps)}). A gap drags trailing averages down on its own — "
                       "some of any apparent decline is this."))
    else:
        checks.append(("➖", "Missing data", "No gaps — every month in range has orders."))

    outliers = quality.get("outlier_months") or []
    if outliers:
        checks.append(("⚠️", "One-off bulk order", f"**{outliers[0]['month']} was "
                       f"{outliers[0]['multiple_of_median_month']}× a normal month.** Months after "
                       "it are a return to normal, not a decline."))
    else:
        checks.append(("➖", "One-off bulk order", "No unusually large months."))

    returns = quality.get("returns") or {}
    if returns.get("return_lines"):
        checks.append(("➖", "Returns / credits", f"{returns['return_lines']} credit line(s) worth "
                       f"{_money(returns['net_return_value'])}, already netted into every figure. "
                       "Not leakage."))
    else:
        checks.append(("➖", "Returns / credits", "None in this history."))

    tier = pack.get("tier_mix") or {}
    if tier.get("status") == "premiumisation_detected":
        checks.append(("✅", "Mix shift direction", "**Favourable.** Value is moving INTO premium "
                       "lines, not out of them. A mix change is not automatically bad."))
    elif tier.get("status") == "downgrade_detected":
        checks.append(("❌", "Mix shift direction", "Unfavourable — value is moving out of premium "
                       "lines into cheaper ones."))

    if pack.get("account_manager_changed"):
        checks.append(("⚠️", "Account manager change", "The rep changed mid-history "
                       f"({', '.join(pack['account_manager'])}). Correlated with everything, "
                       "proof of nothing — context, not cause."))

    for icon, title, detail in checks:
        st.markdown(f"{icon} **{title}** — {detail}")


# How each row bears on the verdict. "Checked" is the one that needs saying
# out loud: those are innocent explanations that were tested and did not
# account for what was found — absence of a finding, deliberately shown.
_READS_AS_ICONS = {"concern": "🔴", "reassuring": "🟢", "checked": "🔍", "context": "⚪"}


def _timeline_tab(report: dict) -> None:
    """The dated evidence log — the same rows the PDF prints.

    The rest of this panel is organised by DIMENSION, which is how the
    analysis is computed but not how a decline is lived through. Re-keying
    the same facts by date is what lets a reader see that the discount
    stepped up two months before the margin gave way, rather than reading
    two separate findings and having to line them up themselves.
    """
    timeline = report.get("evidence_timeline") or []
    if not timeline:
        st.caption(
            "No dated timeline on this report — it was produced before the evidence "
            "log existed. Re-run the investigation to build one."
        )
        return

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
    st.caption(
        "Every row is a fact computed by the deterministic stages, dated to the month it "
        "was observed. \"Checked\" rows are harmless explanations that were tested and did "
        "not account for what was found."
    )


def _actions_tab(report: dict) -> None:
    actions = report.get("recommended_actions") or []
    if actions:
        st.markdown("**Recommended next steps**")
        for index, action in enumerate(actions, start=1):
            st.markdown(f"{index}. {action}")
    elif report["verdict"] == "healthy":
        st.success("No action needed — this account is behaving normally.")
    else:
        st.info("The agent proposed no specific actions.")

    if report.get("defer") and report.get("data_needed_if_deferring"):
        st.markdown("**What would settle this**")
        st.caption("The agent declined to call it. These are the specific gaps to fill.")
        for item in report["data_needed_if_deferring"]:
            st.markdown(f"- {item}")

    projection = (report.get("prioritization") or {}).get("churn_risk_projection")
    if projection:
        st.divider()
        st.markdown("**The cost of doing nothing**")
        st.markdown(
            f"At today's rate, another 12 months costs roughly "
            f"**{_money(max(projection.get('projected_12_month_loss_if_unaddressed') or 0, projection.get('projected_12_month_margin_loss_if_unaddressed') or 0))}**."
        )
        st.caption(projection["basis"])


OPINION_PRESENTATION = {
    "FLAG": ("critical", "Leakage"),
    "NO_FLAG": ("good", "Healthy"),
    "DEFER": ("warning", "Not enough evidence"),
}


def _opinion_line(report: dict, colors: dict) -> None:
    """One line under the banner: where the statistical model leaned, and
    whether the AI landed in the same place. The full account is in the tab."""
    opinion = report.get("model_opinion") or {}
    agreement = report.get("model_agreement")
    if not opinion.get("available") or not agreement:
        return
    role, word = OPINION_PRESENTATION.get(opinion["leaning_outcome"], ("warning", opinion["leaning"]))
    share = max(opinion["p_flag"], opinion["p_no_flag"], opinion["p_defer"])
    if agreement["agrees"]:
        verdict_text, verdict_role = "the AI reached the same call", "good"
    else:
        verdict_text, verdict_role = "the AI disagreed and says why below", "serious"
    st.markdown(
        f"<div style='font-size:0.9rem;color:{colors['text_secondary']};margin:-0.2rem 0 0.8rem'>"
        f"Statistical second opinion: <strong style='color:{colors[role]}'>{word}</strong> "
        f"({'>99' if share > 0.995 else f'{share * 100:.0f}'}%{'' if opinion.get('decisive') else ', not decisive'}) · "
        f"<span style='color:{colors[verdict_role]}'>{verdict_text}</span></div>",
        unsafe_allow_html=True,
    )


def _opinion_tab(report: dict, colors: dict) -> None:
    """The classifier's read of the same numbers, and what the AI made of it."""
    opinion = report.get("model_opinion") or {}
    agreement = report.get("model_agreement")

    if not opinion.get("available"):
        st.info(
            "No statistical second opinion for this run. "
            + (opinion.get("note") or "The verdict rests on the AI's reading of the evidence alone.")
        )
        return

    st.markdown("**What the statistical model saw**")
    st.caption(
        "A classifier trained on thousands of generated accounts whose problems were known in "
        "advance. It reads the same figures as the AI, reduced to numbers, and has never seen "
        "this account. It is a second witness, not the judge — the AI must explain any disagreement."
    )

    rows = pd.DataFrame({
        "outcome": ["Leakage", "Healthy", "Not enough evidence"],
        "probability": [opinion["p_flag"], opinion["p_no_flag"], opinion["p_defer"]],
        "role": ["critical", "good", "warning"],
    })
    rows["colour"] = rows["role"].map(colors)
    rows["label"] = rows["probability"].map(lambda p: f"{p * 100:.0f}%")
    base = alt.Chart(rows).encode(
        y=alt.Y("outcome:N", sort=["Leakage", "Healthy", "Not enough evidence"], title=None,
                axis=alt.Axis(labelColor=colors["text_primary"], labelFontSize=12, ticks=False, domain=False)),
    )
    bars = base.mark_bar(cornerRadiusEnd=3, size=22).encode(
        x=alt.X("probability:Q", scale=alt.Scale(domain=[0, 1]), axis=None),
        color=alt.Color("colour:N", scale=None),
        tooltip=[alt.Tooltip("outcome:N", title="Outcome"), alt.Tooltip("label:N", title="Probability")],
    )
    text = base.mark_text(align="left", dx=6, fontSize=12, color=colors["text_secondary"]).encode(
        x=alt.X("probability:Q"), text="label:N",
    )
    st.altair_chart(
        (bars + text).properties(height=110, background=colors["surface"], padding={"left": 0, "right": 40})
        .configure_view(strokeWidth=0),
        width="stretch",
    )

    if report.get("_opinion_after_the_fact"):
        st.caption(
            "This verdict was saved before the statistical model existed, so the AI did not see "
            "this opinion when it decided. Agreement below is computed after the fact."
        )

    if agreement:
        model_role, model_word = OPINION_PRESENTATION.get(agreement["model_outcome"], ("warning", "—"))
        agent_role, agent_word = OPINION_PRESENTATION.get(agreement["agent_outcome"], ("warning", "—"))
        if agreement["agrees"]:
            st.success(
                f"**Agree.** The model leaned **{model_word.lower()}** and the AI reached the same call"
                + (" — and the model was decisive, which corroborates the AI's confidence."
                   if agreement["model_decisive"] else
                   ", though the model itself was not decisive: the numbers alone are ambiguous here.")
            )
        else:
            st.warning(
                f"**Disagree.** The model leaned **{model_word.lower()}**; the AI said **{agent_word.lower()}**. "
                "The AI's reasons are below. A disagreement is worth reading, not a fault in either: "
                "the model cannot see context, and the AI can be swayed by a story."
            )
        if agreement.get("agent_response"):
            st.markdown(f"> {agreement['agent_response']}")
        elif not agreement["agrees"]:
            st.caption("The AI did not address the disagreement in its answer.")

    drivers = opinion.get("top_drivers") or []
    if drivers:
        st.markdown(f"**What the model's read rests on** — leaning *{opinion['leaning']}*")
        st.caption(
            "Each row is a fact the model had. The number is how much its leading probability "
            "would fall if that fact were hidden from it — measured, not estimated."
        )
        table = [{
            "Fact": d["label"],
            "Value": _driver_value(d),
            "Effect": f"{d['effect_on_leaning'] * 100:+.0f} pts",
            "Reads as": d["direction"],
        } for d in drivers]
        st.dataframe(table, width="stretch", hide_index=True)
        if opinion.get("drivers_note"):
            st.caption(opinion["drivers_note"])

    from ml.predict import model_provenance
    provenance = model_provenance()
    bits = [b for b in (
        f"trained on {provenance['trained_on']}" if provenance.get("trained_on") else None,
        (f"{provenance['held_out_accuracy'] * 100:.0f}% on generated accounts it never saw"
         if provenance.get("held_out_accuracy") is not None else None),
        provenance.get("reference_check"),
    ) if b]
    if bits:
        st.caption("Model provenance: " + " · ".join(bits) + ".")


def _driver_value(driver: dict) -> str:
    feature, value = driver["feature"], driver["value"]
    if feature.startswith("p_"):
        return f"p = {value:.3f}"
    if feature.startswith("sig_") or feature.startswith("has_") or feature in (
            "rev_cp_detected", "season_confirmed", "dip_ongoing", "dip_recovered", "manager_changed", "rev_cp_recovered"):
        return "yes" if value >= 0.5 else "no"
    if feature.endswith("_pp") or feature.endswith("_pct"):
        return f"{value:+.1f}" if feature.endswith("_pp") else f"{value:.1f}%"
    if feature.endswith("_pct_change") or feature in ("rev_h1_h2_pct", "gap_share", "defected_baseline_share",
                                                      "high_tier_base", "high_tier_recent", "max_cat_pct_decline"):
        return f"{value * 100:+.0f}%" if "change" in feature or feature == "rev_h1_h2_pct" else f"{value * 100:.0f}%"
    if feature == "rev_slope_pct_mo":
        return f"{value:+.2f}%/mo"
    if float(value).is_integer():
        return f"{int(value)}"
    return f"{value:.2f}"


def _confidence_tab(report: dict, pack: dict | None) -> None:
    sufficiency = report.get("data_sufficiency") or {}
    label = sufficiency.get("label", "unknown")

    message = (
        f"**Data sufficiency: {label}** — {sufficiency.get('history_months')} months of history, "
        f"{sufficiency.get('order_count')} orders, {sufficiency.get('category_count')} categories."
    )
    (st.error if label == "insufficient" else st.success if label == "sufficient" else st.warning)(message)
    for flag in sufficiency.get("flags", []):
        st.caption(f"• {flag.replace('_', ' ')}")

    st.markdown(f"**Confidence: {report['confidence']}**")
    if report["confidence"] == "low":
        st.caption(
            "A low-confidence answer is a real answer. Treat it as a prompt to look, "
            "not as a conclusion to act on."
        )

    # "returns" is a presence flag, not a capability flag — an account with no
    # credit notes was not a gap in coverage. Excluding it is also what makes
    # the "all six" line below true (see timeline.PRESENCE_ONLY_DIMENSIONS).
    unavailable = [
        n for n, ok in (report.get("analysis_dimensions") or {}).items()
        if not ok and n not in PRESENCE_ONLY_DIMENSIONS
    ]
    if unavailable:
        st.warning(
            "**Not analysed — these columns were absent from the file:** "
            + ", ".join(n.replace("_", " ") for n in unavailable)
            + ". Not measured is not the same as measured and fine."
        )
    else:
        st.success("All six detection dimensions were available in this file.")

    priority = report.get("prioritization") or {}
    if priority.get("reason"):
        st.caption(f"Priority reasoning: {priority['reason']}")


def render_verdict(report: dict, pack: dict | None = None, cached: bool = False) -> None:
    """Render one investigation result. `pack` enables the ruled-out checks and
    the supporting-numbers table; without it those degrade rather than break."""
    colors = active()

    # A verdict saved before the classifier existed carries no opinion of its
    # own. The pack's current opinion is still worth showing beside it, as
    # long as it is said plainly that the AI never saw it when it decided.
    if report.get("model_opinion") is None and (pack or {}).get("model_opinion", {}).get("available"):
        from pipeline.report import model_agreement
        report = {**report, "model_opinion": pack["model_opinion"],
                  "model_agreement": model_agreement(pack, report), "_opinion_after_the_fact": True}

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
    _opinion_line(report, colors)
    _money_row(report)

    st.markdown("**What's happening**")
    st.markdown(report["narrative"])

    evidence, when, ruled_out, opinion, actions, confidence = st.tabs(
        ["Evidence", "What changed, and when", "Ruled out", "Second opinion",
         "What to do", "Confidence & limits"]
    )
    with evidence:
        _evidence_tab(report, pack, colors)
    with when:
        _timeline_tab(report)
    with ruled_out:
        _ruled_out_tab(pack, report)
    with opinion:
        _opinion_tab(report, colors)
    with actions:
        _actions_tab(report)
    with confidence:
        _confidence_tab(report, pack)

