"""
Altair chart builders for the account dashboard.

Two rules shape everything here:

- **No dual axes, ever.** Revenue and margin rate are different scales, so
  they are two stacked charts sharing an x-axis, never one plot with two
  y-scales. A dual axis invents a correlation by choosing where the two
  scales line up, and the whole point of this dashboard is that revenue and
  margin move independently.
- **Every chart has a hover layer.** A crosshair rule plus a tooltip on the
  time series, per-mark tooltips on the bars. The table view underneath
  carries the same numbers, so nothing is reachable only by hovering.

Marks are thin, gridlines are hairlines one shade off the surface, and the
recent window is washed with a neutral band so "baseline vs recent" is
readable without a second color.
"""

from __future__ import annotations

import altair as alt
import pandas as pd

from .palette import TIER_ORDER, tier_range

CHART_HEIGHT = 240
SHORT_HEIGHT = 190


def _base(frame: pd.DataFrame, colors: dict, height: int) -> alt.Chart:
    return alt.Chart(frame).properties(height=height)


def _axis(colors: dict, title: str | None = None, fmt: str | None = None, **kwargs) -> alt.Axis:
    # `format` must be omitted entirely rather than passed as None — Altair 6
    # validates it as a string and rejects an explicit null.
    if fmt is not None:
        kwargs["format"] = fmt
    return alt.Axis(
        title=title,
        labelColor=colors["muted"],
        titleColor=colors["text_secondary"],
        gridColor=colors["grid"],
        domainColor=colors["axis"],
        tickColor=colors["axis"],
        labelFontSize=11,
        titleFontSize=11,
        **kwargs,
    )


def _recent_band(frame: pd.DataFrame, colors: dict) -> alt.Chart | None:
    """Neutral wash over the trailing window every comparison is made against.

    A gray band rather than a second series color: the window is context, not
    another thing being measured.
    """
    recent = frame[frame["window"] == "Recent"]
    if recent.empty:
        return None
    band = pd.DataFrame({"start": [recent["month"].min()], "end": [frame["month"].max()]})
    return (
        alt.Chart(band)
        .mark_rect(fill=colors["band"], opacity=0.55)
        .encode(x=alt.X("start:T", title=None), x2="end:T")
    )


def _hover_layer(frame: pd.DataFrame, colors: dict, value_field: str,
                 tooltips: list[alt.Tooltip], height: int) -> alt.Chart:
    """Crosshair rule + invisible wide hit area, so the target is the whole
    column rather than a pinpoint on the line."""
    hover = alt.selection_point(
        fields=["month"], nearest=True, on="pointerover", empty=False, clear="pointerout"
    )
    return (
        alt.Chart(frame)
        .mark_rule(color=colors["muted"], strokeWidth=1)
        .encode(
            x=alt.X("month:T", title=None),
            opacity=alt.condition(hover, alt.value(0.45), alt.value(0)),
            tooltip=tooltips,
        )
        .add_params(hover)
        .properties(height=height)
    )


def revenue_chart(frame: pd.DataFrame, colors: dict, change_point: str | None = None,
                  gap_months: list[str] | None = None,
                  outlier_months: list[str] | None = None) -> alt.Chart:
    """Monthly revenue, with the events that explain odd-looking months marked
    in place — a skipped month and a bulk month both distort a trend line, and
    a manager should see why before reading a decline into either."""
    tooltips = [
        alt.Tooltip("month:T", title="Month", format="%b %Y"),
        alt.Tooltip("revenue:Q", title="Revenue", format=",.0f"),
        alt.Tooltip("orders:Q", title="Orders"),
        alt.Tooltip("window:N", title="Window"),
    ]

    layers = []
    band = _recent_band(frame, colors)
    if band is not None:
        layers.append(band)

    line = (
        _base(frame, colors, CHART_HEIGHT)
        .mark_line(color=colors["series_1"], strokeWidth=2, point=False)
        .encode(
            x=alt.X("month:T", axis=_axis(colors)),
            y=alt.Y("revenue:Q", axis=_axis(colors, "Revenue (₹)", "~s")),
        )
    )
    layers.append(line)

    points = (
        _base(frame, colors, CHART_HEIGHT)
        .mark_point(color=colors["series_1"], size=45, filled=True)
        .encode(x=alt.X("month:T"), y=alt.Y("revenue:Q"), tooltip=tooltips)
    )
    layers.append(points)

    if outlier_months:
        marks = frame[frame["month"].dt.strftime("%Y-%m").isin(outlier_months)]
        if not marks.empty:
            layers.append(
                alt.Chart(marks).mark_point(
                    shape="triangle-up", size=110, filled=True,
                    color=colors["serious"], stroke=colors["surface"], strokeWidth=2,
                ).encode(
                    x="month:T", y="revenue:Q",
                    tooltip=[alt.Tooltip("month:T", title="Bulk month", format="%b %Y"),
                             alt.Tooltip("revenue:Q", title="Revenue", format=",.0f")],
                )
            )

    if gap_months:
        marks = frame[frame["month"].dt.strftime("%Y-%m").isin(gap_months)]
        if not marks.empty:
            layers.append(
                alt.Chart(marks).mark_point(
                    shape="cross", size=110, filled=True,
                    color=colors["warning"], stroke=colors["surface"], strokeWidth=2,
                ).encode(
                    x="month:T", y="revenue:Q",
                    tooltip=[alt.Tooltip("month:T", title="No orders this month", format="%b %Y")],
                )
            )

    if change_point:
        marker = pd.DataFrame({"month": [pd.Period(change_point, freq="M").to_timestamp()]})
        layers.append(
            alt.Chart(marker).mark_rule(color=colors["critical"], strokeWidth=2)
            .encode(x="month:T", tooltip=[alt.Tooltip("month:T", title="Change-point", format="%b %Y")])
        )

    layers.append(_hover_layer(frame, colors, "revenue", tooltips, CHART_HEIGHT))
    return alt.layer(*layers).resolve_scale(x="shared")


def rate_chart(frame: pd.DataFrame, colors: dict, field: str, title: str,
               color_key: str = "series_1") -> alt.Chart | None:
    """A percentage-rate series (margin, discount, high-tier share) on its own
    axis. Separate from the revenue chart on purpose — see the module docstring.
    """
    if field not in frame.columns or frame[field].isna().all():
        return None

    plot = frame[frame[field].notna()].copy()
    plot["display"] = plot[field] * 100

    tooltips = [
        alt.Tooltip("month:T", title="Month", format="%b %Y"),
        alt.Tooltip("display:Q", title=title, format=".1f"),
    ]

    layers = []
    band = _recent_band(frame, colors)
    if band is not None:
        layers.append(band)

    layers.append(
        _base(plot, colors, SHORT_HEIGHT)
        .mark_line(color=colors[color_key], strokeWidth=2)
        .encode(
            x=alt.X("month:T", axis=_axis(colors)),
            y=alt.Y("display:Q", axis=_axis(colors, f"{title} (%)"),
                    scale=alt.Scale(zero=False, nice=True)),
        )
    )
    layers.append(
        _base(plot, colors, SHORT_HEIGHT)
        .mark_point(color=colors[color_key], size=45, filled=True)
        .encode(x="month:T", y="display:Q", tooltip=tooltips)
    )
    layers.append(_hover_layer(plot, colors, "display", tooltips, SHORT_HEIGHT))
    return alt.layer(*layers).resolve_scale(x="shared")


def tier_mix_chart(long: pd.DataFrame, colors: dict, normalize: bool) -> alt.Chart | None:
    """Revenue by value tier over time.

    Shown as a share by default: the whole point is where the money sits, and
    at absolute scale a shift can hide behind a flat total — which is exactly
    the trap the flagship leak sets.
    """
    if long.empty:
        return None

    stack = "normalize" if normalize else "zero"
    axis_title = "Share of revenue (%)" if normalize else "Revenue (₹)"
    fmt = ".0%" if normalize else "~s"

    return (
        alt.Chart(long)
        .mark_area(
            line=False,
            # 2px surface gap between segments instead of a border around them.
            stroke=colors["surface"], strokeWidth=2, interpolate="monotone",
        )
        .encode(
            x=alt.X("month:T", axis=_axis(colors)),
            y=alt.Y("revenue:Q", stack=stack, axis=_axis(colors, axis_title, fmt)),
            color=alt.Color(
                "tier:N",
                sort=TIER_ORDER,
                scale=alt.Scale(domain=TIER_ORDER, range=tier_range(colors)),
                legend=alt.Legend(
                    title="Value tier", orient="top", direction="horizontal",
                    labelColor=colors["text_secondary"], titleColor=colors["text_secondary"],
                ),
            ),
            order=alt.Order("color_tier_sort_index:Q"),
            tooltip=[
                alt.Tooltip("month:T", title="Month", format="%b %Y"),
                alt.Tooltip("tier:N", title="Tier"),
                alt.Tooltip("revenue:Q", title="Revenue", format=",.0f"),
            ],
        )
        .properties(height=CHART_HEIGHT)
    )


def category_chart(comparison: pd.DataFrame, colors: dict) -> alt.Chart | None:
    """Average monthly revenue per category, baseline vs recent.

    Horizontal so long category names stay readable, and paired rather than
    stacked so a line that went to zero shows as an empty recent bar rather
    than a slightly shorter total.
    """
    if comparison.empty:
        return None

    height = max(200, 34 * comparison["category"].nunique())
    return (
        alt.Chart(comparison)
        .mark_bar(cornerRadiusEnd=4, height=11)
        .encode(
            y=alt.Y("category:N", sort=None, axis=_axis(colors, None), title=None),
            x=alt.X("revenue:Q", axis=_axis(colors, "Avg revenue per month (₹)", "~s")),
            yOffset=alt.YOffset("window:N", sort=["Baseline", "Recent"]),
            color=alt.Color(
                "window:N",
                scale=alt.Scale(domain=["Baseline", "Recent"],
                                range=[colors["series_1"], colors["series_2"]]),
                legend=alt.Legend(
                    title=None, orient="top", direction="horizontal",
                    labelColor=colors["text_secondary"],
                ),
            ),
            tooltip=[
                alt.Tooltip("category:N", title="Category"),
                alt.Tooltip("window:N", title="Window"),
                alt.Tooltip("revenue:Q", title="Avg / month", format=",.0f"),
            ],
        )
        .properties(height=height)
    )


def order_chart(frame: pd.DataFrame, colors: dict, field: str, title: str) -> alt.Chart:
    """Orders per month, or lines per order. Two different scales, so two
    charts rather than one plot with two axes."""
    tooltips = [
        alt.Tooltip("month:T", title="Month", format="%b %Y"),
        alt.Tooltip(f"{field}:Q", title=title, format=",.2f"),
    ]
    plot = frame[frame[field].notna()]

    layers = []
    band = _recent_band(frame, colors)
    if band is not None:
        layers.append(band)
    layers.append(
        alt.Chart(plot)
        .mark_bar(color=colors["series_1"], cornerRadiusEnd=4, size=9)
        .encode(
            x=alt.X("month:T", axis=_axis(colors)),
            y=alt.Y(f"{field}:Q", axis=_axis(colors, title)),
            tooltip=tooltips,
        )
        .properties(height=SHORT_HEIGHT)
    )
    return alt.layer(*layers).resolve_scale(x="shared")
