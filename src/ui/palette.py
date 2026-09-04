"""
Chart palette.

Every value here was checked with the data-viz validator against the actual
Streamlit light surface (#ffffff) rather than picked by eye:

  tier ramp    #184f95 / #2a78d6 / #86b6ef   ordinal, one hue, monotone L
  series pair  #2a78d6 / #eb6834             CVD dE 24.7, normal dE 33.6

Two rules drive the choices:

- Value tiers are an ORDERED scale (High > Mid > Low), so they get a single-hue
  ordinal ramp, not three categorical hues. Three unrelated hues would say the
  tiers are three different things rather than three rungs of one ladder.
- The high tier is the darkest step: "more ink" means "more value".

Status colors are reserved for status and never reused as a series color, and
every status is rendered with an icon and a word so the meaning never rests on
hue alone.
"""

from __future__ import annotations

COLORS = {
    "surface": "#ffffff",
    "series_1": "#2a78d6",
    "series_2": "#eb6834",
    "tier_High": "#184f95",
    "tier_Mid": "#2a78d6",
    "tier_Low": "#86b6ef",
    "text_primary": "#0b0b0b",
    "text_secondary": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
    "good": "#0ca30c",
    "warning": "#fab219",
    "serious": "#ec835a",
    "critical": "#d03b3b",
    "band": "#f0efec",
}

TIER_ORDER = ["High", "Mid", "Low"]


def active() -> dict:
    """The one palette. The app is pinned to Streamlit's light theme in
    .streamlit/config.toml, so charts never have to guess which surface they
    are being drawn on — and a validated palette stays validated."""
    return COLORS


def tier_range(colors: dict) -> list[str]:
    return [colors[f"tier_{tier}"] for tier in TIER_ORDER]


# Detector status -> (status role, icon, plain-language label). The icon and
# label are not decoration: on a light surface `warning` and `serious` sit
# below 3:1 contrast by design, so the color alone must never be the message.
STATUS_MEANING = {
    # revenue
    "material_decline": ("critical", "▼", "Material decline"),
    "mild_drift": ("warning", "▽", "Mild drift"),
    "growth": ("good", "▲", "Growing"),
    # margin
    "erosion_detected": ("critical", "▼", "Margin eroding"),
    "improvement_detected": ("good", "▲", "Margin improving"),
    # discount
    "creep_detected": ("critical", "▲", "Discount creeping"),
    "discipline_improved": ("good", "▼", "Discount tightened"),
    # tier mix
    "downgrade_detected": ("critical", "▼", "Downgrading"),
    "premiumisation_detected": ("good", "▲", "Trading up"),
    # order pattern
    "fragmentation_detected": ("serious", "◆", "Fragmenting"),
    "baskets_shrinking": ("warning", "▽", "Baskets shrinking"),
    # shared
    "stable": ("good", "●", "Stable"),
    "insufficient_history": ("warning", "?", "Not enough history"),
    "unavailable": ("warning", "—", "Not measured"),
}


def status_style(status: str | None, colors: dict) -> tuple[str, str, str]:
    """(hex, icon, label) for a detector status."""
    role, icon, label = STATUS_MEANING.get(status or "unavailable", ("warning", "—", str(status)))
    return colors[role], icon, label
