"""
Visual chrome for the Streamlit demo.

Mirrors youkti-app suite chrome (AppTab, AppStepper, Button, Card tokens)
because those React components cannot be imported here. Chart colours stay
in palette.py — they were validated against a white surface, and every chart
still draws on that surface. Copy is unchanged.

Motion is gated on Streamlit session state so a widget rerun does not replay
entrance animations. Hover and focus transitions stay on; they are what make
the page feel smooth after the first paint. `prefers-reduced-motion` turns
the rest off.
"""

from __future__ import annotations

import base64
from html import escape
from pathlib import Path

import streamlit as st

from .palette import COLORS

_YOUKTI_LOGO = Path(__file__).resolve().parents[2] / "assets" / "youkti_logo.svg"
_YOUKTI_LOGO_URI: str | None = None


def _youkti_logo_uri() -> str | None:
    """data: URI for the Youkti wordmark — cached after first read."""
    global _YOUKTI_LOGO_URI
    if _YOUKTI_LOGO_URI is not None:
        return _YOUKTI_LOGO_URI or None
    if not _YOUKTI_LOGO.is_file():
        _YOUKTI_LOGO_URI = ""
        return None
    raw = base64.b64encode(_YOUKTI_LOGO.read_bytes()).decode("ascii")
    _YOUKTI_LOGO_URI = f"data:image/svg+xml;base64,{raw}"
    return _YOUKTI_LOGO_URI

def inject() -> None:
    """Apply the theme. Safe to call on every rerun — Streamlit requires it.

    Injected via ``st.html`` so the stylesheet is not scoped to a markdown
    block — markdown-scoped CSS cannot widen ``stMainBlockContainer``.
    """
    first_paint = not st.session_state.get("_rl_theme_settled")
    st.session_state["_rl_theme_settled"] = True
    extra = "" if first_paint else (
        "<style>.rl-rise{animation:none!important;opacity:1;transform:none}</style>"
    )
    st.html(_CSS + extra, width="stretch")


def header(title: str, kicker: str = "") -> None:
    kicker_html = f'<div class="rl-kicker">{escape(kicker)}</div>' if kicker else ""
    st.markdown(
        f"""<header class="rl-masthead rl-rise">
          <div>
            {kicker_html}
            <p class="rl-title">{escape(title)}</p>
          </div>
        </header>""",
        unsafe_allow_html=True,
    )


def sidebar_brand(title: str, kicker: str = "", *, on_home=None) -> None:
    # A <p>, not <header>/<h1> — Streamlit's markdown sanitizer strips those
    # and dumps the leftovers as visible source. Logo sits above the product
    # name, matching youkti-app's header wordmark. When `on_home` is set the
    # logo is a button that returns to the account book.
    logo = _youkti_logo_uri()
    kicker_html = f'<p class="rl-kicker">{escape(kicker)}</p>' if kicker else ""
    if on_home is not None and logo:
        st.markdown(
            f'<style>'
            f'[class*="st-key-sidebar-youkti-home"] button {{'
            f'background: url("{logo}") left center / auto 1.75rem no-repeat !important;'
            f'background-color: transparent !important;'
            f'border: none !important; box-shadow: none !important;'
            f'color: transparent !important; min-height: 2.35rem !important;'
            f'width: 7.5rem !important; max-width: 7.5rem !important;'
            f'padding: 0 !important; justify-content: flex-start !important;'
            f'}}'
            f'[class*="st-key-sidebar-youkti-home"] button * {{'
            f'visibility: hidden !important;'
            f'}}'
            f'[class*="st-key-sidebar-youkti-home"] button:hover {{'
            f'background-color: transparent !important; opacity: 0.82 !important;'
            f'border: none !important; box-shadow: none !important;'
            f'}}'
            f'</style>',
            unsafe_allow_html=True,
        )
        st.button(
            "Youkti",
            key="sidebar-youkti-home",
            type="tertiary",
            on_click=on_home,
            help="Back to all accounts",
        )
        st.markdown(
            f'<div class="rl-side-brand">{kicker_html}'
            f'<p class="rl-side-title">{escape(title)}</p></div>',
            unsafe_allow_html=True,
        )
        return

    logo_html = (
        f'<img class="rl-side-logo" src="{logo}" alt="Youkti" />'
        if logo else ""
    )
    st.markdown(
        f'<div class="rl-side-brand">{logo_html}{kicker_html}'
        f'<p class="rl-side-title">{escape(title)}</p></div>',
        unsafe_allow_html=True,
    )


def page(title: str, caption: str, *, show_logo: bool = False) -> None:
    # Optional Youkti wordmark above the title — same asset as the sidebar
    # brand, sized a touch larger for the full-bleed book home.
    logo = _youkti_logo_uri() if show_logo else None
    logo_html = (
        f'<img class="rl-book-logo" src="{logo}" alt="Youkti" />'
        if logo else ""
    )
    # One line, no indentation: a blank or indented line inside the block
    # makes Markdown close the HTML and print the rest as a code block.
    st.markdown(
        f'<section class="rl-step rl-rise">{logo_html}'
        f'<p class="rl-step-title">{escape(title)}</p>'
        f'<p class="rl-step-caption">{escape(caption)}</p>'
        f'</section>',
        unsafe_allow_html=True,
    )


def identity(title: str, meta: str) -> None:
    st.markdown(
        f"""<div class="rl-identity rl-rise">
          <p class="rl-identity-title">{escape(title)}</p>
          <p class="rl-identity-meta">{meta}</p>
        </div>""",
        unsafe_allow_html=True,
    )


def status_strip(items: list[dict]) -> None:
    """items: label, color, icon, text, hint."""
    cells = []
    for i, item in enumerate(items):
        cells.append(
            f"""<div class="rl-signal rl-rise" style="--d:{i};--sig:{escape(item['color'], quote=True)}" title="{escape(item.get('hint') or '', quote=True)}">
              <div class="rl-signal-label">{escape(item['label'])}</div>
              <div class="rl-signal-value">
                <span class="rl-signal-icon">{escape(item['icon'])}</span>
                {escape(item['text'])}
              </div>
            </div>"""
        )
    st.markdown(f'<div class="rl-signal-row">{"".join(cells)}</div>', unsafe_allow_html=True)


def kpi_row(tiles: list[dict]) -> None:
    """tiles: label, value, delta, tone ('good'|'bad'|'neutral'), hint."""
    cards = []
    for i, tile in enumerate(tiles):
        tone = tile.get("tone") or "neutral"
        delta = tile.get("delta")
        delta_html = (
            f'<div class="rl-kpi-delta rl-kpi-delta-{escape(tone)}">{escape(delta)}</div>'
            if delta else ""
        )
        cards.append(
            f"""<div class="rl-kpi rl-rise" style="--d:{i}" title="{escape(tile.get('hint') or '', quote=True)}">
              <div class="rl-kpi-label">{escape(tile['label'])}</div>
              <div class="rl-kpi-value">{escape(tile['value'])}</div>
              {delta_html}
            </div>"""
        )
    st.markdown(f'<div class="rl-kpi-row">{"".join(cards)}</div>', unsafe_allow_html=True)


def banner(role: str, title: str, lines: list[str]) -> None:
    color = COLORS.get(role, COLORS["warning"])
    body = "".join(f'<div class="rl-banner-line">{line}</div>' for line in lines)
    st.markdown(
        f"""<div class="rl-banner rl-rise" style="--tone:{escape(color, quote=True)}">
          <div class="rl-banner-title">{escape(title)}</div>
          {body}
        </div>""",
        unsafe_allow_html=True,
    )


_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@24,400,0,0&display=swap');

:root {{
  --youkti-primary: #7650A1;
  --youkti-secondary: #7270B2;
  --youkti-tertiary: #81C1D3;
  --youkti-primary-hover: color-mix(in oklab, #7650A1 82%, black);
  --rl-ink: #252525;
  --rl-secondary: #52525b;
  --rl-muted: #6b7280;
  --rl-line: #e5e7eb;
  --rl-axis: {COLORS['axis']};
  --rl-band: {COLORS['band']};
  --rl-paper: #ffffff;
  --rl-card: #ffffff;
  --rl-navy: var(--youkti-primary);
  --rl-blue: var(--youkti-secondary);
  --rl-good: {COLORS['good']};
  --rl-warn: {COLORS['warning']};
  --rl-serious: {COLORS['serious']};
  --rl-crit: {COLORS['critical']};
  --rl-ease: cubic-bezier(0.22, 1, 0.36, 1);
  --rl-shadow: 0 1px 2px 0 rgb(0 0 0 / 0.05);
  --rl-shadow-hover: 0 1px 3px 0 rgb(0 0 0 / 0.08);
  --radius: 0.625rem;
}}

html, body, [data-testid="stAppViewContainer"], .stApp {{
  background: var(--rl-paper) !important;
  color: var(--rl-ink);
}}

h1, h2, h3, h4, p, label,
.rl-masthead, .rl-step, .rl-identity, .rl-signal, .rl-kpi, .rl-banner,
[data-testid="stMarkdownContainer"], [data-testid="stWidgetLabel"],
[data-testid="stCaptionContainer"] {{
  font-family: ui-sans-serif, system-ui, sans-serif;
}}

/* Ligature icons — do not inherit the UI font. */
[data-testid="stIconMaterial"],
[data-testid="stIconMaterial"] *,
[data-testid="stExpanderToggleIcon"],
[data-testid="stExpanderToggleIcon"] * {{
  font-family: "Material Symbols Rounded" !important;
  font-weight: 400 !important;
  font-style: normal !important;
  font-variation-settings: "FILL" 0, "wght" 400, "GRAD" 0, "opsz" 24;
  font-feature-settings: "liga" !important;
  letter-spacing: normal !important;
  line-height: 1 !important;
  text-transform: none !important;
  white-space: nowrap !important;
  speak: never;
}}

/* Hide Streamlit chrome — the page is the product. */
[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"],
#MainMenu, footer, .stDeployButton {{
  display: none !important;
}}

/* Streamlit pins a heading-anchor link next to every h1–h4. Hide it. */
[data-testid="stHeaderActionElements"],
[data-testid="stHeadingWithActionElements"] a,
[data-testid="stHeadingWithActionElements"] [data-testid="StyledLinkIconContainer"],
.stHeadingWithActionElements a,
.stMarkdown h1 a, .stMarkdown h2 a, .stMarkdown h3 a, .stMarkdown h4 a,
[data-testid="stMarkdownContainer"] a[href^="#"],
[data-testid="stHeading"] a[href^="#"] {{
  display: none !important;
}}

.stApp {{
  background: #ffffff !important;
}}

/* Streamlit 1.63 pins centered layout at 736px (sizes.contentMaxWidth).
   Wide mode only lifts that above ~864px, and markdown-scoped CSS cannot
   reach this node. Kill the cap on every copy of the main column. */
.stMain,
[data-testid="stMain"],
.stMainBlockContainer,
.stMainBlockContainer.block-container,
[data-testid="stMainBlockContainer"],
.block-container,
.e15ve43o4,
section.main,
section.main > div {{
  max-width: none !important;
  width: 100% !important;
}}

.stMainBlockContainer,
[data-testid="stMainBlockContainer"],
.block-container {{
  padding-top: 1.5rem !important;
  padding-bottom: 4.5rem !important;
  padding-left: 2.25rem !important;
  padding-right: 2.25rem !important;
}}

@media (min-width: 1600px) {{
  .stMainBlockContainer,
  [data-testid="stMainBlockContainer"],
  .block-container {{
    padding-left: 2.75rem !important;
    padding-right: 2.75rem !important;
  }}
}}

[data-testid="stHtml"] {{
  height: 0 !important;
  min-height: 0 !important;
  margin: 0 !important;
  padding: 0 !important;
  overflow: hidden !important;
}}

/* Home sidebar — pin to the top. Streamlit reserves header height and
   vertically centers short nav; both leave the dead space above. */
[data-testid="stSidebar"] {{
  background: #fff !important;
  border-right: 1px solid var(--rl-line) !important;
  top: 0 !important;
  height: 100vh !important;
  padding-top: 0 !important;
  margin-top: 0 !important;
}}
/* Account book home — full-bleed table, no left rail until a row is opened. */
.stApp:has(.rl-book-home) [data-testid="stSidebar"],
[data-testid="stApp"]:has(.rl-book-home) [data-testid="stSidebar"],
.stApp:has(.rl-book-home) [data-testid="stSidebarCollapsedControl"],
[data-testid="stApp"]:has(.rl-book-home) [data-testid="stSidebarCollapsedControl"],
.stApp:has(.rl-book-home) [data-testid="collapsedControl"],
[data-testid="stApp"]:has(.rl-book-home) [data-testid="collapsedControl"] {{
  display: none !important;
  visibility: hidden !important;
  width: 0 !important;
  min-width: 0 !important;
  max-width: 0 !important;
  margin: 0 !important;
  padding: 0 !important;
  border: none !important;
  overflow: hidden !important;
  pointer-events: none !important;
}}
.stApp:has(.rl-book-home) [data-testid="stAppViewContainer"],
[data-testid="stApp"]:has(.rl-book-home) [data-testid="stAppViewContainer"],
.stApp:has(.rl-book-home) [data-testid="stMain"],
[data-testid="stApp"]:has(.rl-book-home) [data-testid="stMain"],
.stApp:has(.rl-book-home) .main,
[data-testid="stApp"]:has(.rl-book-home) .main {{
  margin-left: 0 !important;
  max-width: 100% !important;
  width: 100% !important;
}}
.rl-book-home {{
  display: none !important;
}}
[data-testid="stSidebar"] > div,
[data-testid="stSidebar"] [data-testid="stSidebarContent"] {{
  padding-top: 0 !important;
  margin-top: 0 !important;
  height: 100% !important;
  justify-content: flex-start !important;
  align-items: stretch !important;
  background: #fff !important;
}}
[data-testid="stSidebarHeader"] {{
  display: none !important;
  height: 0 !important;
  min-height: 0 !important;
  padding: 0 !important;
}}
[data-testid="stSidebarUserContent"] {{
  padding: 1rem 0.75rem 1.5rem !important;
  margin-top: 0 !important;
  justify-content: flex-start !important;
  align-items: stretch !important;
}}
[data-testid="stSidebarUserContent"] > div {{
  align-items: stretch !important;
}}
[data-testid="stSidebarCollapseButton"],
[data-testid="collapsedControl"] {{
  display: none !important;
}}

[data-testid="stDialog"],
[data-testid="stModal"] {{
  display: flex !important;
  align-items: center !important;
  justify-content: center !important;
}}
[data-testid="stDialog"] [role="dialog"],
[data-testid="stModal"] [role="dialog"] {{
  margin: auto !important;
}}

/* Entrance — skipped after the first paint via .rl-settled. */
@keyframes rl-rise {{
  from {{ opacity: 0; transform: translateY(12px); }}
  to   {{ opacity: 1; transform: translateY(0); }}
}}
.rl-rise {{
  animation: rl-rise 0.55s var(--rl-ease) both;
  animation-delay: calc(var(--d, 0) * 55ms);
}}

/* Masthead */
.rl-masthead {{
  display: flex;
  align-items: center;
  gap: 0.9rem;
  margin: 0 0 0.35rem;
}}
.rl-mark {{
  display: none !important;
}}
.rl-kicker {{
  font-size: 0.72rem;
  font-weight: 600;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--rl-muted);
  margin-bottom: 0.12rem;
}}
.rl-title {{
  font-size: 1.72rem;
  font-weight: 700;
  letter-spacing: -0.035em;
  line-height: 1.15;
  color: var(--rl-ink);
  margin: 0;
}}

.rl-side-brand {{
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  gap: 0.65rem;
  margin: 0.15rem 0 1.1rem;
}}
.rl-side-logo {{
  display: block;
  height: 1.75rem;
  width: auto;
  max-width: 7.5rem;
  object-fit: contain;
}}
.rl-book-logo {{
  display: block;
  height: 2.35rem;
  width: auto;
  max-width: 10rem;
  object-fit: contain;
}}
.rl-side-title {{
  font-size: 1.05rem;
  font-weight: 700;
  letter-spacing: -0.03em;
  line-height: 1.25;
  color: var(--rl-ink);
  text-align: left;
  margin: 0 0 0.15rem;
}}

/* Page heading — title then caption, no step number. */
.rl-step {{
  display: flex;
  flex-direction: column;
  gap: 0.35rem;
  margin: 0.15rem 0 0.85rem;
}}
/* Match sidebar brand spacing when the wordmark sits above the page title. */
.rl-step > .rl-side-logo,
.rl-step > .rl-book-logo {{
  margin-bottom: 0.3rem;
}}
.rl-step-title {{
  font-size: 1.18rem;
  font-weight: 700;
  letter-spacing: -0.025em;
  margin: 0;
  color: var(--rl-ink);
}}
.rl-step-caption {{
  margin: 0;
  color: var(--rl-secondary);
  font-size: 0.94rem;
  line-height: 1.55;
  max-width: 72ch;
}}

.rl-identity {{
  margin: 0.15rem 0 0.85rem;
}}
.rl-identity-title {{
  font-size: 1.28rem;
  font-weight: 700;
  letter-spacing: -0.03em;
  margin: 0 0 0.25rem;
}}
.rl-identity-meta {{
  margin: 0;
  color: var(--rl-secondary);
  font-size: 0.9rem;
}}
.rl-identity-meta strong {{
  color: var(--rl-ink);
  font-weight: 600;
}}

/* Status strip */
.rl-signal-row {{
  display: grid;
  grid-template-columns: repeat(6, minmax(0, 1fr));
  gap: 0.55rem;
  margin: 0.15rem 0 0.4rem;
}}
.rl-signal {{
  background: var(--rl-card);
  border: 1px solid var(--rl-line);
  border-radius: 12px;
  padding: 0.7rem 0.75rem 0.72rem;
  position: relative;
  overflow: hidden;
  transition: transform 0.28s var(--rl-ease), box-shadow 0.28s ease, border-color 0.28s ease;
  box-shadow: var(--rl-shadow);
}}
.rl-signal::before {{
  content: "";
  position: absolute;
  left: 0; top: 0; bottom: 0;
  width: 3px;
  background: var(--sig);
}}
.rl-signal:hover {{
  transform: translateY(-2px);
  box-shadow: var(--rl-shadow-hover);
  border-color: var(--rl-axis);
}}
.rl-signal-label {{
  font-size: 0.68rem;
  font-weight: 650;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--rl-muted);
  margin-bottom: 0.28rem;
}}
.rl-signal-value {{
  font-size: 0.92rem;
  font-weight: 650;
  color: var(--sig);
  letter-spacing: -0.015em;
  display: flex;
  align-items: center;
  gap: 0.28rem;
}}
.rl-signal-icon {{
  font-size: 0.78rem;
}}

/* KPI tiles */
.rl-kpi-row {{
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 0.7rem;
  margin: 0.15rem 0 0.55rem;
}}
.rl-kpi {{
  background: var(--rl-card);
  border: 1px solid var(--rl-line);
  border-radius: 14px;
  padding: 0.95rem 1.05rem 1rem;
  box-shadow: var(--rl-shadow);
  transition: transform 0.3s var(--rl-ease), box-shadow 0.3s ease, border-color 0.3s ease;
}}
.rl-kpi:hover {{
  transform: translateY(-3px);
  box-shadow: var(--rl-shadow-hover);
  border-color: var(--rl-axis);
}}
.rl-kpi-label {{
  font-size: 0.75rem;
  font-weight: 600;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  color: var(--rl-muted);
}}
.rl-kpi-value {{
  font-size: 1.55rem;
  font-weight: 700;
  letter-spacing: -0.04em;
  line-height: 1.2;
  margin: 0.28rem 0 0.18rem;
  color: var(--rl-ink);
}}
.rl-kpi-delta {{
  font-size: 0.8rem;
  font-weight: 600;
}}
.rl-kpi-delta-good {{ color: var(--rl-good); }}
.rl-kpi-delta-bad {{ color: var(--rl-crit); }}
.rl-kpi-delta-neutral {{ color: var(--rl-secondary); }}

/* Verdict / decision banners */
.rl-banner {{
  border-left: 4px solid var(--tone);
  background: linear-gradient(90deg, color-mix(in srgb, var(--tone) 10%, var(--rl-card)), var(--rl-card));
  border: 1px solid var(--rl-line);
  border-left-width: 4px;
  border-left-color: var(--tone);
  border-radius: 14px;
  padding: 1rem 1.15rem 1.05rem;
  margin: 0.2rem 0 0.95rem;
  box-shadow: var(--rl-shadow);
}}
.rl-banner-title {{
  font-size: 1.28rem;
  font-weight: 700;
  letter-spacing: -0.03em;
  color: var(--tone);
}}
.rl-banner-line {{
  font-size: 0.92rem;
  color: var(--rl-secondary);
  margin-top: 0.28rem;
}}
.rl-banner-line strong {{
  color: var(--rl-ink);
}}

.rl-chip-row {{
  margin: -0.15rem 0 0.85rem;
}}
.rl-chip-kicker {{
  font-size: 0.72rem;
  font-weight: 650;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--rl-muted);
  margin-bottom: 0.4rem;
}}
.rl-chip {{
  display: inline-block;
  padding: 0.2rem 0.65rem;
  margin: 0 0.35rem 0.35rem 0;
  border-radius: 999px;
  background: color-mix(in srgb, var(--rl-crit) 12%, var(--rl-card));
  color: var(--rl-crit);
  font-size: 0.82rem;
  font-weight: 650;
  letter-spacing: -0.01em;
}}

/* AppTab — vertical list, icon + label inline, left rule when active. */
[class*="st-key-dashview-"] {{
  margin-bottom: 0.15rem;
}}
[class*="st-key-dashview-"] button {{
  display: flex !important;
  flex-direction: row !important;
  align-items: center !important;
  justify-content: flex-start !important;
  gap: 0.5rem !important;
  min-height: auto !important;
  width: 100% !important;
  padding: 0.55rem 0.85rem !important;
  border: none !important;
  border-left: 2px solid transparent !important;
  border-radius: 0 0.375rem 0.375rem 0 !important;
  background: transparent !important;
  color: #6b7280 !important;
  font-size: 0.875rem !important;
  font-weight: 600 !important;
  letter-spacing: 0 !important;
  line-height: 1.25 !important;
  white-space: nowrap !important;
  box-shadow: none !important;
  transition: color 0.15s ease, border-color 0.15s ease, background 0.15s ease !important;
}}
[class*="st-key-dashview-"] button:hover {{
  transform: none !important;
  color: #1f2937 !important;
  background: color-mix(in oklab, var(--youkti-primary) 6%, white) !important;
  box-shadow: none !important;
}}
[class*="st-key-dashview-"] button[kind="primary"],
[class*="st-key-dashview-"] [data-testid="stBaseButton-primary"] {{
  background: color-mix(in oklab, var(--youkti-primary) 10%, white) !important;
  border-left-color: var(--youkti-primary) !important;
  color: var(--youkti-primary) !important;
}}
[class*="st-key-dashview-"] button [data-testid="stIconMaterial"] {{
  font-size: 1rem !important;
  width: 1rem !important;
  height: 1rem !important;
  color: inherit !important;
}}

/* Streamlit widgets */
hr, [data-testid="stDivider"] {{
  border-color: var(--rl-line) !important;
  margin: 1.35rem 0 !important;
}}

[data-testid="stVerticalBlockBorderWrapper"] {{
  background: var(--rl-card);
  border-color: var(--rl-line) !important;
  border-radius: 14px !important;
  box-shadow: var(--rl-shadow);
  transition: box-shadow 0.28s ease, transform 0.28s var(--rl-ease);
}}
[data-testid="stVerticalBlockBorderWrapper"]:hover {{
  box-shadow: var(--rl-shadow-hover);
}}

.stButton > button,
[data-testid="stBaseButton-secondary"],
[data-testid="stBaseButton-primary"] {{
  border-radius: 0.375rem !important;
  font-weight: 500 !important;
  font-size: 0.875rem !important;
  letter-spacing: 0 !important;
  min-height: 2.25rem !important;
  box-shadow: var(--rl-shadow) !important;
  transition: background 0.15s ease, border-color 0.15s ease, color 0.15s ease !important;
  transform: none !important;
}}
.stButton > button:hover,
[data-testid="stBaseButton-secondary"]:hover,
[data-testid="stBaseButton-primary"]:hover {{
  transform: none !important;
}}
[data-testid="stBaseButton-primary"],
.stButton > button[kind="primary"] {{
  background: var(--youkti-primary) !important;
  border-color: var(--youkti-primary) !important;
  color: #fff !important;
}}
[data-testid="stBaseButton-primary"]:hover,
.stButton > button[kind="primary"]:hover {{
  background: var(--youkti-primary-hover) !important;
  border-color: var(--youkti-primary-hover) !important;
}}

/* Beat the generic button box so every view tab stays a flat list item. */
[class*="st-key-dashview-"] .stButton > button,
[class*="st-key-dashview-"] [data-testid="stBaseButton-secondary"],
[class*="st-key-dashview-"] [data-testid="stBaseButton-primary"],
[class*="st-key-dashview-"] button {{
  min-height: auto !important;
  border: none !important;
  border-left: 2px solid transparent !important;
  border-radius: 0 0.375rem 0.375rem 0 !important;
  background: transparent !important;
  box-shadow: none !important;
  color: #6b7280 !important;
}}
[class*="st-key-dashview-"] button:hover,
[class*="st-key-dashview-"] [data-testid="stBaseButton-secondary"]:hover,
[class*="st-key-dashview-"] [data-testid="stBaseButton-primary"]:hover,
[class*="st-key-dashview-"] button[kind="primary"]:hover {{
  background: color-mix(in oklab, var(--youkti-primary) 8%, white) !important;
  border-color: transparent !important;
  border-left-color: var(--youkti-primary) !important;
  color: var(--youkti-primary) !important;
  box-shadow: none !important;
}}
[class*="st-key-dashview-"] button[kind="primary"],
[class*="st-key-dashview-"] [data-testid="stBaseButton-primary"] {{
  background: color-mix(in oklab, var(--youkti-primary) 10%, white) !important;
  border-left-color: var(--youkti-primary) !important;
  color: var(--youkti-primary) !important;
}}

[class*="st-key-navpage-"] {{
  margin-bottom: 0.2rem;
}}
[class*="st-key-navpage-"] .stButton > button,
[class*="st-key-navpage-"] [data-testid="stBaseButton-secondary"],
[class*="st-key-navpage-"] [data-testid="stBaseButton-primary"],
[class*="st-key-navpage-"] button {{
  display: flex !important;
  flex-direction: row !important;
  align-items: center !important;
  justify-content: flex-start !important;
  gap: 0.7rem !important;
  min-height: auto !important;
  width: 100% !important;
  padding: 0.55rem 0.55rem !important;
  border: none !important;
  border-radius: 0.5rem !important;
  background: transparent !important;
  color: #6b7280 !important;
  font-size: 0.95rem !important;
  font-weight: 650 !important;
  letter-spacing: 0 !important;
  line-height: 1.3 !important;
  white-space: nowrap !important;
  text-align: left !important;
  box-shadow: none !important;
}}
[class*="st-key-navpage-"] button::before {{
  flex: 0 0 1.5rem;
  width: 1.5rem;
  height: 1.5rem;
  border-radius: 999px;
  background: var(--youkti-primary);
  color: #fff;
  font-size: 11px;
  font-weight: 700;
  line-height: 1.5rem;
  text-align: center;
  box-shadow: var(--rl-shadow);
}}
[data-testid="stSidebar"] [class*="st-key-arya-new-chat"] button,
[data-testid="stSidebar"] [class*="st-key-arya-new-chat"] .stButton > button {{
  display: flex !important;
  justify-content: center !important;
  align-items: center !important;
  gap: 0.4rem !important;
  min-height: 2.25rem !important;
  margin: 0 0 0.65rem !important;
  padding: 0.45rem 0.75rem !important;
  border-radius: 0.5rem !important;
  background: var(--youkti-primary) !important;
  border: 1px solid var(--youkti-primary) !important;
  color: #fff !important;
  font-weight: 650 !important;
  box-shadow: none !important;
}}
[data-testid="stSidebar"] [class*="st-key-arya-new-chat"] button *,
[data-testid="stSidebar"] [class*="st-key-arya-new-chat"] button [data-testid="stIconMaterial"] {{
  display: inline-flex !important;
  color: #fff !important;
}}
[data-testid="stSidebar"] [class*="st-key-arya-new-chat"] button:hover {{
  background: var(--youkti-primary-hover) !important;
  border-color: var(--youkti-primary-hover) !important;
  color: #fff !important;
}}
[class*="st-key-navpage-data"] button::before {{ content: "1"; }}
[class*="st-key-navpage-verdict"] button::before {{ content: "2"; }}
[class*="st-key-navpage-cltv"] button::before {{ content: "3"; }}
[class*="st-key-navpage-score"] button::before {{ content: "3"; }}
[class*="st-key-navpage-compare"] button::before {{ content: "3"; }}
[class*="st-key-navpage-"] button [data-testid="stIconMaterial"] {{
  display: none !important;
}}
[class*="st-key-navpage-"] button p,
[class*="st-key-navpage-"] button span,
[class*="st-key-navpage-"] button > div {{
  justify-content: flex-start !important;
  text-align: left !important;
  margin-left: 0 !important;
  margin-right: auto !important;
}}
[class*="st-key-navpage-"] button:hover,
[class*="st-key-navpage-"] [data-testid="stBaseButton-secondary"]:hover {{
  transform: none !important;
  background: color-mix(in oklab, var(--youkti-primary) 8%, white) !important;
  background-color: color-mix(in oklab, var(--youkti-primary) 8%, white) !important;
  color: #1f2937 !important;
  box-shadow: none !important;
}}
[data-testid="stSidebar"] [class*="st-key-navpage-"] button[kind="primary"],
[data-testid="stSidebar"] [class*="st-key-navpage-"] button[kind="primary"]:hover,
[data-testid="stSidebar"] [class*="st-key-navpage-"] [data-testid="stBaseButton-primary"],
[data-testid="stSidebar"] [class*="st-key-navpage-"] [data-testid="stBaseButton-primary"]:hover,
[data-testid="stSidebar"] [class*="st-key-navpage-"] .stButton > button[kind="primary"],
[data-testid="stSidebar"] [class*="st-key-navpage-"] .stButton > button[kind="primary"]:hover {{
  background: color-mix(in oklab, var(--youkti-primary) 10%, white) !important;
  background-color: color-mix(in oklab, var(--youkti-primary) 10%, white) !important;
  border-color: transparent !important;
  color: var(--youkti-primary) !important;
  box-shadow: none !important;
  transform: none !important;
}}
[data-testid="stSidebar"] [class*="st-key-navpage-"] button[kind="primary"] *,
[data-testid="stSidebar"] [class*="st-key-navpage-"] [data-testid="stBaseButton-primary"] * {{
  color: var(--youkti-primary) !important;
}}
[data-testid="stSidebar"] [class*="st-key-navpage-"] button[kind="primary"]::before,
[data-testid="stSidebar"] [class*="st-key-navpage-"] [data-testid="stBaseButton-primary"]::before {{
  background: var(--youkti-primary);
  color: #fff;
}}
[class*="st-key-sidebar-source"] .stButton > button,
[class*="st-key-sidebar-source"] button {{
  justify-content: flex-start !important;
  min-height: auto !important;
  padding: 0.55rem 0.8rem !important;
  border: none !important;
  background: transparent !important;
  box-shadow: none !important;
  color: var(--rl-muted) !important;
  font-weight: 600 !important;
}}
[class*="st-key-sidebar-source"] button:hover {{
  background: color-mix(in oklab, var(--youkti-primary) 6%, white) !important;
  color: #1f2937 !important;
  box-shadow: none !important;
  transform: none !important;
}}

[class*="st-key-file-info"] .stButton > button,
[class*="st-key-file-info"] button {{
  background: color-mix(in oklab, var(--youkti-tertiary) 28%, white) !important;
  border: 1px solid color-mix(in oklab, var(--youkti-tertiary) 55%, white) !important;
  color: #1f4e5a !important;
  border-radius: 0.375rem !important;
  padding-left: 0.85rem !important;
  padding-right: 0.85rem !important;
  box-shadow: none !important;
}}
[class*="st-key-file-info"] button:hover {{
  background: color-mix(in oklab, var(--youkti-tertiary) 40%, white) !important;
  border-color: var(--youkti-tertiary) !important;
  color: #1f4e5a !important;
}}

/* One border on the field — not on the wrapper as well. */
[data-testid="stSelectbox"] > div,
[data-testid="stMultiSelect"] > div {{
  border: none !important;
  box-shadow: none !important;
  outline: none !important;
  background: transparent !important;
}}
[data-testid="stSelectbox"] [data-baseweb="select"] > div,
[data-testid="stMultiSelect"] [data-baseweb="select"] > div {{
  min-height: 2.5rem !important;
  border: 1px solid var(--rl-line) !important;
  border-radius: 0.375rem !important;
  background: #fff !important;
  box-shadow: none !important;
  outline: none !important;
  cursor: pointer !important;
  padding-left: 0.75rem !important;
  padding-right: 0.5rem !important;
  transition: border-color 0.15s ease, box-shadow 0.15s ease;
}}
[data-testid="stSelectbox"] *:focus,
[data-testid="stSelectbox"] *:focus-visible,
[data-testid="stMultiSelect"] *:focus,
[data-testid="stMultiSelect"] *:focus-visible {{
  outline: none !important;
}}
[data-testid="stSelectbox"]:hover [data-baseweb="select"] > div,
[data-testid="stMultiSelect"]:hover [data-baseweb="select"] > div {{
  border-color: var(--youkti-primary) !important;
}}
[data-testid="stSelectbox"]:focus-within [data-baseweb="select"] > div,
[data-testid="stMultiSelect"]:focus-within [data-baseweb="select"] > div {{
  border-color: var(--youkti-primary) !important;
  box-shadow: 0 0 0 3px color-mix(in oklab, var(--youkti-primary) 28%, transparent) !important;
}}

.stTabs [data-baseweb="tab-list"] {{
  gap: 0.5rem;
  border-bottom: 1px solid var(--rl-line);
  background: transparent;
}}
.stTabs [data-baseweb="tab"] {{
  padding: 0.5rem 1rem;
  font-weight: 600;
  font-size: 0.875rem;
  color: #6b7280;
  border-radius: 0;
  border-bottom: 2px solid transparent;
  background: transparent !important;
  transition: color 0.15s ease, border-color 0.15s ease;
}}
.stTabs [data-baseweb="tab"]:hover {{
  color: #1f2937;
  background: transparent !important;
}}
.stTabs [aria-selected="true"] {{
  color: var(--youkti-primary) !important;
  border-bottom-color: var(--youkti-primary) !important;
  box-shadow: none !important;
}}

[data-testid="stExpander"] {{
  background: var(--rl-card);
  border: 1px solid var(--rl-line) !important;
  border-radius: 12px !important;
  box-shadow: var(--rl-shadow);
  transition: box-shadow 0.25s ease;
}}
[data-testid="stExpander"]:hover {{
  box-shadow: var(--rl-shadow-hover);
}}

[data-testid="stAlert"] {{
  border-radius: 12px !important;
  border: 1px solid var(--rl-line) !important;
  box-shadow: var(--rl-shadow);
}}

@keyframes yk-chart-in {{
  from {{ opacity: 0; transform: translateY(8px); }}
  to   {{ opacity: 1; transform: translateY(0); }}
}}
@keyframes yk-reveal-x {{
  from {{ clip-path: inset(0 100% 0 0); }}
  to   {{ clip-path: inset(0 0 0 0); }}
}}
@keyframes yk-mark-in {{
  from {{ opacity: 0; transform: translateY(6px); }}
  to   {{ opacity: 1; transform: translateY(0); }}
}}

[data-testid="stVegaLiteChart"],
[data-testid="stArrowVegaLiteChart"] {{
  background: var(--rl-card);
  border: 1px solid var(--rl-line);
  border-radius: 14px;
  padding: 0.7rem 0.55rem 0.15rem;
  box-shadow: var(--rl-shadow);
  animation: yk-chart-in 0.22s var(--rl-ease) both;
}}
[data-testid="stVegaLiteChart"] .mark-line path,
[data-testid="stArrowVegaLiteChart"] .mark-line path,
[data-testid="stVegaLiteChart"] .mark-area path,
[data-testid="stArrowVegaLiteChart"] .mark-area path {{
  animation: yk-reveal-x 0.4s var(--rl-ease) both;
}}
[data-testid="stVegaLiteChart"] .mark-symbol path,
[data-testid="stVegaLiteChart"] .mark-point path,
[data-testid="stVegaLiteChart"] .mark-rect path,
[data-testid="stVegaLiteChart"] .mark-bar path,
[data-testid="stArrowVegaLiteChart"] .mark-symbol path,
[data-testid="stArrowVegaLiteChart"] .mark-point path,
[data-testid="stArrowVegaLiteChart"] .mark-rect path,
[data-testid="stArrowVegaLiteChart"] .mark-bar path {{
  animation: yk-mark-in 0.28s var(--rl-ease) both;
}}
[data-testid="stVegaLiteChart"] .role-axis,
[data-testid="stVegaLiteChart"] .role-legend,
[data-testid="stVegaLiteChart"] .role-grid,
[data-testid="stArrowVegaLiteChart"] .role-axis,
[data-testid="stArrowVegaLiteChart"] .role-legend,
[data-testid="stArrowVegaLiteChart"] .role-grid {{
  animation: none !important;
}}

[data-testid="stDataFrame"] {{
  border-radius: 12px !important;
  overflow: hidden;
  border: 1px solid var(--rl-line);
  box-shadow: var(--rl-shadow);
}}

.rl-book-wrap {{
  width: 100%;
  height: calc(100dvh - 17rem);
  max-height: calc(100dvh - 17rem);
  overflow: auto;
  border: 1px solid var(--rl-line);
  border-radius: 12px;
  box-shadow: var(--rl-shadow);
  background: #fff;
}}
.rl-book {{
  width: 100%;
  border-collapse: collapse;
  font-size: 0.8125rem;
}}
.rl-book th {{
  position: sticky;
  top: 0;
  z-index: 1;
  padding: 0.65rem 0.75rem;
  text-align: left;
  font-size: 0.72rem;
  font-weight: 650;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  color: var(--rl-muted);
  background: #fff;
  border-bottom: 1px solid var(--rl-line);
  white-space: nowrap;
}}
.rl-book td {{
  padding: 0.7rem 0.75rem;
  border-bottom: 1px solid var(--rl-line);
  color: var(--rl-ink);
  white-space: nowrap;
}}
.rl-book tbody tr {{
  position: relative;
  cursor: pointer;
}}
.rl-book tbody tr:last-child td {{
  border-bottom: none;
}}
.rl-book tbody tr:hover td {{
  background: color-mix(in oklab, var(--youkti-primary) 5%, white);
}}
.rl-book a.rl-book-hit {{
  display: block;
  margin: -0.7rem -0.75rem;
  padding: 0.7rem 0.75rem;
  color: inherit;
  text-decoration: none;
}}
.rl-book td:first-child a.rl-book-hit {{
  font-weight: 650;
}}
.rl-book-status {{
  display: inline-flex;
  align-items: center;
  gap: 0.28rem;
  padding: 0.12rem 0.5rem 0.12rem 0.4rem;
  border-radius: 999px;
  background: color-mix(in oklab, var(--sig) 14%, white);
  color: var(--sig);
  font-weight: 650;
}}
.rl-book-status-icon {{
  font-size: 0.72rem;
  line-height: 1;
}}
.rl-book-muted {{
  color: var(--rl-muted);
}}
.rl-book-bar {{
  display: inline-flex;
  align-items: center;
  gap: 0.45rem;
  min-width: 7.25rem;
}}
.rl-book-bar-track {{
  display: block;
  width: 4.25rem;
  height: 0.45rem;
  border-radius: 999px;
  background: #e5e7eb;
  overflow: hidden;
}}
.rl-book-bar-fill {{
  display: block;
  width: var(--pct);
  height: 100%;
  border-radius: 999px;
  background: var(--sig);
}}
.rl-book-bar-value {{
  font-variant-numeric: tabular-nums;
  font-weight: 650;
  color: var(--rl-ink);
}}


/* Floating AryaChat
   Dock: content-sized popup, bottom-right.
   Fullscreen: simple scrollable page — no height:100% chains (those clipped
   Recent chats + composer under an empty header). */
[class*="st-key-arya_fab"] {{
  position: fixed !important;
  right: 1.25rem !important;
  bottom: 1.25rem !important;
  left: auto !important;
  top: auto !important;
  z-index: 1000 !important;
  width: auto !important;
  height: auto !important;
  margin: 0 !important;
}}
[class*="st-key-arya_fab"] .stButton > button {{
  min-height: 3rem !important;
  padding: 0.65rem 1.15rem !important;
  border-radius: 999px !important;
  box-shadow: 0 10px 28px rgb(15 23 42 / 0.22) !important;
}}
[class*="st-key-arya_dock"] {{
  position: fixed !important;
  right: 1.25rem !important;
  bottom: 1.25rem !important;
  left: auto !important;
  top: auto !important;
  width: min(780px, calc(100vw - 2rem)) !important;
  height: auto !important;
  max-height: min(520px, calc(100vh - 3rem)) !important;
  z-index: 1000 !important;
  border-radius: 1rem !important;
  background: #fff !important;
  border: 1px solid #cbd5e1 !important;
  box-shadow: 0 18px 50px rgb(15 23 42 / 0.22) !important;
  overflow: auto !important;
  padding: 0 !important;
  margin: 0 !important;
  box-sizing: border-box !important;
}}
/*
 * Fullscreen shell only — never match arya_full_header / arya_full_body
 * (class*=\"st-key-arya_full\" is a prefix of those keys and used to paint
 * them fixed inset:0, which blanked the screen and left white below).
 * Panes are position:fixed to the viewport; Streamlit columns will not
 * stretch to a percentage height no matter what the parent says.
 */
[class*="st-key-arya_full"]:not([class*="st-key-arya_full_"]) {{
  position: fixed !important;
  inset: 0 !important;
  width: 100vw !important;
  height: 100vh !important;
  max-width: 100vw !important;
  max-height: 100vh !important;
  z-index: 1200 !important;
  border-radius: 0 !important;
  background: #fff !important;
  border: none !important;
  box-shadow: none !important;
  overflow: hidden !important;
  padding: 0 !important;
  margin: 0 !important;
  box-sizing: border-box !important;
}}
.stApp:has([class*="st-key-arya_full"]:not([class*="st-key-arya_full_"])) [data-testid="stSidebar"],
[data-testid="stApp"]:has([class*="st-key-arya_full"]:not([class*="st-key-arya_full_"])) [data-testid="stSidebar"],
.stApp:has([class*="st-key-arya_full"]:not([class*="st-key-arya_full_"])) [data-testid="stSidebarCollapsedControl"],
[data-testid="stApp"]:has([class*="st-key-arya_full"]:not([class*="st-key-arya_full_"])) [data-testid="stSidebarCollapsedControl"],
.stApp:has([class*="st-key-arya_full"]:not([class*="st-key-arya_full_"])) [data-testid="collapsedControl"],
[data-testid="stApp"]:has([class*="st-key-arya_full"]:not([class*="st-key-arya_full_"])) [data-testid="collapsedControl"] {{
  display: none !important;
  visibility: hidden !important;
  width: 0 !important;
  min-width: 0 !important;
  max-width: 0 !important;
  margin: 0 !important;
  padding: 0 !important;
  border: none !important;
  overflow: hidden !important;
  pointer-events: none !important;
}}
.stApp:has([class*="st-key-arya_full"]:not([class*="st-key-arya_full_"])) [data-testid="stAppViewContainer"],
[data-testid="stApp"]:has([class*="st-key-arya_full"]:not([class*="st-key-arya_full_"])) [data-testid="stAppViewContainer"],
.stApp:has([class*="st-key-arya_full"]:not([class*="st-key-arya_full_"])) [data-testid="stMain"],
[data-testid="stApp"]:has([class*="st-key-arya_full"]:not([class*="st-key-arya_full_"])) [data-testid="stMain"],
.stApp:has([class*="st-key-arya_full"]:not([class*="st-key-arya_full_"])) .main,
[data-testid="stApp"]:has([class*="st-key-arya_full"]:not([class*="st-key-arya_full_"])) .main {{
  margin-left: 0 !important;
  max-width: 100% !important;
  width: 100% !important;
}}
/* Header bar — pinned to the top of the viewport. */
[class*="st-key-arya_full_header"] {{
  position: fixed !important;
  top: 0 !important;
  left: 0 !important;
  right: 0 !important;
  width: 100% !important;
  height: 4.25rem !important;
  padding: 0.85rem 1.5rem !important;
  border-bottom: 1px solid #e2e8f0 !important;
  box-sizing: border-box !important;
  background: #fff !important;
  z-index: 1210 !important;
}}
[class*="st-key-arya_full_header"] [data-testid="stHorizontalBlock"] {{
  width: 100% !important;
  height: auto !important;
  align-items: center !important;
  display: flex !important;
  flex-direction: row !important;
  flex-wrap: nowrap !important;
}}
[class*="st-key-arya_full_header"] [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:first-child {{
  flex: 1 1 auto !important;
  width: auto !important;
  min-width: 0 !important;
  max-width: none !important;
  height: auto !important;
}}
[class*="st-key-arya_full_header"] [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:last-child {{
  flex: 0 0 22rem !important;
  width: 22rem !important;
  min-width: 22rem !important;
  max-width: 22rem !important;
  margin-left: auto !important;
  height: auto !important;
}}
[class*="st-key-arya_full_header"] .stButton > button,
[class*="st-key-arya_full_header"] button {{
  min-height: 2.35rem !important;
  white-space: nowrap !important;
}}
/* Exit fullscreen stays neutral grey in every state — Streamlit's default
   secondary button turns primary-purple on hover and draws a purple focus
   ring, which read as an accent block on the button's edge. */
[class*="st-key-arya-full-collapse"] button,
[class*="st-key-arya-full-collapse"] .stButton > button {{
  background: #fff !important;
  border: 1px solid #cbd5e1 !important;
  color: #334155 !important;
  box-shadow: none !important;
  outline: none !important;
}}
[class*="st-key-arya-full-collapse"] button:hover,
[class*="st-key-arya-full-collapse"] button:focus,
[class*="st-key-arya-full-collapse"] button:focus-visible,
[class*="st-key-arya-full-collapse"] button:active {{
  background: #f1f5f9 !important;
  border-color: #94a3b8 !important;
  color: #0f172a !important;
  box-shadow: none !important;
  outline: none !important;
}}
[class*="st-key-arya-full-collapse"] button::before,
[class*="st-key-arya-full-collapse"] button::after {{
  content: none !important;
  display: none !important;
}}
/* Body is only a mount point — panes are fixed below. */
[class*="st-key-arya_full_body"] {{
  width: 100% !important;
  height: 0 !important;
  overflow: visible !important;
  padding: 0 !important;
  margin: 0 !important;
}}
/* Side-by-side panes (dock keeps flow layout; fullscreen panes are fixed). */
[class*="st-key-arya_dock"] [data-testid="stHorizontalBlock"],
[class*="st-key-arya_full"]:not([class*="st-key-arya_full_"]) [data-testid="stHorizontalBlock"] {{
  display: flex !important;
  flex-direction: row !important;
  flex-wrap: nowrap !important;
  align-items: stretch !important;
  width: 100% !important;
  gap: 0 !important;
}}
[class*="st-key-arya_full_header"] [data-testid="stHorizontalBlock"],
[class*="st-key-arya_composer"] [data-testid="stHorizontalBlock"] {{
  height: auto !important;
}}
[class*="st-key-arya_dock"] [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:first-child {{
  flex: 0 0 11rem !important;
  width: 11rem !important;
  min-width: 11rem !important;
  max-width: 11rem !important;
}}
[class*="st-key-arya_dock"] [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:last-child {{
  flex: 1 1 auto !important;
  min-width: 0 !important;
  width: auto !important;
}}
[class*="st-key-arya_composer"] [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:first-child {{
  flex: 1 1 auto !important;
  width: auto !important;
  min-width: 0 !important;
  max-width: none !important;
  height: auto !important;
}}
[class*="st-key-arya_composer"] [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:last-child {{
  flex: 0 0 2.75rem !important;
  width: 2.75rem !important;
  min-width: 2.75rem !important;
  max-width: 2.75rem !important;
  height: auto !important;
}}
/* Left rail — fixed from under the header to the bottom of the screen. */
[class*="st-key-arya_left"] {{
  background: #f8fafc !important;
  border-right: 1px solid #e2e8f0 !important;
  padding: 0.75rem 0.65rem !important;
  box-sizing: border-box !important;
}}
[class*="st-key-arya_full"] [class*="st-key-arya_left"] {{
  position: fixed !important;
  top: 4.25rem !important;
  left: 0 !important;
  bottom: 0 !important;
  width: 17.5rem !important;
  height: auto !important;
  z-index: 1205 !important;
  background: #f0f4f9 !important;
  padding: 1rem 0.85rem 0.85rem !important;
  overflow: hidden !important;
  display: flex !important;
  flex-direction: column !important;
}}
[class*="st-key-arya_full"] [class*="st-key-arya_left"] > div,
[class*="st-key-arya_full"] [class*="st-key-arya_left"] > div > [data-testid="stVerticalBlock"] {{
  height: 100% !important;
  min-height: 0 !important;
  display: flex !important;
  flex-direction: column !important;
  gap: 0.55rem !important;
  overflow: hidden !important;
}}
[class*="st-key-arya_left_scroll"] {{
  max-height: 12rem !important;
  overflow-x: hidden !important;
  overflow-y: auto !important;
}}
[class*="st-key-arya_full"] [class*="st-key-arya_left_scroll"] {{
  flex: 1 1 auto !important;
  max-height: none !important;
  min-height: 0 !important;
  overflow-y: auto !important;
}}
[class*="st-key-arya_left_foot"] {{
  display: flex !important;
  flex-direction: column !important;
  gap: 0.45rem !important;
  margin-top: 0.75rem !important;
  padding-top: 0.75rem !important;
  border-top: 1px solid #e2e8f0 !important;
}}
[class*="st-key-arya_full"] [class*="st-key-arya_left_foot"] {{
  flex: 0 0 auto !important;
  margin-top: auto !important;
  background: #f0f4f9 !important;
}}
[class*="st-key-arya_left_foot"] .stButton > button,
[class*="st-key-arya_left_foot"] button {{
  width: 100% !important;
  min-height: 2.35rem !important;
}}
/* Right pane (dock): arya_right_idle / arya_right_chat. Fullscreen uses
   its own key, arya_chatpane, below. */
[class*="st-key-arya_right"] {{
  padding: 0.85rem 1.15rem 0.9rem !important;
  box-sizing: border-box !important;
  background: #fff !important;
}}
/*
 * Fullscreen right pane (arya_chatpane), idle or chatting — the thread and
 * the chatbar are each pinned to the viewport, not laid out by flex. A flex
 * height chain has to pass through every Streamlit wrapper (BorderWrapper >
 * div > VerticalBlock) and one wrapper without a definite height leaves the
 * thread a few lines tall with the bar floating mid-screen. Fixed boxes need
 * no chain: the thread owns the space between the header and the bar and
 * scrolls; the bar owns the bottom strip in both states, so it never moves
 * when the first answer arrives. Geometry mirrors the panes above: header
 * 4.25rem, left rail 17.5rem, bar strip --arya-bar-h.
 */
[class*="st-key-arya_chatpane"] {{
  --arya-bar-h: 5.5rem;
  position: fixed !important;
  top: 4.25rem !important;
  left: 17.5rem !important;
  right: 0 !important;
  bottom: 0 !important;
  width: auto !important;
  height: auto !important;
  padding: 0 !important;
  margin: 0 !important;
  background: #fff !important;
  overflow: hidden !important;
  z-index: 1205 !important;
  box-sizing: border-box !important;
}}
[class*="st-key-arya_full"] [class*="st-key-arya_chatpane"] [class*="st-key-arya_thread"] {{
  position: fixed !important;
  top: 4.25rem !important;
  left: 17.5rem !important;
  right: 0 !important;
  bottom: var(--arya-bar-h) !important;
  width: auto !important;
  max-width: none !important;
  height: auto !important;
  max-height: none !important;
  margin: 0 !important;
  padding: 1.25rem 2rem 0.75rem !important;
  overflow-x: hidden !important;
  overflow-y: auto !important;
  box-sizing: border-box !important;
  z-index: 1206 !important;
}}
[class*="st-key-arya_full"] [class*="st-key-arya_chatpane"] [class*="st-key-arya_composer"] {{
  position: fixed !important;
  left: 17.5rem !important;
  right: 0 !important;
  bottom: 0 !important;
  width: auto !important;
  max-width: none !important;
  height: var(--arya-bar-h) !important;
  margin: 0 !important;
  padding: 0.75rem 2rem 1rem !important;
  background: #fff !important;
  box-sizing: border-box !important;
  z-index: 1207 !important;
}}
/* Content stays a readable column, centred in the pane. */
[class*="st-key-arya_chatpane"] [class*="st-key-arya_thread"] > div > [data-testid="stVerticalBlock"],
[class*="st-key-arya_chatpane"] [class*="st-key-arya_composer"] > div > [data-testid="stVerticalBlock"] {{
  width: 100% !important;
  max-width: 64rem !important;
  margin-left: auto !important;
  margin-right: auto !important;
}}
/* Idle: the hero sits in the middle of the thread area; the bar stays put. */
[class*="st-key-arya_chatpane"] [class*="st-key-arya_thread"]:has(.rl-arya-hero) > div,
[class*="st-key-arya_chatpane"] [class*="st-key-arya_thread"]:has(.rl-arya-hero) > div > [data-testid="stVerticalBlock"] {{
  height: 100% !important;
  min-height: 0 !important;
  display: flex !important;
  flex-direction: column !important;
  justify-content: center !important;
}}
[class*="st-key-arya_chatpane"] .rl-arya-hero {{
  height: auto !important;
  min-height: 0 !important;
  padding: 1.25rem 1.5rem 0.5rem !important;
}}
.rl-arya-widget-title {{
  margin: 0;
  padding: 0.15rem 0;
  font-size: 1.2rem;
  font-weight: 700;
  letter-spacing: -0.02em;
  color: #0f172a;
}}
.rl-arya-hero {{
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 1.5rem 0.75rem;
  text-align: center;
  box-sizing: border-box;
}}
.rl-arya-hero-title {{
  margin: 0;
  font-size: clamp(1.15rem, 2vw, 1.45rem);
  font-weight: 650;
  letter-spacing: -0.03em;
  color: #0f172a;
}}
.rl-arya-hero-sub {{
  margin: 0.45rem 0 0;
  font-size: 0.95rem;
  color: #64748b;
}}
[class*="st-key-arya_full"] .rl-arya-hero-title {{
  font-size: clamp(1.4rem, 2.5vw, 1.9rem);
}}
[class*="st-key-arya_thread"] {{
  max-height: 16rem !important;
  overflow-y: auto !important;
  padding: 0.15rem 0.1rem 0.35rem !important;
}}
/* The autoscroll script (aryachat._stick_thread_to_bottom) rides in a
   script-only st.html; drop its wrapper so it adds no flex gap in the thread. */
[class*="st-key-arya_thread"] [data-testid="stElementContainer"]:has(> [data-testid="stHtml"]) {{
  display: none !important;
}}
[class*="st-key-arya_composer"] {{
  padding-top: 0.35rem !important;
}}
/* Dock while chatting: same pin — thread scrolls, bar stays down. */
[class*="st-key-arya_dock"] [class*="st-key-arya_right_chat"] > div > [data-testid="stVerticalBlock"],
[class*="st-key-arya_dock"] [class*="st-key-arya_right"]:not(:has(.rl-arya-hero)) > div > [data-testid="stVerticalBlock"] {{
  display: flex !important;
  flex-direction: column !important;
  min-height: 0 !important;
  max-height: min(26rem, calc(100vh - 10rem)) !important;
}}
[class*="st-key-arya_dock"] [class*="st-key-arya_right_chat"] [class*="st-key-arya_thread"],
[class*="st-key-arya_dock"] [class*="st-key-arya_right"]:not(:has(.rl-arya-hero)) [class*="st-key-arya_thread"] {{
  flex: 1 1 0 !important;
  min-height: 0 !important;
  max-height: none !important;
  overflow-y: auto !important;
}}
[class*="st-key-arya_dock"] [class*="st-key-arya_right_chat"] [class*="st-key-arya_composer"],
[class*="st-key-arya_dock"] [class*="st-key-arya_right"]:not(:has(.rl-arya-hero)) [class*="st-key-arya_composer"] {{
  flex: 0 0 auto !important;
  margin-top: auto !important;
  background: #fff !important;
}}
[class*="st-key-arya_composer"] [data-testid="stForm"] {{
  border: none !important;
  padding: 0 !important;
}}
[class*="st-key-arya_composer"] [data-testid="stHorizontalBlock"] {{
  align-items: center !important;
  gap: 0.55rem !important;
  height: auto !important;
}}
[class*="st-key-arya_composer"] [data-testid="stFormSubmitButton"] button {{
  width: 2.75rem !important;
  min-width: 2.75rem !important;
  height: 2.75rem !important;
  border-radius: 999px !important;
  font-size: 1.15rem !important;
  font-weight: 700 !important;
  padding: 0 !important;
  gap: 0 !important;
}}
[class*="st-key-arya_composer"] [data-testid="stFormSubmitButton"] button [data-testid="stIconMaterial"] {{
  font-size: 1.25rem !important;
}}
[class*="st-key-arya_composer"] [data-baseweb="input"],
[class*="st-key-arya_composer"] [data-testid="stTextInputRootElement"] {{
  min-height: 2.75rem !important;
  border-radius: 1.35rem !important;
  border: 1px solid #111827 !important;
}}
[class*="st-key-arya_composer"] [data-testid="InputInstructions"] {{
  display: none !important;
}}
[class*="st-key-arya_dock"] [data-testid="stChatMessage"],
[class*="st-key-arya_full"] [data-testid="stChatMessage"] {{
  max-width: 100%;
}}

[class*="st-key-account_pagination"] {{
  margin-top: 1rem !important;
  padding-top: 0.15rem !important;
  min-height: 2.75rem !important;
  overflow: visible !important;
}}
[class*="st-key-account_pagination"] [data-testid="stHorizontalBlock"],
[class*="st-key-account_pagination"] [data-testid="stVerticalBlock"],
[class*="st-key-account_pagination"] [data-testid="stColumn"],
[class*="st-key-account_pagination"] [data-testid="element-container"] {{
  overflow: visible !important;
}}
[class*="st-key-account-page-"] {{
  flex: 0 0 2rem !important;
  width: 2rem !important;
  min-width: 2rem !important;
}}
[class*="st-key-account-page-"] .stButton > button,
[class*="st-key-account-page-"] button {{
  width: 2rem !important;
  min-width: 2rem !important;
  max-width: 2rem !important;
  min-height: 2rem !important;
  height: 2rem !important;
  padding: 0 !important;
}}
[class*="st-key-account_rows_per_page"] {{
  max-width: 5.5rem;
}}

/* Always-on outline — do not clear the wrapper border first, or it never shows. */
[class*="st-key-account_search"] {{
  width: 100%;
}}
[class*="st-key-account_search"] [data-baseweb="input"],
[class*="st-key-account_search"] [data-baseweb="input"] > div,
[class*="st-key-account_search"] [data-testid="stTextInputRootElement"] {{
  min-height: 2.5rem !important;
  border: 1px solid var(--rl-line) !important;
  border-radius: 0.375rem !important;
  background: #fff !important;
  box-shadow: 0 0 0 1px var(--rl-line) !important;
}}
[class*="st-key-account_search"] input {{
  min-height: 2.5rem !important;
  border: none !important;
  box-shadow: none !important;
  outline: none !important;
  background: transparent !important;
}}
[class*="st-key-account_search"]:hover [data-baseweb="input"],
[class*="st-key-account_search"]:hover [data-baseweb="input"] > div {{
  border-color: var(--youkti-primary) !important;
  box-shadow: 0 0 0 1px var(--youkti-primary) !important;
}}
[class*="st-key-account_search"]:focus-within [data-baseweb="input"],
[class*="st-key-account_search"]:focus-within [data-baseweb="input"] > div {{
  border-color: var(--youkti-primary) !important;
  box-shadow: 0 0 0 3px color-mix(in oklab, var(--youkti-primary) 28%, transparent) !important;
}}

[data-testid="stPopover"] button {{
  min-height: 2.5rem !important;
  border: 1px solid var(--rl-line) !important;
  border-radius: 0.375rem !important;
  background: #fff !important;
  box-shadow: none !important;
}}
[data-testid="stPopover"] button:hover {{
  border-color: var(--youkti-primary) !important;
}}
[class*="st-key-account_filters"] button {{
  gap: 0.4rem !important;
}}
[class*="st-key-account_filters"] button::after,
[class*="st-key-account_filters"] button::before {{
  content: none !important;
  display: none !important;
}}
/* Funnel stays; anything after the label is Streamlit's chevron. */
[class*="st-key-account_filters"] button [data-testid="stIconMaterial"] ~ [data-testid="stIconMaterial"],
[class*="st-key-account_filters"] button [data-testid="stIcon"] ~ [data-testid="stIcon"],
[class*="st-key-account_filters"] button p ~ [data-testid="stIconMaterial"],
[class*="st-key-account_filters"] button p ~ [data-testid="stIcon"],
[class*="st-key-account_filters"] button [data-testid="stMarkdownContainer"] ~ [data-testid="stIconMaterial"],
[class*="st-key-account_filters"] button [data-testid="stMarkdownContainer"] ~ [data-testid="stIcon"],
[class*="st-key-account_filters"] button [data-testid="stExpanderToggleIcon"],
[class*="st-key-account_filters"] button > div > :last-child:not(:first-child):has([data-testid="stIconMaterial"]),
[class*="st-key-account_filters"] button > div > :last-child:not(:first-child):has(svg),
[class*="st-key-account_filters"] button > :last-child:not(:first-child):has([data-testid="stIconMaterial"]),
[class*="st-key-account_filters"] button > div > :last-child:not(:first-child):has(svg) {{
  display: none !important;
}}

[class*="st-key-account_filter_"] [data-baseweb="select"] > div,
[class*="st-key-account_filter_"] [data-baseweb="select"] > div:first-child,
[data-testid="stPopover"] [data-baseweb="select"] > div {{
  min-height: 2.5rem !important;
  border: 1px solid var(--rl-line) !important;
  border-radius: 0.375rem !important;
  background: #fff !important;
  box-shadow: 0 0 0 1px var(--rl-line) !important;
}}
[class*="st-key-account_filter_"]:hover [data-baseweb="select"] > div,
[data-testid="stPopover"] [data-testid="stSelectbox"]:hover [data-baseweb="select"] > div,
[data-testid="stPopover"] [data-testid="stMultiSelect"]:hover [data-baseweb="select"] > div {{
  border-color: var(--youkti-primary) !important;
  box-shadow: 0 0 0 1px var(--youkti-primary) !important;
}}
[class*="st-key-account_filter_"]:focus-within [data-baseweb="select"] > div,
[data-testid="stPopover"] [data-testid="stSelectbox"]:focus-within [data-baseweb="select"] > div,
[data-testid="stPopover"] [data-testid="stMultiSelect"]:focus-within [data-baseweb="select"] > div {{
  border-color: var(--youkti-primary) !important;
  box-shadow: 0 0 0 3px color-mix(in oklab, var(--youkti-primary) 28%, transparent) !important;
}}

/* Radio pills */
[data-testid="stRadio"] [role="radiogroup"] {{
  gap: 0.35rem;
}}
[data-testid="stRadio"] label {{
  transition: transform 0.2s var(--rl-ease);
}}
[data-testid="stRadio"] label:hover {{
  transform: translateY(-1px);
}}

@keyframes rl-spin {{
  to {{ transform: rotate(360deg); }}
}}

/* Youkti circle — replace Streamlit's fade/blur while the script reruns. */
[data-testid="stApp"][data-test-script-state="running"] [data-testid="stAppViewContainer"],
[data-testid="stApp"][data-test-script-state="rerunRequested"] [data-testid="stAppViewContainer"] {{
  filter: none !important;
}}
[data-testid="stApp"][data-test-script-state="running"]::after,
[data-testid="stApp"][data-test-script-state="rerunRequested"]::after {{
  content: "";
  position: fixed;
  inset: 0;
  background: color-mix(in oklab, #fff 64%, transparent);
  z-index: 9998;
  pointer-events: all;
}}
[data-testid="stApp"][data-test-script-state="running"]::before,
[data-testid="stApp"][data-test-script-state="rerunRequested"]::before {{
  content: "";
  position: fixed;
  top: 50%;
  left: 50%;
  width: 2.35rem;
  height: 2.35rem;
  margin: -1.175rem 0 0 -1.175rem;
  border: 3px solid color-mix(in oklab, var(--youkti-primary) 22%, white);
  border-top-color: var(--youkti-primary);
  border-radius: 50%;
  animation: rl-spin 0.7s linear infinite;
  z-index: 9999;
  pointer-events: none;
}}
/* AryaChat already streams in-place — don't veil dock/fullscreen with the page loader. */
[data-testid="stApp"]:has([class*="st-key-arya_dock"])[data-test-script-state="running"]::before,
[data-testid="stApp"]:has([class*="st-key-arya_dock"])[data-test-script-state="rerunRequested"]::before,
[data-testid="stApp"]:has([class*="st-key-arya_dock"])[data-test-script-state="running"]::after,
[data-testid="stApp"]:has([class*="st-key-arya_dock"])[data-test-script-state="rerunRequested"]::after,
[data-testid="stApp"]:has([class*="st-key-arya_full"]:not([class*="st-key-arya_full_"]))[data-test-script-state="running"]::before,
[data-testid="stApp"]:has([class*="st-key-arya_full"]:not([class*="st-key-arya_full_"]))[data-test-script-state="rerunRequested"]::before,
[data-testid="stApp"]:has([class*="st-key-arya_full"]:not([class*="st-key-arya_full_"]))[data-test-script-state="running"]::after,
[data-testid="stApp"]:has([class*="st-key-arya_full"]:not([class*="st-key-arya_full_"]))[data-test-script-state="rerunRequested"]::after {{
  content: none !important;
  display: none !important;
}}
[data-testid="stSkeletonElement"],
.stSkeleton {{
  display: none !important;
}}

[data-testid="stSpinner"] {{
  letter-spacing: -0.01em;
}}

/* Caption + markdown polish */
[data-testid="stCaptionContainer"] {{
  color: var(--rl-secondary) !important;
}}

::selection {{
  background: color-mix(in srgb, var(--rl-blue) 28%, transparent);
}}

/* Scrollbar */
::-webkit-scrollbar {{ width: 10px; height: 10px; }}
::-webkit-scrollbar-track {{ background: transparent; }}
::-webkit-scrollbar-thumb {{
  background: var(--rl-axis);
  border-radius: 999px;
  border: 2px solid var(--rl-paper);
}}

@media (max-width: 1100px) {{
  .rl-signal-row {{ grid-template-columns: repeat(3, minmax(0, 1fr)); }}
}}
@media (max-width: 720px) {{
  .rl-signal-row, .rl-kpi-row {{ grid-template-columns: 1fr 1fr; }}
  .rl-title {{ font-size: 1.38rem; }}
}}

@media (prefers-reduced-motion: reduce) {{
  *, *::before, *::after {{
    animation: none !important;
    transition: none !important;
  }}
  [data-testid="stApp"][data-test-script-state="running"]::before {{
    animation: rl-spin 0.7s linear infinite !important;
  }}
}}

/* Last so Streamlit / earlier select rules cannot wipe the outline. */
[class*="st-key-account_rows_per_page"] {{
  border: 1px solid #d1d5db !important;
  border-radius: 0.375rem !important;
  background: #fff !important;
  box-sizing: border-box !important;
  max-width: 6.25rem !important;
  margin-left: -0.35rem !important;
  min-height: 2.25rem !important;
  height: auto !important;
  overflow: visible !important;
}}
[class*="st-key-account_rows_per_page"] [data-testid="stSelectbox"],
[class*="st-key-account_rows_per_page"] [data-baseweb="select"],
[class*="st-key-account_rows_per_page"] [data-testid="stSelectbox"] > div,
[class*="st-key-account_rows_per_page"] [data-baseweb="select"] > div {{
  min-height: 2.25rem !important;
  height: auto !important;
  border: none !important;
  box-shadow: none !important;
  padding-top: 0 !important;
  padding-bottom: 0 !important;
  overflow: visible !important;
}}
[class*="st-key-account_rows_per_page"]:hover {{
  border-color: var(--youkti-primary) !important;
}}
[class*="st-key-account_rows_per_page"]:focus-within {{
  border-color: var(--youkti-primary) !important;
  box-shadow: 0 0 0 3px color-mix(in oklab, var(--youkti-primary) 28%, transparent) !important;
}}

/* Chat bar — black at rest, purple only while focused. */
[data-testid="stChatInput"] > div,
[data-testid="stChatInput"] [data-baseweb="base-input"],
[data-testid="stChatInput"] [data-baseweb="textarea"] {{
  border: 1px solid #111827 !important;
  border-radius: 1.5rem !important;
  background: #fff !important;
  box-shadow: none !important;
}}
[data-testid="stChatInput"] textarea {{
  border: none !important;
  box-shadow: none !important;
  outline: none !important;
  background: transparent !important;
}}
[data-testid="stChatInput"]:hover > div,
[data-testid="stChatInput"]:hover [data-baseweb="base-input"],
[data-testid="stChatInput"]:hover [data-baseweb="textarea"] {{
  border-color: #111827 !important;
  box-shadow: none !important;
}}
[data-testid="stChatInput"]:focus-within > div,
[data-testid="stChatInput"]:focus-within [data-baseweb="base-input"],
[data-testid="stChatInput"]:focus-within [data-baseweb="textarea"] {{
  border-color: var(--youkti-primary) !important;
  box-shadow: 0 0 0 3px color-mix(in oklab, var(--youkti-primary) 28%, transparent) !important;
}}
</style>
"""
