"""AryaChat loader and checkmark — the same Lottie files youkti-app uses."""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path

import streamlit as st

_DIR = Path(__file__).resolve().parent


@cache
def _animation(name: str) -> dict:
    return json.loads((_DIR / name).read_text(encoding="utf-8"))


def mark_html(kind: str) -> str:
    """A 28px slot the injected player fills. `kind` is spinner or check."""
    return f'<div class="rl-arya-mark rl-arya-mark-{kind}" aria-hidden="true"></div>'


def inject_player() -> None:
    """Load lottie-web and play every spinner / checkmark mark on this page."""
    payload = json.dumps({
        "spinner": _animation("spinner-half-circles.json"),
        "check": _animation("check-in-reveal.json"),
    })
    st.html(
        f"""<script>
(function() {{
  window.__RL_LOTTIE = {payload};
  function mount(el) {{
    if (!window.lottie || !el || el.classList.contains("rl-arya-ready")) return;
    const kind = el.classList.contains("rl-arya-mark-spinner") ? "spinner" : "check";
    const data = window.__RL_LOTTIE[kind];
    if (!data) return;
    el.classList.add("rl-arya-ready");
    el.innerHTML = "";
    window.lottie.loadAnimation({{
      container: el,
      renderer: "svg",
      loop: kind === "spinner",
      autoplay: true,
      animationData: JSON.parse(JSON.stringify(data)),
    }});
  }}
  function scan() {{
    document.querySelectorAll(".rl-arya-mark-spinner, .rl-arya-mark-check").forEach(mount);
  }}
  function boot() {{
    if (!window.__rlLottieObs) {{
      window.__rlLottieObs = new MutationObserver(scan);
      window.__rlLottieObs.observe(document.body, {{ childList: true, subtree: true }});
    }}
    scan();
  }}
  if (window.lottie) {{ boot(); return; }}
  let src = document.getElementById("rl-lottie-src");
  if (!src) {{
    src = document.createElement("script");
    src.id = "rl-lottie-src";
    src.src = "https://cdnjs.cloudflare.com/ajax/libs/lottie-web/5.12.2/lottie.min.js";
    src.onload = boot;
    document.head.appendChild(src);
    return;
  }}
  const wait = setInterval(function() {{
    if (window.lottie) {{ clearInterval(wait); boot(); }}
  }}, 40);
}})();
</script>""",
        unsafe_allow_javascript=True,
    )
