"""Shared visual design: CSS, header, KPI cards and status badges."""
from __future__ import annotations

import html

import streamlit as st

from config import APP_NAME, APP_SUBTITLE, DEMO_BANNER

PRIMARY = "#0F2A4A"
ACCENT = "#0E9F8E"
STATUS = {
    "AWAITING_VERIFICATION": ("#B54708", "#FEF0C7", "Awaiting verification"),
    "CONFIRMED": ("#067647", "#DCFAE6", "Confirmed"),
    "REJECTED": ("#B42318", "#FEE4E2", "Rejected"),
    "SUSPECTED_LITTERING": ("#B42318", "#FEE4E2", "Suspected littering"),
    "UNCERTAIN": ("#6941C6", "#F4EBFF", "Uncertain"),
    "UNPAID": ("#B54708", "#FEF0C7", "Unpaid (demo)"),
    "PAID_SIMULATED": ("#067647", "#DCFAE6", "Paid (simulated)"),
}
CHART_COLORS = {"AWAITING_VERIFICATION": "#F79009", "CONFIRMED": "#12B76A", "REJECTED": "#F04438",
                "SUSPECTED_LITTERING": "#F04438", "UNCERTAIN": "#7A5AF8"}

CSS = f"""
<style>
:root {{ --primary:{PRIMARY}; --accent:{ACCENT}; }}
.block-container {{ padding-top: 1.4rem; max-width: 1280px; }}
[data-testid="stSidebar"] {{ background: linear-gradient(180deg,#0F2A4A 0%,#12385F 100%); }}
[data-testid="stSidebar"] * {{ color: #E6EEF7 !important; }}
[data-testid="stSidebar"] .stButton button {{ background: rgba(255,255,255,.10); border:1px solid rgba(255,255,255,.35); }}
.scai-card .lbl {{ white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
.scai-hero {{ background: linear-gradient(120deg,#0F2A4A 0%,#145374 60%,#0E9F8E 100%); color:#fff;
  padding: 1.4rem 1.6rem; border-radius: 16px; margin-bottom: 1rem; box-shadow: 0 6px 24px rgba(15,42,74,.18); }}
.scai-hero h1 {{ color:#fff; font-size:1.7rem; margin:0 0 .2rem 0; font-weight:700; }}
.scai-hero p {{ color:#D6E6F5; margin:0; font-size:.95rem; }}
.scai-demo {{ display:inline-block; margin-top:.6rem; background:rgba(255,255,255,.14); border:1px solid rgba(255,255,255,.35);
  padding:.15rem .6rem; border-radius:999px; font-size:.75rem; letter-spacing:.04em; font-weight:600; }}
.scai-card {{ background:#fff; border:1px solid #E4E7EC; border-radius:14px; padding:1rem 1.1rem;
  box-shadow:0 1px 3px rgba(16,24,40,.06); height:100%; }}
.scai-card .lbl {{ color:#667085; font-size:.78rem; text-transform:uppercase; letter-spacing:.05em; font-weight:600; }}
.scai-card .val {{ color:#101828; font-size:1.7rem; font-weight:700; margin-top:.15rem; }}
.scai-card .sub {{ color:#667085; font-size:.8rem; }}
.scai-card.accent {{ border-top:4px solid var(--accent); }}
.scai-badge {{ display:inline-block; padding:.12rem .55rem; border-radius:999px; font-size:.75rem; font-weight:600; }}
.scai-section {{ font-weight:700; color:#101828; font-size:1.05rem; margin:.4rem 0 .5rem 0; }}
.scai-note {{ background:#F8FAFC; border-left:4px solid var(--accent); padding:.7rem .9rem; border-radius:8px;
  color:#344054; font-size:.88rem; }}
.scai-feature {{ background:#fff; border:1px solid #E4E7EC; border-radius:14px; padding:1.1rem; height:100%; }}
.scai-feature h4 {{ margin:.2rem 0 .4rem 0; color:#0F2A4A; }}
.scai-feature p {{ color:#475467; font-size:.9rem; margin:0; }}
div.stButton > button[kind="primary"] {{ background:var(--accent); border-color:var(--accent); }}
</style>
"""


def apply() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def hero(title: str = APP_NAME, subtitle: str = APP_SUBTITLE) -> None:
    st.markdown(
        f"""<div class="scai-hero"><h1>{html.escape(title)}</h1><p>{html.escape(subtitle)}</p>
        <span class="scai-demo">{html.escape(DEMO_BANNER)}</span></div>""",
        unsafe_allow_html=True,
    )


def kpi(col, label: str, value, sub: str = "", accent: bool = False) -> None:
    col.markdown(
        f"""<div class="scai-card{' accent' if accent else ''}"><div class="lbl">{html.escape(label)}</div>
        <div class="val">{html.escape(str(value))}</div><div class="sub">{html.escape(sub)}</div></div>""",
        unsafe_allow_html=True,
    )


def badge(status: str | None) -> str:
    fg, bg, text = STATUS.get(status or "", ("#344054", "#F2F4F7", status or "-"))
    return f'<span class="scai-badge" style="color:{fg};background:{bg}">{html.escape(text)}</span>'


def section(title: str) -> None:
    st.markdown(f'<div class="scai-section">{html.escape(title)}</div>', unsafe_allow_html=True)


def note(text: str) -> None:
    st.markdown(f'<div class="scai-note">{text}</div>', unsafe_allow_html=True)
