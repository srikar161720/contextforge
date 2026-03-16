"""
app/main.py — ContextForge Streamlit entry point.

Sets up page config, applies the shared layout (global CSS + sidebar),
and renders the landing-page hero with a CTA button. Streamlit's native
multi-page app handles page routing via the pages/ directory naming convention.

Run with:
    streamlit run app/main.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the project root is on sys.path so ``from app.`` imports work
# when Streamlit runs this file as the entry-point script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from app.components.layout import apply_layout

# ── Page config (must be first Streamlit call) ────────────────────────────────

st.set_page_config(
    page_title="ContextForge",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "ContextForge — LLM Context Ablation Testing Tool\n\nBuilt for the Amazon Nova AI Hackathon 2026.",
    },
)

# ── Shared layout (CSS + sidebar) ────────────────────────────────────────────

apply_layout()

# ── Main page — branding hero ────────────────────────────────────────────────

_img_col1, _img_col2, _img_col3 = st.columns([1, 8, 1])
with _img_col2:
    st.image(
        str(Path(__file__).resolve().parent / "assets" / "content.jpg"),
        width=760,
    )

st.markdown(
    """
    <div style="max-width:980px; margin:0 auto; text-align:center; padding: 1.3rem 1rem 0.9rem 1rem;">
        <div style="margin:0.2rem 0 0.25rem 0; font-size:3.35rem; color:#1f2937; font-weight:700; line-height:1.1;">
            ContextForge
        </div>
        <p style="font-size:1.28rem; color:#6b7280; margin:0.3rem 0 1rem 0; font-weight:500;">
            LLM Context Ablation Testing
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div style="max-width:980px; margin:0 auto; text-align:center; padding: 0.9rem 1rem 1.8rem 1rem;">
        <p style="font-size:1.04rem; color:#4b5563; max-width:760px;
                  margin:0 auto 1.9rem auto; line-height:1.7;">
            Systematically remove context sections from your LLM payloads,
            measure quality impact via LLM-as-judge scoring, and get
            optimization recommendations with cost savings projections &mdash;
            powered by Amazon Nova.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

_cta_col1, _cta_col2, _cta_col3 = st.columns([1, 1, 1])
with _cta_col2:
    if st.button("Try it out!", type="primary", use_container_width=True):
        st.switch_page("pages/1_upload.py")
