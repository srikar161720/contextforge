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

st.markdown(
    """
    <div style="text-align:center; padding: 2.5rem 1rem 1.5rem 1rem;">
        <span style="font-size:3rem;">🔬</span>
        <h1 style="margin:0.4rem 0 0.2rem 0; font-size:2rem;
                   color:#1f2937; font-weight:700;">
            ContextForge
        </h1>
        <p style="font-size:0.95rem; color:#6b7280; margin:0.2rem 0 0.8rem 0;">
            LLM Context Ablation Testing
        </p>
        <p style="font-size:0.88rem; color:#4b5563; max-width:600px;
                  margin:0 auto 1.5rem auto; line-height:1.6;">
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
