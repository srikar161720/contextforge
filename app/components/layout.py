"""
app/components/layout.py — Shared layout: global CSS theme + sidebar.

Call ``apply_layout()`` at the top of every page (after ``st.set_page_config``
if present) to inject the pastel CSS theme and render the consistent sidebar
with workflow-status indicators and experiment cost.
"""

from __future__ import annotations

import streamlit as st

# ── Global CSS — pastel modern theme ─────────────────────────────────────────

_FONT_LINKS = """
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@24,400,0,0">
<link rel="stylesheet" href="https://fonts.googleapis.com/icon?family=Material+Icons+Round">
"""

_CSS = """
<style>
/* ── Base & fonts ── */
html, body, .stApp {
    font-family: 'Inter', sans-serif;
}

/* Preserve Material Symbols on Streamlit icon elements */
.material-symbols-rounded,
.material-symbols-outlined,
[class*="material-symbols"] {
    font-family: 'Material Symbols Rounded' !important;
}

/* Preserve Material Icons ligatures used by Streamlit internals */
.material-icons,
.material-icons-round,
[class*="material-icons"] {
    font-family: 'Material Icons Round' !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #f0ebff 0%, #eff6ff 100%);
    border-right: 1px solid #e5e7eb;
}
[data-testid="stSidebar"] > div:first-child {
    transition: opacity 0.15s ease-in;
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: #4b5563;
    font-size: 0.85rem;
}

/* ── Capitalize sidebar page names ── */
[data-testid="stSidebarNav"] a span {
    text-transform: capitalize;
}

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: #ffffff;
    border-radius: 12px;
    padding: 1rem 1.25rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.07);
    border-left: 4px solid #7c6fea;
}
[data-testid="stMetricLabel"] {
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    color: #6b7280 !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
[data-testid="stMetricValue"] {
    font-size: 1.65rem !important;
    font-weight: 700 !important;
    color: #1f2937 !important;
}

/* ── Primary button ── */
.stButton > button[kind="primary"],
.stButton > button {
    background: linear-gradient(135deg, #7c6fea 0%, #60a5fa 100%);
    color: #ffffff;
    border: none;
    border-radius: 8px;
    padding: 0.5rem 1.4rem;
    font-weight: 600;
    font-size: 0.9rem;
    transition: opacity 0.15s ease, transform 0.1s ease;
    box-shadow: 0 2px 6px rgba(124,111,234,0.35);
}
.stButton > button:hover {
    opacity: 0.9;
    transform: translateY(-1px);
}

/* ── Cancel button (red gradient) ── */
.st-key-cancel-section .stButton > button {
    background: linear-gradient(135deg, #ef4444 0%, #f97316 100%) !important;
    box-shadow: 0 2px 6px rgba(239,68,68,0.35) !important;
}

/* ── Expanders ── */
[data-testid="stExpander"] {
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    overflow: hidden;
    margin-bottom: 0.5rem;
}
[data-testid="stExpander"] summary {
    padding: 0.65rem 1rem;
    background: #fafafa;
    font-weight: 500;
    color: #374151;
}

/* ── Progress bar ── */
[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg, #7c6fea 0%, #60a5fa 100%);
    border-radius: 6px;
}
[data-testid="stProgress"] > div {
    background: #e5e7eb;
    border-radius: 6px;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    border: 2px dashed #c4b5fd;
    border-radius: 12px;
    background: #faf5ff;
    padding: 1rem;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border-radius: 10px;
    overflow: hidden;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}

/* ── Alert / info boxes ── */
[data-testid="stAlert"] {
    border-radius: 10px;
}

/* ── Selectbox & sliders ── */
[data-testid="stSelectbox"] > div,
[data-testid="stSlider"] {
    border-radius: 8px;
}

/* ── Download button ── */
[data-testid="stDownloadButton"] > button {
    background: linear-gradient(135deg, #6ee7b7 0%, #60a5fa 100%);
    color: #1f2937;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    font-size: 0.95rem;
    padding: 0.6rem 1.6rem;
    box-shadow: 0 2px 6px rgba(110,231,183,0.4);
    transition: opacity 0.15s ease;
}
[data-testid="stDownloadButton"] > button:hover {
    opacity: 0.9;
}

/* ── Section/page headers ── */
h1 { color: #1f2937; font-weight: 700; }
h2 { color: #374151; font-weight: 600; border-bottom: 2px solid #c4b5fd; padding-bottom: 0.3rem; }
h3 { color: #374151; font-weight: 600; }

/* ── Spinner ── */
[data-testid="stSpinner"] {
    color: #7c6fea;
}
</style>
"""


def _render_sidebar() -> None:
    """Render the shared sidebar with workflow status and experiment cost."""
    with st.sidebar:
        # ── Workflow status indicators ──
        st.markdown(
            '<p style="font-size:0.75rem; font-weight:600; color:#6b7280; '
            'text-transform:uppercase; letter-spacing:0.06em; margin-bottom:0.5rem;">'
            "WORKFLOW STATUS</p>",
            unsafe_allow_html=True,
        )

        payload = st.session_state.get("payload")
        ablation_results = st.session_state.get("ablation_results")
        diet_plan = st.session_state.get("diet_plan")
        report_html = st.session_state.get("report_html")

        def _status_row(label: str, done: bool) -> str:
            icon = "\u2705" if done else "\u2b1c"
            color = "#059669" if done else "#9ca3af"
            return (
                f'<div style="display:flex; align-items:center; gap:8px; '
                f"margin-bottom:6px; font-size:0.85rem; color:{color};\">"
                f"<span>{icon}</span><span>{label}</span></div>"
            )

        st.markdown(
            _status_row("Upload & Configure", payload is not None)
            + _status_row("Run Experiment", ablation_results is not None)
            + _status_row("View Results", ablation_results is not None)
            + _status_row("Context Diet Plan", diet_plan is not None)
            + _status_row("Download Report", report_html is not None),
            unsafe_allow_html=True,
        )

        # ── Cost display ──
        if ablation_results is not None:
            st.markdown("---")
            st.markdown(
                '<p style="font-size:0.75rem; font-weight:600; color:#6b7280; '
                'text-transform:uppercase; letter-spacing:0.06em; margin-bottom:0.4rem;">'
                "EXPERIMENT COST</p>",
                unsafe_allow_html=True,
            )
            cost = ablation_results.total_cost
            calls = ablation_results.total_api_calls
            st.markdown(
                f'<div style="background:#ffffff; border-radius:10px; padding:0.75rem 1rem; '
                f'box-shadow:0 1px 3px rgba(0,0,0,0.07);">'
                f'<span style="font-size:1.3rem; font-weight:700; color:#7c6fea;">'
                f"${cost:.4f}</span>"
                f'<span style="font-size:0.75rem; color:#9ca3af; margin-left:6px;">'
                f"({calls} API calls)</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.markdown(
            '<p style="font-size:0.72rem; color:#9ca3af; text-align:center;">'
            "Amazon Nova AI Hackathon 2026</p>",
            unsafe_allow_html=True,
        )


def apply_layout() -> None:
    """Inject global CSS and render the shared sidebar.

    Call this once at the top of every page, after ``st.set_page_config``
    (if present).
    """
    st.markdown(_FONT_LINKS, unsafe_allow_html=True)
    st.markdown(_CSS, unsafe_allow_html=True)
    _render_sidebar()
