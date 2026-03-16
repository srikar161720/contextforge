"""
app/pages/5_report.py — HTML report generation and download.

Generates a self-contained HTML report (via core/report_generator.py) from
the ablation results, diet plan, and payload. The report is cached in
st.session_state to avoid regeneration on every page visit.

An optional "Include Code Interpreter analysis" toggle passes the BedrockClient
to report_generator so Nova Code Interpreter adds a statistical narrative.

Session state read:
  - ablation_results (AblationResults)
  - payload          (ContextPayload)
  - diet_plan        (str — markdown)
  - bedrock_client   (BedrockClient) [optional]
  - payload_name     (str)
  - experiment_config (ExperimentConfig)

Session state written:
  - report_html (str — complete HTML)
"""

from __future__ import annotations

import streamlit as st
import streamlit.components.v1 as components

from app.components.layout import apply_layout
from core.report_generator import generate_report

apply_layout()

# ── Page ──────────────────────────────────────────────────────────────────────

st.title("📥 Download Report")
st.markdown(
    "Generate and download a self-contained HTML report with embedded Plotly charts, "
    "the Context Diet Plan, and an optional Code Interpreter statistical narrative."
)

results = st.session_state.get("ablation_results")
payload = st.session_state.get("payload")
diet_plan = st.session_state.get("diet_plan")
client  = st.session_state.get("bedrock_client")
payload_name     = st.session_state.get("payload_name", "context_payload")
experiment_config = st.session_state.get("experiment_config")
experiment_mode  = experiment_config.mode.value if experiment_config else "demo"

# ── Guards ────────────────────────────────────────────────────────────────────

missing = []
if results is None:
    missing.append("Ablation Results (run an experiment)")
if payload is None:
    missing.append("Context Payload (upload a payload)")
if diet_plan is None:
    missing.append("Context Diet Plan (visit the Diet Plan page)")

if missing:
    st.info(
        "The following are required before generating the report:\n\n"
        + "\n".join(f"- {m}" for m in missing)
    )
    nav_cols = st.columns(len(missing))
    page_map = {
        "Ablation Results (run an experiment)":              "pages/2_progress.py",
        "Context Payload (upload a payload)":                "pages/1_upload.py",
        "Context Diet Plan (visit the Diet Plan page)":      "pages/4_diet_plan.py",
    }
    for col, m in zip(nav_cols, missing):
        with col:
            target = page_map.get(m, "pages/1_upload.py")
            label  = m.split("(")[0].strip().title()
            if st.button(f"→ {label}", use_container_width=True):
                st.switch_page(target)
    st.stop()

# ── Summary banner ────────────────────────────────────────────────────────────

st.markdown("---")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Sections", len(results.section_impacts))
with col2:
    st.metric("Token Reduction", f"{results.lean_token_reduction:.1%}")
with col3:
    st.metric("Quality Retention", f"{results.lean_quality_retention:.1%}")
with col4:
    st.metric("Experiment Cost", f"${results.total_cost:.4f}")

st.markdown("---")

# ── Report generation options ─────────────────────────────────────────────────

st.subheader("Report Options")

include_ci = st.checkbox(
    "Include Code Interpreter analysis",
    value=False,
    help=(
        "When enabled, Nova Code Interpreter will verify the ablation statistics "
        "and add a short analytical narrative to the report. "
        "Adds ~30–60 seconds and a small API cost."
    ),
)

if include_ci and client is None:
    st.warning(
        "No active Bedrock client in session. "
        "Code Interpreter analysis will be skipped — a local fallback narrative will be used instead."
    )

# ── Generate / cache report ───────────────────────────────────────────────────

report_html: str | None = st.session_state.get("report_html")

col_gen, col_regen = st.columns([3, 1])
with col_gen:
    generate_btn = st.button(
        "⚙️ Generate Report",
        type="primary",
        use_container_width=True,
        disabled=(report_html is not None),
    )
with col_regen:
    regen_btn = st.button(
        "🔄 Regenerate",
        use_container_width=True,
        help="Discard cached report and regenerate.",
    )

if regen_btn:
    st.session_state.pop("report_html", None)
    report_html = None
    st.rerun()

if (generate_btn or report_html is None) and report_html is None:
    client_for_report = client if (include_ci and client is not None) else None

    with st.spinner("Generating HTML report… this may take a few seconds."):
        try:
            report_html = generate_report(
                results=results,
                payload=payload,
                diet_plan=diet_plan,
                client=client_for_report,
                payload_name=payload_name,
                experiment_mode=experiment_mode,
            )
            st.session_state["report_html"] = report_html
            st.success("Report generated successfully!")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Report generation failed: {exc}")
            st.stop()

# ── Download button ───────────────────────────────────────────────────────────

if report_html is not None:
    st.markdown("---")

    file_name = f"contextforge_{payload_name}_{experiment_mode}.html"

    st.download_button(
        label="📥 Download HTML Report",
        data=report_html,
        file_name=file_name,
        mime="text/html",
        use_container_width=True,
    )

    st.markdown(
        '<p style="font-size:0.8rem; color:#9ca3af; text-align:center; margin-top:4px;">'
        "The report is fully self-contained — open it in any browser without an internet connection "
        "(Plotly loads from CDN on first open)."
        "</p>",
        unsafe_allow_html=True,
    )

    # ── Report preview ──────────────────────────────────────────────────────

    with st.expander("👁 Preview Report", expanded=False):
        st.markdown(
            '<p style="font-size:0.82rem; color:#6b7280; margin-bottom:0.5rem;">'
            "Scrollable preview of the generated report. "
            "Interactive charts may be limited in the preview — download for full interactivity."
            "</p>",
            unsafe_allow_html=True,
        )
        components.html(report_html, height=700, scrolling=True)

# ── Navigation ────────────────────────────────────────────────────────────────

st.markdown("---")
if st.button("← Back to Diet Plan", use_container_width=False):
    st.switch_page("pages/4_diet_plan.py")
