"""
app/pages/4_diet_plan.py — Context Diet Plan display.

Generates (or re-displays) the Context Diet Plan from Nova extended thinking
(HIGH tier). The plan is cached in st.session_state to avoid re-generation
on every page visit. A "Regenerate" button forces a fresh API call.

Session state read:
  - ablation_results (AblationResults)
  - payload          (ContextPayload)
  - bedrock_client   (BedrockClient)

Session state written:
  - diet_plan (str — markdown)
"""

from __future__ import annotations

import streamlit as st

from core.diet_planner import generate_diet_plan

# ── Page ──────────────────────────────────────────────────────────────────────

st.title("🥗 Context Diet Plan")
st.markdown(
    "An AI-generated optimization guide produced by Nova extended thinking (HIGH tier). "
    "The plan names specific sections to remove, projects token and cost savings, "
    "and explains each recommendation with ablation evidence."
)

results = st.session_state.get("ablation_results")
payload = st.session_state.get("payload")
client  = st.session_state.get("bedrock_client")

if results is None or payload is None:
    st.info("No ablation results found. Run an experiment first.")
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("← Upload Payload"):
            st.switch_page("pages/1_upload.py")
    with col_b:
        if st.button("⚙️ Go to Progress"):
            st.switch_page("pages/2_progress.py")
    st.stop()

if client is None:
    st.warning(
        "No active Bedrock client found in session. "
        "If you refreshed the page, please re-run the experiment from the progress page "
        "to restore the client connection."
    )

# ── Summary stats banner ──────────────────────────────────────────────────────

st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Token Reduction", f"{results.lean_token_reduction:.1%}")
with col2:
    st.metric("Quality Retention", f"{results.lean_quality_retention:.1%}")
with col3:
    removable_count = sum(
        1 for imp in results.section_impacts
        if imp.classification in ("removable", "harmful")
    )
    st.metric("Removable Sections", removable_count)

st.markdown("---")

# ── Generation / cache logic ──────────────────────────────────────────────────

diet_plan: str | None = st.session_state.get("diet_plan")

col_regen, col_nav = st.columns([3, 1])
with col_regen:
    st.markdown(
        '<p style="color:#6b7280; font-size:0.83rem;">'
        "The plan is cached in this session. Use the button to regenerate with a fresh API call.</p>",
        unsafe_allow_html=True,
    )
with col_nav:
    force_regen = st.button("🔄 Regenerate", help="Generate a new diet plan via Nova HIGH tier")

if diet_plan is None or force_regen:
    if client is None:
        st.error(
            "Cannot generate the diet plan — no Bedrock client available. "
            "Re-run the experiment from the progress page."
        )
        st.stop()

    with st.spinner(
        "Generating Context Diet Plan via Nova extended thinking (HIGH)… "
        "This may take 30–90 seconds."
    ):
        try:
            diet_plan = generate_diet_plan(
                results=results,
                payload=payload,
                client=client,
            )
            st.session_state["diet_plan"] = diet_plan
            if force_regen:
                st.success("Diet plan regenerated successfully.")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Failed to generate diet plan: {exc}")
            st.stop()

# ── Display diet plan ─────────────────────────────────────────────────────────

st.markdown(
    '<div style="background:#ffffff; border:1px solid #e5e7eb; border-radius:12px; '
    'padding:1.5rem 2rem; box-shadow:0 1px 4px rgba(0,0,0,0.06); margin-top:1rem;">',
    unsafe_allow_html=True,
)
st.markdown(diet_plan)
st.markdown("</div>", unsafe_allow_html=True)

# ── Navigation ────────────────────────────────────────────────────────────────

st.markdown("---")

col_results, col_report = st.columns(2)
with col_results:
    if st.button("← Back to Results", use_container_width=True):
        st.switch_page("pages/3_results.py")
with col_report:
    if st.button("📥 Download Report →", type="primary", use_container_width=True):
        st.switch_page("pages/5_report.py")
