"""
app/pages/3_results.py — Interactive ablation results dashboard.

Displays the full set of ablation findings using the four Plotly chart
components plus a sortable section detail table, redundancy cluster list,
ordering recommendations, and lean configuration summary.

Session state read:
  - ablation_results (AblationResults)
  - payload          (ContextPayload)  [optional — used for token totals]
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.components.heatmap import render_heatmap
from app.components.impact_chart import render_impact_chart
from app.components.layout import apply_layout
from app.components.pareto_chart import render_pareto_chart
from app.components.tier_radar import render_tier_radar
from core.models import AblationResults

apply_layout()

# ── Classification colours (pastel, matching the design system) ───────────────
_CLS_COLORS: dict[str, tuple[str, str]] = {
    "essential": ("#fca5a5", "#fef2f2"),
    "moderate":  ("#fdba74", "#fff7ed"),
    "removable": ("#6ee7b7", "#ecfdf5"),
    "harmful":   ("#93c5fd", "#eff6ff"),
}


def _classification_badge(cls: str) -> str:
    """Return an HTML badge string for a classification label."""
    txt, bg = _CLS_COLORS.get(cls, ("#374151", "#f3f4f6"))
    return (
        f'<span style="background:{bg}; color:{txt}; font-size:0.72rem; '
        f'font-weight:600; padding:2px 10px; border-radius:16px;">'
        f"{cls.capitalize()}</span>"
    )


def _chart_container(title: str) -> None:
    """Render a styled chart section header."""
    st.markdown(
        f'<div style="margin-top:1.5rem; margin-bottom:0.5rem;">'
        f'<h3 style="margin:0;">{title}</h3>'
        f"</div>",
        unsafe_allow_html=True,
    )


# ── Page ──────────────────────────────────────────────────────────────────────

st.title("📊 Results Dashboard")

results: AblationResults | None = st.session_state.get("ablation_results")
payload = st.session_state.get("payload")

if results is None:
    st.info("No results yet. Run an experiment first.")
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("← Upload Payload"):
            st.switch_page("pages/1_upload.py")
    with col_b:
        if st.button("⚙️ Go to Progress"):
            st.switch_page("pages/2_progress.py")
    st.stop()

# ── Summary cards ─────────────────────────────────────────────────────────────

st.markdown("---")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Sections Analysed", len(results.section_impacts))
with col2:
    st.metric("Token Reduction", f"{results.lean_token_reduction:.1%}")
with col3:
    st.metric("Quality Retention", f"{results.lean_quality_retention:.1%}")
with col4:
    st.metric("Experiment Cost", f"${results.total_cost:.4f}")

# Classification breakdown
cls_counts: dict[str, int] = {"essential": 0, "moderate": 0, "removable": 0, "harmful": 0}
for imp in results.section_impacts:
    if imp.classification in cls_counts:
        cls_counts[imp.classification] += 1

badges_html = " &nbsp; ".join(
    f'<span style="font-size:0.78rem;">'
    + _classification_badge(cls)
    + f' <span style="color:#6b7280;">{count}</span></span>'
    for cls, count in cls_counts.items()
    if count > 0
)
st.markdown(
    f'<div style="margin-top:0.5rem; margin-bottom:1rem;">{badges_html}</div>',
    unsafe_allow_html=True,
)

# Navigation shortcuts
nav_col1, nav_col2, nav_col3 = st.columns(3)
with nav_col1:
    if st.button("🥗 Context Diet Plan →", use_container_width=True):
        st.switch_page("pages/4_diet_plan.py")
with nav_col2:
    if st.button("📥 Download Report →", use_container_width=True):
        st.switch_page("pages/5_report.py")
with nav_col3:
    if st.button("← Run New Experiment", use_container_width=True):
        st.switch_page("pages/1_upload.py")

st.markdown("---")

# ── Section Impact Chart (full width) ────────────────────────────────────────

_chart_container("📈 Section Impact Ranking")
st.caption(
    "Sections sorted by average quality delta when removed. "
    "Green = high positive impact (essential); coral = negative (harmful)."
)
fig_impact = render_impact_chart(results)
st.plotly_chart(fig_impact, use_container_width=True)

# ── Heatmap + Tier Radar (two columns) ───────────────────────────────────────

col_heat, col_radar = st.columns(2)

with col_heat:
    _chart_container("🗺️ Quality Heatmap")
    st.caption("Section × reasoning tier quality delta. Red = high impact, Blue = low/negative.")
    fig_heat = render_heatmap(results)
    st.plotly_chart(fig_heat, use_container_width=True)

with col_radar:
    _chart_container("🎯 Tier Sensitivity Radar")
    st.caption("Top 8 sections by tier sensitivity — shows quality delta across reasoning tiers.")
    fig_radar = render_tier_radar(results)
    st.plotly_chart(fig_radar, use_container_width=True)

# ── Pareto Frontier (full width) ──────────────────────────────────────────────

_chart_container("🏆 Pareto Frontier")
st.caption("Quality vs. token count tradeoff. Points on the frontier are non-dominated configurations.")
fig_pareto = render_pareto_chart(results)
st.plotly_chart(fig_pareto, use_container_width=True)

# ── Section Detail Table ──────────────────────────────────────────────────────

st.markdown("---")
_chart_container("📋 Section Detail Table")

if results.section_impacts:
    rows = []
    for imp in results.section_impacts:
        rows.append({
            "Label":           imp.label,
            "Type":            imp.section_type,
            "Tokens":          imp.token_count,
            "Avg Δ":           round(imp.avg_quality_delta, 3),
            "Classification":  imp.classification,
            "Tier Sensitivity": round(imp.tier_sensitivity, 4),
            "Quality/Token":   round(imp.quality_per_token, 6),
        })

    df = pd.DataFrame(rows)
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Label":           st.column_config.TextColumn("Label", width="medium"),
            "Type":            st.column_config.TextColumn("Type", width="small"),
            "Tokens":          st.column_config.NumberColumn("Tokens", format="%d"),
            "Avg Δ":           st.column_config.NumberColumn("Avg Δ", format="%.3f"),
            "Classification":  st.column_config.TextColumn("Classification", width="small"),
            "Tier Sensitivity": st.column_config.NumberColumn("Tier Sensitivity", format="%.4f"),
            "Quality/Token":   st.column_config.NumberColumn("Quality/Token", format="%.6f"),
        },
    )
else:
    st.info("No section impact data available.")

# ── Redundancy Clusters ───────────────────────────────────────────────────────

st.markdown("---")

with st.expander("🔁 Redundancy Clusters", expanded=False):
    if results.redundancy_clusters:
        st.markdown(
            f'<p style="color:#6b7280; font-size:0.85rem; margin-bottom:0.75rem;">'
            f"Detected <strong>{len(results.redundancy_clusters)}</strong> redundant "
            f"section pair(s) above the similarity threshold.</p>",
            unsafe_allow_html=True,
        )
        for id1, id2, sim in results.redundancy_clusters:
            sim_color = "#dc2626" if sim >= 0.9 else "#d97706" if sim >= 0.8 else "#059669"
            st.markdown(
                f'<div style="display:flex; align-items:center; gap:10px; '
                f"padding:0.5rem 0.75rem; margin-bottom:6px; background:#f8f9fc; "
                f'border-radius:8px; border-left:3px solid {sim_color};">'
                f'<code style="font-size:0.8rem;">{id1}</code>'
                f'<span style="color:#9ca3af;">↔</span>'
                f'<code style="font-size:0.8rem;">{id2}</code>'
                f'<span style="margin-left:auto; font-size:0.78rem; '
                f'font-weight:600; color:{sim_color};">sim: {sim:.3f}</span>'
                f"</div>",
                unsafe_allow_html=True,
            )
    else:
        st.info("No redundancy clusters detected above the configured threshold.")

# ── Ordering Recommendations ──────────────────────────────────────────────────

with st.expander("↕️ Ordering Recommendations", expanded=False):
    if results.ordering_recommendations:
        st.markdown(
            f'<p style="color:#6b7280; font-size:0.85rem; margin-bottom:0.75rem;">'
            f"Ordering experiments tested {len(results.ordering_recommendations)} "
            f"section(s) at 3 positions (start, middle, end).</p>",
            unsafe_allow_html=True,
        )
        for rec in results.ordering_recommendations:
            label       = rec.get("label", rec.get("section_id", "?"))
            best_pos    = rec.get("best_position", "N/A")
            quality_gain = rec.get("quality_gain", 0.0)
            deltas      = rec.get("quality_deltas", {})
            delta_parts = " | ".join(
                f"{pos}: {d:+.2f}" for pos, d in deltas.items()
            )
            gain_color = "#059669" if quality_gain > 0 else "#6b7280"
            st.markdown(
                f'<div style="padding:0.6rem 0.85rem; margin-bottom:6px; '
                f'background:#f8f9fc; border-radius:8px; '
                f'border-left:3px solid #c4b5fd;">'
                f'<span style="font-weight:600; color:#374151;">{label}</span>'
                f'<span style="font-size:0.78rem; color:#6b7280; margin-left:8px;">'
                f"Best position: <strong>{best_pos}</strong></span>"
                f'<span style="font-size:0.78rem; color:{gain_color}; margin-left:8px;">'
                f"Gain: {quality_gain:+.3f}</span>"
                f'<div style="font-size:0.75rem; color:#9ca3af; margin-top:3px;">'
                f"{delta_parts}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
    else:
        st.info("No ordering experiments were run (Full mode only).")

# ── Lean Configuration ────────────────────────────────────────────────────────

with st.expander("✂️ Lean Configuration", expanded=False):
    lean_ids = set(results.lean_configuration)
    all_ids  = {imp.section_id for imp in results.section_impacts}
    removed  = all_ids - lean_ids

    col_keep, col_remove = st.columns(2)
    with col_keep:
        st.markdown(
            f'<p style="font-weight:600; color:#059669; font-size:0.85rem;">'
            f"✅ Retained ({len(lean_ids)})</p>",
            unsafe_allow_html=True,
        )
        for sid in sorted(lean_ids):
            st.markdown(
                f'<div style="font-size:0.8rem; color:#374151; '
                f"padding:3px 8px; background:#ecfdf5; border-radius:6px; "
                f'margin-bottom:3px;"><code>{sid}</code></div>',
                unsafe_allow_html=True,
            )
    with col_remove:
        st.markdown(
            f'<p style="font-weight:600; color:#dc2626; font-size:0.85rem;">'
            f"🗑 Removed ({len(removed)})</p>",
            unsafe_allow_html=True,
        )
        for sid in sorted(removed):
            st.markdown(
                f'<div style="font-size:0.8rem; color:#374151; '
                f"padding:3px 8px; background:#fef2f2; border-radius:6px; "
                f'margin-bottom:3px;"><code>{sid}</code></div>',
                unsafe_allow_html=True,
            )
