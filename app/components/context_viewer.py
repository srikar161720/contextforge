"""
app/components/context_viewer.py — Context section list visualization.

Renders a styled section list directly into Streamlit showing each section's
label, type badge, token count, classification badge (if impacts provided),
and a collapsible content preview.
"""

from __future__ import annotations

import streamlit as st

from core.models import AblationResults, ContextPayload, SectionImpact, SectionType

# ── Classification badge palette (pastel) ────────────────────────────────────
_CLASSIFICATION_STYLES: dict[str, tuple[str, str]] = {
    "essential": ("#fca5a5", "#fef2f2"),   # (text_color, bg_color)
    "moderate":  ("#d97706", "#fff7ed"),
    "removable": ("#059669", "#ecfdf5"),
    "harmful":   ("#2563eb", "#eff6ff"),
}

# ── Section type badge palette (pastel) ──────────────────────────────────────
_TYPE_COLORS: dict[str, str] = {
    SectionType.SYSTEM_PROMPT.value:     "#c4b5fd",   # lavender
    SectionType.FEW_SHOT_EXAMPLE.value:  "#a5f3fc",   # sky
    SectionType.RAG_DOCUMENT.value:      "#bef264",   # lime
    SectionType.CONVERSATION_TURN.value: "#fde68a",   # amber
    SectionType.TOOL_DEFINITION.value:   "#f0abfc",   # pink
    SectionType.CUSTOM.value:            "#d4d4d8",   # neutral grey
}

_TYPE_LABELS: dict[str, str] = {
    SectionType.SYSTEM_PROMPT.value:     "System Prompt",
    SectionType.FEW_SHOT_EXAMPLE.value:  "Few-Shot",
    SectionType.RAG_DOCUMENT.value:      "RAG Doc",
    SectionType.CONVERSATION_TURN.value: "Conv. Turn",
    SectionType.TOOL_DEFINITION.value:   "Tool Def.",
    SectionType.CUSTOM.value:            "Custom",
}


def render_context_viewer(
    payload: ContextPayload,
    impacts: list[SectionImpact] | None = None,
) -> None:
    """Render a styled section list into the current Streamlit container.

    Args:
        payload: The parsed ContextPayload to display.
        impacts: Optional list of SectionImpact objects. When provided,
                 each section row also shows its classification badge.
    """
    # Build impact lookup for O(1) access
    impact_map: dict[str, SectionImpact] = (
        {imp.section_id: imp for imp in impacts} if impacts else {}
    )

    # Total tokens for progress bar sizing
    max_tokens = max((s.token_count for s in payload.sections), default=1)

    for section in payload.sections:
        imp = impact_map.get(section.id)
        _render_section_row(section, imp, max_tokens)


# ── Private helpers ───────────────────────────────────────────────────────────


def _render_section_row(
    section,
    imp: SectionImpact | None,
    max_tokens: int,
) -> None:
    """Render a single section as a styled expander row."""
    type_color = _TYPE_COLORS.get(section.section_type.value, "#d4d4d8")
    type_label = _TYPE_LABELS.get(section.section_type.value, section.section_type.value)

    # Build classification badge HTML (only if impact data exists)
    classification_html = ""
    if imp is not None:
        cls = imp.classification
        txt_color, bg_color = _CLASSIFICATION_STYLES.get(
            cls, ("#374151", "#f3f4f6")
        )
        classification_html = (
            f'<span style="'
            f"background:{bg_color}; color:{txt_color}; font-size:0.7rem; "
            f"font-weight:600; padding:2px 10px; border-radius:16px; "
            f'margin-left:6px;">{cls.capitalize()}</span>'
        )

    # Token bar (proportional width, capped to avoid overflow)
    token_pct = min(100, int(section.token_count / max(max_tokens, 1) * 100))
    delta_html = ""
    if imp is not None:
        sign = "+" if imp.avg_quality_delta >= 0 else ""
        delta_html = (
            f'<span style="font-size:0.75rem; color:#6b7280; margin-left:8px;">'
            f"Δ {sign}{imp.avg_quality_delta:.2f}"
            f"</span>"
        )

    header_html = (
        f'<div style="display:flex; align-items:center; flex-wrap:wrap; gap:4px;">'
        f'<span style="font-weight:600; color:#1f2937; font-size:0.9rem;">'
        f"{section.label}</span>"
        f'<span style="background:{type_color}; color:#374151; font-size:0.68rem; '
        f'font-weight:600; padding:2px 8px; border-radius:16px;">'
        f"{type_label}</span>"
        f"{classification_html}"
        f'<span style="font-size:0.75rem; color:#6b7280; margin-left:auto;">'
        f"{section.token_count:,} tokens</span>"
        f"{delta_html}"
        f"</div>"
        f'<div style="margin-top:4px; height:4px; border-radius:6px; '
        f'background:#e5e7eb; overflow:hidden;">'
        f'<div style="width:{token_pct}%; height:100%; border-radius:6px; '
        f'background:{type_color};"></div>'
        f"</div>"
    )

    with st.expander(f"{section.label} — {section.token_count:,} tokens", expanded=False):
        st.markdown(header_html, unsafe_allow_html=True)
        st.markdown("---")
        # Truncate very long content in the preview
        preview = section.content[:600]
        if len(section.content) > 600:
            preview += f"\n\n*… {len(section.content) - 600:,} more characters*"
        st.markdown(
            f'<div style="font-size:0.82rem; color:#4b5563; '
            f'font-family:monospace; white-space:pre-wrap;">{preview}</div>',
            unsafe_allow_html=True,
        )
        if section.metadata:
            with st.expander("Metadata", expanded=False):
                st.json(section.metadata)
