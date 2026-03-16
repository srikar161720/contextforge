"""
app/components/tier_radar.py — Reasoning tier sensitivity radar chart.

Shows the top 8 sections by tier_sensitivity as overlapping radar/spider traces.
Each section gets one trace, with the reasoning tiers as axes.

Adapted from core/report_generator._build_tier_radar() with a pastel light theme.
"""

from __future__ import annotations

import plotly.graph_objects as go

from core.models import AblationResults

# ── Pastel light theme constants ──────────────────────────────────────────────
_PAPER_BG   = "#ffffff"
_FONT_COLOR = "#374151"
_GRID_COLOR = "#e5e7eb"

# Pastel trace colours (cycles through for up to 8 sections)
_TRACE_COLORS = [
    "#7c6fea",  # soft purple
    "#60a5fa",  # pastel blue
    "#6ee7b7",  # mint green
    "#fdba74",  # pastel orange
    "#f0abfc",  # pink
    "#fde68a",  # pastel amber
    "#a5f3fc",  # sky
    "#fca5a5",  # pastel coral
]


def render_tier_radar(results: AblationResults) -> go.Figure:
    """Build a tier-sensitivity radar chart for the top 8 sections.

    Args:
        results: Fully populated AblationResults from the ablation sweep.

    Returns:
        Plotly Figure ready for st.plotly_chart().
    """
    impacts = results.section_impacts
    if not impacts:
        return _empty_figure("No section impact data available")

    # Top 8 by tier_sensitivity for a readable radar
    top = sorted(impacts, key=lambda x: x.tier_sensitivity, reverse=True)[:8]

    # Check any section has multiple tiers
    if not any(len(imp.quality_delta_by_tier) > 1 for imp in top):
        return _empty_figure("Radar chart requires multiple reasoning tiers")

    fig = go.Figure()
    for i, imp in enumerate(top):
        tiers = list(imp.quality_delta_by_tier.keys())
        vals  = [imp.quality_delta_by_tier[t] for t in tiers]
        # Close the polygon
        tiers_closed = tiers + [tiers[0]]
        vals_closed  = vals + [vals[0]]

        color = _TRACE_COLORS[i % len(_TRACE_COLORS)]
        fig.add_trace(go.Scatterpolar(
            r=vals_closed,
            theta=[t.capitalize() for t in tiers_closed],
            fill="toself",
            name=imp.label,
            opacity=0.60,
            line={"color": color, "width": 1.5},
            fillcolor=color,
        ))

    fig.update_layout(
        paper_bgcolor=_PAPER_BG,
        font={"color": _FONT_COLOR, "family": "Inter, sans-serif"},
        height=420,
        margin={"l": 50, "r": 50, "t": 40, "b": 40},
        polar={
            "bgcolor": "#f8f9fc",
            "radialaxis": {
                "visible": True,
                "title": {"text": "Quality Δ", "font": {"color": _FONT_COLOR}},
                "tickfont": {"color": _FONT_COLOR},
                "gridcolor": _GRID_COLOR,
                "linecolor": _GRID_COLOR,
            },
            "angularaxis": {
                "tickfont": {"color": _FONT_COLOR},
                "gridcolor": _GRID_COLOR,
                "linecolor": _GRID_COLOR,
            },
        },
        legend={
            "orientation": "v",
            "x": 1.05,
            "font": {"color": _FONT_COLOR, "size": 11},
            "bgcolor": "rgba(255,255,255,0.9)",
            "bordercolor": "#e5e7eb",
            "borderwidth": 1,
        },
    )
    return fig


# ── Private helpers ───────────────────────────────────────────────────────────


def _empty_figure(message: str) -> go.Figure:
    """Return a blank figure with a centred placeholder message."""
    fig = go.Figure()
    fig.update_layout(
        paper_bgcolor=_PAPER_BG,
        font={"color": _FONT_COLOR},
        height=200,
        annotations=[{
            "text": message,
            "xref": "paper", "yref": "paper",
            "x": 0.5, "y": 0.5,
            "showarrow": False,
            "font": {"size": 14, "color": "#9ca3af"},
        }],
    )
    return fig
