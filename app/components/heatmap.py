"""
app/components/heatmap.py — Section × reasoning-tier quality delta heatmap.

Builds an interactive Plotly heatmap for the Streamlit results dashboard.
Each cell shows the avg_quality_delta for a (section, tier) pair.
Positive (warm) = section helped quality; negative (cool) = section hurt.

Adapted from core/report_generator._build_heatmap() with a pastel light theme.
"""

from __future__ import annotations

import plotly.graph_objects as go

from core.models import AblationResults

# ── Pastel light theme constants ──────────────────────────────────────────────
_PLOT_BG    = "#f8f9fc"
_PAPER_BG   = "#ffffff"
_GRID_COLOR = "#e5e7eb"
_FONT_COLOR = "#374151"

_TIER_ORDER = ["disabled", "low", "medium", "high"]


def render_heatmap(results: AblationResults) -> go.Figure:
    """Build a section × reasoning-tier quality delta heatmap.

    Args:
        results: Fully populated AblationResults from the ablation sweep.

    Returns:
        Plotly Figure ready for st.plotly_chart().
    """
    impacts = results.section_impacts
    if not impacts:
        return _empty_figure("No section impact data available")

    # Collect all tiers present across sections, in canonical order
    tiers = sorted(
        {tier for imp in impacts for tier in imp.quality_delta_by_tier},
        key=lambda t: _TIER_ORDER.index(t) if t in _TIER_ORDER else 99,
    )

    labels   = [imp.label for imp in impacts]
    z_matrix = [
        [imp.quality_delta_by_tier.get(tier, 0.0) for tier in tiers]
        for imp in impacts
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=z_matrix,
            x=[t.capitalize() for t in tiers],
            y=labels,
            colorscale="RdBu",
            zmid=0,
            colorbar={
                "title": {"text": "Quality Δ", "font": {"color": _FONT_COLOR}},
                "tickfont": {"color": _FONT_COLOR},
            },
            hoverongaps=False,
            hovertemplate="<b>%{y}</b><br>Tier: %{x}<br>Δ: %{z:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        paper_bgcolor=_PAPER_BG,
        plot_bgcolor=_PLOT_BG,
        font={"color": _FONT_COLOR, "family": "Inter, sans-serif"},
        height=max(320, 30 * len(impacts)),
        margin={"l": 180, "r": 60, "t": 20, "b": 60},
        xaxis={
            "title": {"text": "Reasoning Tier", "font": {"color": _FONT_COLOR}},
            "tickfont": {"color": _FONT_COLOR},
            "gridcolor": _GRID_COLOR,
        },
        yaxis={
            "tickfont": {"color": _FONT_COLOR},
            "autorange": "reversed",
            "gridcolor": _GRID_COLOR,
        },
    )
    return fig


# ── Private helpers ───────────────────────────────────────────────────────────


def _empty_figure(message: str) -> go.Figure:
    """Return a blank figure with a centred placeholder message."""
    fig = go.Figure()
    fig.update_layout(
        paper_bgcolor=_PAPER_BG,
        plot_bgcolor=_PLOT_BG,
        font={"color": _FONT_COLOR},
        height=200,
        annotations=[{
            "text": message,
            "xref": "paper", "yref": "paper",
            "x": 0.5, "y": 0.5,
            "showarrow": False,
            "font": {"size": 14, "color": "#9ca3af"},
        }],
        xaxis={"visible": False},
        yaxis={"visible": False},
    )
    return fig
