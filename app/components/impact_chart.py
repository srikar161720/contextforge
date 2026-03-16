"""
app/components/impact_chart.py — Section impact horizontal bar chart.

Ranks all sections by avg_quality_delta (highest to lowest impact).
Color-coded by classification using the pastel design system.

Adapted from core/report_generator._build_impact_chart() with a pastel light theme.
"""

from __future__ import annotations

import plotly.graph_objects as go

from core.models import AblationResults

# ── Pastel light theme constants ──────────────────────────────────────────────
_PLOT_BG    = "#f8f9fc"
_PAPER_BG   = "#ffffff"
_GRID_COLOR = "#e5e7eb"
_FONT_COLOR = "#374151"

# Pastel bar colors keyed by classification thresholds (delta value)
# essential (>=2.0): pastel green, moderate (>=0.5): pastel amber,
# removable (>=0.0): pastel orange, harmful (<0): pastel coral
_COLOR_ESSENTIAL = "#6ee7b7"   # mint green — high positive delta
_COLOR_MODERATE  = "#fde68a"   # pastel amber
_COLOR_REMOVABLE = "#fdba74"   # pastel orange
_COLOR_HARMFUL   = "#fca5a5"   # pastel coral — negative delta


def render_impact_chart(results: AblationResults) -> go.Figure:
    """Build a horizontal bar chart of sections ranked by avg_quality_delta.

    Args:
        results: Fully populated AblationResults from the ablation sweep.

    Returns:
        Plotly Figure ready for st.plotly_chart().
    """
    impacts = results.section_impacts
    if not impacts:
        return _empty_figure("No section impact data available")

    # Sort ascending so largest positive delta appears at the top
    sorted_impacts = sorted(impacts, key=lambda x: x.avg_quality_delta)
    labels  = [imp.label for imp in sorted_impacts]
    deltas  = [imp.avg_quality_delta for imp in sorted_impacts]
    colours = [_bar_color(d) for d in deltas]
    classifications = [imp.classification for imp in sorted_impacts]

    hover_texts = [
        f"<b>{lbl}</b><br>Avg Δ: {d:+.2f}<br>Classification: {cls}"
        for lbl, d, cls in zip(labels, deltas, classifications)
    ]

    fig = go.Figure(
        data=go.Bar(
            x=deltas,
            y=labels,
            orientation="h",
            marker_color=colours,
            marker_line_color="rgba(0,0,0,0.08)",
            marker_line_width=1,
            text=hover_texts,
            hovertemplate="%{text}<extra></extra>",
        )
    )
    fig.update_layout(
        paper_bgcolor=_PAPER_BG,
        plot_bgcolor=_PLOT_BG,
        font={"color": _FONT_COLOR, "family": "Inter, sans-serif"},
        height=max(320, 28 * len(impacts)),
        margin={"l": 180, "r": 60, "t": 20, "b": 60},
        xaxis={
            "title": {"text": "Average Quality Delta (baseline − ablated)", "font": {"color": _FONT_COLOR}},
            "tickfont": {"color": _FONT_COLOR},
            "gridcolor": _GRID_COLOR,
            "zerolinecolor": "#9ca3af",
            "zerolinewidth": 1.5,
        },
        yaxis={
            "tickfont": {"color": _FONT_COLOR},
            "gridcolor": _GRID_COLOR,
        },
        shapes=[{
            "type": "line",
            "x0": 0, "x1": 0,
            "y0": -0.5, "y1": len(impacts) - 0.5,
            "line": {"color": "#9ca3af", "width": 1, "dash": "dot"},
        }],
    )
    return fig


# ── Private helpers ───────────────────────────────────────────────────────────


def _bar_color(delta: float) -> str:
    """Return the pastel bar color for a given quality delta value."""
    if delta >= 2.0:
        return _COLOR_ESSENTIAL
    if delta >= 0.5:
        return _COLOR_MODERATE
    if delta >= 0.0:
        return _COLOR_REMOVABLE
    return _COLOR_HARMFUL


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
