"""
app/components/pareto_chart.py — Pareto frontier quality-vs-tokens scatter chart.

Shows all computed configurations as scatter points, with the Pareto frontier
highlighted as a connecting line. Interactive hover shows configuration details.

Adapted from core/report_generator._build_pareto_chart() with a pastel light theme.
"""

from __future__ import annotations

import plotly.graph_objects as go

from core.models import AblationResults

# ── Pastel light theme constants ──────────────────────────────────────────────
_PLOT_BG      = "#f8f9fc"
_PAPER_BG     = "#ffffff"
_GRID_COLOR   = "#e5e7eb"
_FONT_COLOR   = "#374151"
_POINT_COLOR  = "#7c6fea"   # soft purple — primary accent
_LINE_COLOR   = "#c4b5fd"   # lavender — frontier line


def render_pareto_chart(results: AblationResults) -> go.Figure:
    """Build a quality vs. token-count scatter plot with the Pareto frontier.

    Args:
        results: Fully populated AblationResults from the ablation sweep.

    Returns:
        Plotly Figure ready for st.plotly_chart().
    """
    configs = results.pareto_configurations
    if not configs:
        return _empty_figure("No Pareto configurations available")

    tokens  = [c["tokens"]  for c in configs]
    quality = [c["quality"] for c in configs]
    n_secs  = [len(c.get("section_ids", [])) for c in configs]
    hover   = [
        f"Sections: {ns}<br>Tokens: {t:,}<br>Quality: {q:.2f}"
        for ns, t, q in zip(n_secs, tokens, quality)
    ]

    # Sort Pareto frontier by tokens for the connecting line
    sorted_pairs = sorted(zip(tokens, quality), key=lambda p: p[0])
    px = [p[0] for p in sorted_pairs]
    py = [p[1] for p in sorted_pairs]

    fig = go.Figure()

    # Pareto frontier connecting line
    fig.add_trace(go.Scatter(
        x=px,
        y=py,
        mode="lines",
        line={"color": _LINE_COLOR, "width": 2, "dash": "dot"},
        name="Pareto frontier",
        showlegend=True,
    ))

    # Configuration points
    fig.add_trace(go.Scatter(
        x=tokens,
        y=quality,
        mode="markers",
        marker={
            "size": 11,
            "color": _POINT_COLOR,
            "opacity": 0.85,
            "line": {"width": 1.5, "color": "#ffffff"},
        },
        text=hover,
        hovertemplate="%{text}<extra></extra>",
        name="Configurations",
        showlegend=True,
    ))

    fig.update_layout(
        paper_bgcolor=_PAPER_BG,
        plot_bgcolor=_PLOT_BG,
        font={"color": _FONT_COLOR, "family": "Inter, sans-serif"},
        height=400,
        margin={"l": 70, "r": 60, "t": 20, "b": 60},
        xaxis={
            "title": {"text": "Token Count", "font": {"color": _FONT_COLOR}},
            "tickfont": {"color": _FONT_COLOR},
            "gridcolor": _GRID_COLOR,
        },
        yaxis={
            "title": {"text": "Quality Score", "font": {"color": _FONT_COLOR}},
            "tickfont": {"color": _FONT_COLOR},
            "gridcolor": _GRID_COLOR,
        },
        legend={
            "font": {"color": _FONT_COLOR},
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
