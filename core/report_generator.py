"""
core/report_generator.py — HTML report generation (Phase 5).

Produces a self-contained, downloadable HTML report from ablation results.
The report embeds four interactive Plotly charts, a Jinja2-rendered template,
the Context Diet Plan, and an optional Code Interpreter narrative.

Pipeline:
  1. Build four Plotly figures (heatmap, impact waterfall, Pareto, tier radar).
  2. Optionally call Nova Code Interpreter for a statistical verification narrative.
  3. Convert diet plan markdown to HTML.
  4. Render the Jinja2 template with all data and chart fragments.
  5. Return a complete HTML string (and optionally write to disk).

Critical rules:
  - Code Interpreter is used ONLY for report flair — not for core statistics.
  - All Bedrock calls go through BedrockClient.invoke_raw() (Code Interpreter
    responses return raw content blocks, not plain text).
  - Failures in Code Interpreter gracefully fall back to a local narrative.
  - plotly.io.to_html() is called with full_html=False, include_plotlyjs=False
    so the CDN <script> tag in the template loads Plotly once.

See context/api-reference.md for Code Interpreter response block parsing.
See context/architecture.md for the report generation role in the pipeline.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from pathlib import Path

import plotly.graph_objects as go
import plotly.io as pio
from jinja2 import Environment, FileSystemLoader, select_autoescape

from core.models import AblationResults, ContextPayload, SectionImpact
from infra.bedrock_client import BedrockClient

logger = logging.getLogger(__name__)

# Template directory relative to this file
_TEMPLATE_DIR = Path(__file__).parent.parent / "data" / "templates"
_TEMPLATE_FILE = "report_template.html"

# Pricing for cost estimates
_INPUT_PRICE_PER_M  = 0.30
_OUTPUT_PRICE_PER_M = 2.50

# Plotly chart colour palette (matches dark theme)
_PLOTLY_THEME = "plotly_dark"


# ── Public interface ──────────────────────────────────────────────────────────


def generate_report(
    results:        AblationResults,
    payload:        ContextPayload,
    diet_plan:      str,
    client:         BedrockClient | None = None,
    payload_name:   str = "context_payload",
    experiment_mode: str = "demo",
) -> str:
    """Generate a self-contained HTML ablation report.

    Args:
        results:          Fully populated AblationResults from the ablation sweep.
        payload:          Original ContextPayload (for section token data).
        diet_plan:        Markdown-formatted Context Diet Plan from diet_planner.py.
        client:           Optional BedrockClient for Code Interpreter narrative.
                          Pass None to skip the Code Interpreter section.
        payload_name:     Human-readable payload name shown in the report header.
        experiment_mode:  "demo" | "quick" | "full" — shown as a badge in the header.

    Returns:
        Complete HTML string for the report.
    """
    # 1. Build Plotly charts
    heatmap_fig     = _build_heatmap(results)
    impact_fig      = _build_impact_chart(results)
    pareto_fig      = _build_pareto_chart(results)
    tier_radar_fig  = _build_tier_radar(results)

    heatmap_html     = _fig_to_html(heatmap_fig)
    impact_html      = _fig_to_html(impact_fig)
    pareto_html      = _fig_to_html(pareto_fig)
    tier_radar_html  = _fig_to_html(tier_radar_fig)

    # 2. Optional Code Interpreter narrative
    ci_narrative_html = ""
    if client is not None:
        ci_narrative_html = _generate_code_interpreter_narrative(results, client)

    # 3. Convert diet plan markdown to basic HTML
    diet_plan_html = _markdown_to_html(diet_plan)

    # 4. Compute summary values for the template
    total_tokens = payload.total_tokens
    lean_section_ids = set(results.lean_configuration)
    lean_tokens = sum(
        s.token_count for s in payload.sections if s.id in lean_section_ids
    )

    section_counts = {c: 0 for c in ("essential", "moderate", "removable", "harmful")}
    for imp in results.section_impacts:
        if imp.classification in section_counts:
            section_counts[imp.classification] += 1

    template_vars = {
        # Header
        "generated_at":      datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "payload_name":      payload_name,
        "experiment_mode":   experiment_mode,
        # Summary cards
        "total_tokens":          total_tokens,
        "lean_tokens":           lean_tokens,
        "baseline_cost_per_call": _compute_input_cost(total_tokens),
        "lean_cost_per_call":    _compute_input_cost(lean_tokens),
        "lean_token_reduction":  results.lean_token_reduction,
        "lean_quality_retention": results.lean_quality_retention,
        "section_count":         len(results.section_impacts),
        "removable_count":       section_counts["removable"] + section_counts["harmful"],
        "essential_count":       section_counts["essential"],
        "total_cost":            results.total_cost,
        "total_api_calls":       results.total_api_calls,
        "total_input_tokens":    results.total_input_tokens,
        "total_output_tokens":   results.total_output_tokens,
        # Chart fragments
        "heatmap_html":          heatmap_html,
        "impact_chart_html":     impact_html,
        "pareto_chart_html":     pareto_html,
        "tier_radar_html":       tier_radar_html,
        # Section detail table
        "section_impacts":       results.section_impacts,
        # Redundancy
        "redundancy_clusters":   results.redundancy_clusters,
        # Ordering
        "ordering_recommendations": results.ordering_recommendations,
        # Diet plan
        "diet_plan_html":        diet_plan_html,
        # Code Interpreter narrative
        "ci_narrative_html":     ci_narrative_html,
    }

    # 5. Render Jinja2 template
    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATE_DIR)),
        autoescape=select_autoescape(["html"]),
    )
    # Allow the diet plan and chart HTML to be rendered without escaping
    env.globals["__builtins__"] = {}
    template = env.get_template(_TEMPLATE_FILE)
    # Mark diet_plan_html and chart fragments as safe so Jinja2 does not escape them
    from markupsafe import Markup
    template_vars["diet_plan_html"]     = Markup(diet_plan_html)
    template_vars["ci_narrative_html"]  = Markup(ci_narrative_html)
    template_vars["heatmap_html"]       = Markup(heatmap_html)
    template_vars["impact_chart_html"]  = Markup(impact_html)
    template_vars["pareto_chart_html"]  = Markup(pareto_html)
    template_vars["tier_radar_html"]    = Markup(tier_radar_html)

    return template.render(**template_vars)


def save_report(html: str, output_path: Path) -> Path:
    """Write the HTML report string to disk.

    Args:
        html:        Complete HTML string from generate_report().
        output_path: Destination file path (created if parent dirs exist).

    Returns:
        The resolved output path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    logger.info("Report saved to %s (%d bytes)", output_path, len(html))
    return output_path.resolve()


# ── Plotly chart builders ─────────────────────────────────────────────────────


def _build_heatmap(results: AblationResults) -> go.Figure:
    """Build a section × reasoning-tier quality delta heatmap.

    Each cell shows the avg_quality_delta for that (section, tier) pair.
    Positive (red) = section helped quality; negative (blue) = section hurt.
    """
    impacts = results.section_impacts
    if not impacts:
        return _empty_figure("No section impact data available")

    # Collect all tiers that appear in at least one section
    tiers = sorted(
        {tier for imp in impacts for tier in imp.quality_delta_by_tier},
        key=lambda t: ["disabled", "low", "medium", "high"].index(t)
        if t in ["disabled", "low", "medium", "high"]
        else 99,
    )

    labels  = [imp.label for imp in impacts]
    z_matrix = [
        [imp.quality_delta_by_tier.get(tier, 0.0) for tier in tiers]
        for imp in impacts
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=z_matrix,
            x=tiers,
            y=labels,
            colorscale="RdBu",
            zmid=0,
            colorbar={"title": "Quality Δ"},
            hoverongaps=False,
            hovertemplate="Section: %{y}<br>Tier: %{x}<br>Δ: %{z:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        template=_PLOTLY_THEME,
        height=max(300, 30 * len(impacts)),
        margin={"l": 160, "r": 40, "t": 20, "b": 60},
        xaxis_title="Reasoning Tier",
        yaxis_title="",
        yaxis={"autorange": "reversed"},
    )
    return fig


def _build_impact_chart(results: AblationResults) -> go.Figure:
    """Build a horizontal bar chart of sections ranked by avg_quality_delta."""
    impacts = results.section_impacts
    if not impacts:
        return _empty_figure("No section impact data available")

    sorted_impacts = sorted(impacts, key=lambda x: x.avg_quality_delta)
    labels  = [imp.label for imp in sorted_impacts]
    deltas  = [imp.avg_quality_delta for imp in sorted_impacts]
    colours = [
        "#48bb78" if d >= 2.0 else
        "#63b3ed" if d >= 0.5 else
        "#f6ad55" if d >= 0.0 else
        "#fc8181"
        for d in deltas
    ]

    fig = go.Figure(
        data=go.Bar(
            x=deltas,
            y=labels,
            orientation="h",
            marker_color=colours,
            hovertemplate="<b>%{y}</b><br>Avg Δ: %{x:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        template=_PLOTLY_THEME,
        height=max(300, 28 * len(impacts)),
        margin={"l": 160, "r": 40, "t": 20, "b": 60},
        xaxis_title="Average Quality Delta (baseline − ablated)",
        yaxis_title="",
        shapes=[
            {"type": "line", "x0": 0, "x1": 0,
             "y0": -0.5, "y1": len(impacts) - 0.5,
             "line": {"color": "#718096", "width": 1, "dash": "dot"}},
        ],
    )
    return fig


def _build_pareto_chart(results: AblationResults) -> go.Figure:
    """Build a quality vs. token-count scatter plot with the Pareto frontier."""
    configs = results.pareto_configurations
    if not configs:
        return _empty_figure("No Pareto configurations available")

    tokens   = [c["tokens"]  for c in configs]
    quality  = [c["quality"] for c in configs]
    n_secs   = [len(c.get("section_ids", [])) for c in configs]
    hover    = [
        f"Sections: {ns}<br>Tokens: {t:,}<br>Quality: {q:.2f}"
        for ns, t, q in zip(n_secs, tokens, quality)
    ]

    # Sort Pareto frontier by tokens for the connecting line
    sorted_pairs = sorted(zip(tokens, quality), key=lambda x: x[0])
    px = [p[0] for p in sorted_pairs]
    py = [p[1] for p in sorted_pairs]

    fig = go.Figure()
    # Pareto frontier line
    fig.add_trace(go.Scatter(
        x=px, y=py,
        mode="lines",
        line={"color": "#4299e1", "width": 1.5, "dash": "dot"},
        name="Pareto frontier",
        showlegend=True,
    ))
    # Configuration points
    fig.add_trace(go.Scatter(
        x=tokens, y=quality,
        mode="markers",
        marker={"size": 10, "color": "#63b3ed", "line": {"width": 1, "color": "#2b6cb0"}},
        text=hover,
        hovertemplate="%{text}<extra></extra>",
        name="Configurations",
        showlegend=True,
    ))
    fig.update_layout(
        template=_PLOTLY_THEME,
        height=400,
        margin={"l": 60, "r": 40, "t": 20, "b": 60},
        xaxis_title="Token Count",
        yaxis_title="Quality Score",
    )
    return fig


def _build_tier_radar(results: AblationResults) -> go.Figure:
    """Build a tier-sensitivity radar chart for the top 8 sections by tier_sensitivity."""
    impacts = results.section_impacts
    if not impacts:
        return _empty_figure("No section impact data available")

    # Top 8 by tier_sensitivity for a readable radar
    top = sorted(impacts, key=lambda x: x.tier_sensitivity, reverse=True)[:8]

    fig = go.Figure()
    for imp in top:
        tiers = list(imp.quality_delta_by_tier.keys())
        vals  = [imp.quality_delta_by_tier[t] for t in tiers]
        # Close the polygon
        tiers_closed = tiers + [tiers[0]]
        vals_closed  = vals  + [vals[0]]
        fig.add_trace(go.Scatterpolar(
            r=vals_closed,
            theta=tiers_closed,
            fill="toself",
            name=imp.label,
            opacity=0.65,
        ))

    fig.update_layout(
        template=_PLOTLY_THEME,
        height=450,
        margin={"l": 40, "r": 40, "t": 30, "b": 40},
        polar={
            "radialaxis": {
                "visible": True,
                "title": {"text": "Quality Δ"},
            }
        },
        legend={"orientation": "v", "x": 1.02},
    )
    return fig


# ── Code Interpreter narrative ────────────────────────────────────────────────


def _generate_code_interpreter_narrative(
    results: AblationResults,
    client:  BedrockClient,
) -> str:
    """Generate a statistical verification narrative via Nova Code Interpreter.

    Sends a summary of ablation statistics to Nova and asks it to write and
    execute Python code to verify the key findings, then returns a brief
    analytical narrative.

    Falls back to a locally-generated HTML summary if the API call fails or
    Code Interpreter does not activate.

    Returns:
        HTML string with the narrative (paragraphs + code output if available).
    """
    summary = _build_ci_summary(results)
    prompt = f"""\
You have the following ablation experiment results from a context optimization study.
Please verify the key statistics using Python code and provide a brief analytical
narrative (3-5 sentences) summarizing the most important findings.

{summary}

Write Python code to:
1. Verify the token reduction percentage.
2. Count sections by classification.
3. Compute the potential cost savings at $0.30/1M input tokens.

After running the code, provide your analytical narrative.
"""

    try:
        blocks, usage = client.invoke_raw(
            system=(
                "You are a data analyst verifying ablation experiment statistics. "
                "Use Python code to verify the numbers, then summarize the findings."
            ),
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            reasoning_tier="medium",
            tool_config={"tools": [{"systemTool": {"name": "nova_code_interpreter"}}]},
        )
        logger.info(
            "Code Interpreter narrative generated — input=%d, output=%d",
            usage.get("input_tokens", 0),
            usage.get("output_tokens", 0),
        )
        return _parse_ci_blocks_to_html(blocks)

    except Exception as exc:  # noqa: BLE001
        logger.warning("Code Interpreter narrative failed: %s — using fallback.", exc)
        return _generate_local_narrative_html(results)


def _build_ci_summary(results: AblationResults) -> str:
    """Build a plain-text summary of ablation stats for the CI prompt."""
    section_counts = {"essential": 0, "moderate": 0, "removable": 0, "harmful": 0}
    total_tokens   = sum(imp.token_count for imp in results.section_impacts)
    lean_tokens    = sum(
        imp.token_count
        for imp in results.section_impacts
        if imp.section_id in set(results.lean_configuration)
    )
    for imp in results.section_impacts:
        if imp.classification in section_counts:
            section_counts[imp.classification] += 1

    lines = [
        f"Total sections analysed: {len(results.section_impacts)}",
        f"  - Essential:  {section_counts['essential']}",
        f"  - Moderate:   {section_counts['moderate']}",
        f"  - Removable:  {section_counts['removable']}",
        f"  - Harmful:    {section_counts['harmful']}",
        f"Total context tokens (baseline): {total_tokens:,}",
        f"Lean config tokens:              {lean_tokens:,}",
        f"Token reduction fraction:        {results.lean_token_reduction:.4f}",
        f"Quality retention fraction:      {results.lean_quality_retention:.4f}",
        f"Total API cost for this sweep:   ${results.total_cost:.4f}",
        f"Redundancy pairs detected:       {len(results.redundancy_clusters)}",
    ]
    return "\n".join(lines)


def _parse_ci_blocks_to_html(blocks: list[dict]) -> str:
    """Parse Code Interpreter response blocks into an HTML narrative string."""
    text_parts: list[str] = []
    code_snippets: list[str] = []
    code_outputs: list[str] = []

    for block in blocks:
        if "text" in block:
            text_parts.append(block["text"])
        elif "toolUse" in block:
            code = block["toolUse"].get("input", {}).get("snippet", "")
            if code:
                code_snippets.append(code)
        elif "toolResult" in block:
            content = block["toolResult"].get("content", [])
            for item in content:
                if isinstance(item, dict):
                    stdout = item.get("json", {}).get("stdOut", "")
                    if stdout.strip():
                        code_outputs.append(stdout.strip())

    if not text_parts and not code_outputs:
        return ""

    html_parts: list[str] = []
    for text in text_parts:
        paragraphs = [p.strip() for p in text.strip().split("\n\n") if p.strip()]
        for para in paragraphs:
            html_parts.append(f"<p>{para}</p>")

    if code_outputs:
        for output in code_outputs:
            escaped = output.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            html_parts.append(f'<pre style="margin-top:0.75rem;">{escaped}</pre>')

    return "\n".join(html_parts)


def _generate_local_narrative_html(results: AblationResults) -> str:
    """Fallback: generate a brief narrative from local statistics."""
    n_removable = sum(
        1 for imp in results.section_impacts
        if imp.classification in ("removable", "harmful")
    )
    n_essential = sum(
        1 for imp in results.section_impacts
        if imp.classification == "essential"
    )
    token_pct = results.lean_token_reduction * 100
    quality_pct = results.lean_quality_retention * 100

    return (
        f"<p>Ablation analysis across {len(results.section_impacts)} sections identified "
        f"{n_essential} essential section(s) and {n_removable} removable section(s). "
        f"The lean configuration achieves a {token_pct:.1f}% token reduction while "
        f"retaining {quality_pct:.1f}% of baseline response quality.</p>"
    )


# ── Markdown to HTML ──────────────────────────────────────────────────────────


def _markdown_to_html(text: str) -> str:
    """Convert a Markdown string to basic HTML for embedding in the report.

    Handles headings (h1–h3), bold/italic, code blocks, inline code, unordered
    lists, ordered lists, blockquotes, and paragraphs. Does not require an
    external markdown library.
    """
    if not text:
        return ""

    lines      = text.split("\n")
    html_lines: list[str] = []
    in_pre     = False
    in_ul      = False
    in_ol      = False
    pre_buf:   list[str] = []

    def _close_list():
        nonlocal in_ul, in_ol
        if in_ul:
            html_lines.append("</ul>")
            in_ul = False
        if in_ol:
            html_lines.append("</ol>")
            in_ol = False

    def _inline(s: str) -> str:
        """Apply inline Markdown formatting."""
        # Bold
        s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
        s = re.sub(r"__(.+?)__",     r"<strong>\1</strong>", s)
        # Italic
        s = re.sub(r"\*(.+?)\*", r"<em>\1</em>", s)
        s = re.sub(r"_(.+?)_",   r"<em>\1</em>", s)
        # Inline code
        s = re.sub(r"`(.+?)`", r"<code>\1</code>", s)
        return s

    for line in lines:
        # Code block fence
        if line.startswith("```"):
            if in_pre:
                html_lines.append("<pre>" + "\n".join(pre_buf) + "</pre>")
                pre_buf = []
                in_pre  = False
            else:
                _close_list()
                in_pre = True
            continue
        if in_pre:
            pre_buf.append(line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
            continue

        # Headings
        m = re.match(r"^(#{1,3})\s+(.*)", line)
        if m:
            _close_list()
            level = len(m.group(1))
            html_lines.append(f"<h{level}>{_inline(m.group(2))}</h{level}>")
            continue

        # Blockquote
        if line.startswith("> "):
            _close_list()
            html_lines.append(f"<blockquote>{_inline(line[2:])}</blockquote>")
            continue

        # Horizontal rule
        if re.match(r"^[-*_]{3,}$", line.strip()):
            _close_list()
            html_lines.append("<hr>")
            continue

        # Unordered list
        m = re.match(r"^[-*+]\s+(.*)", line)
        if m:
            if not in_ul:
                _close_list()
                html_lines.append("<ul>")
                in_ul = True
            html_lines.append(f"<li>{_inline(m.group(1))}</li>")
            continue

        # Ordered list
        m = re.match(r"^\d+\.\s+(.*)", line)
        if m:
            if not in_ol:
                _close_list()
                html_lines.append("<ol>")
                in_ol = True
            html_lines.append(f"<li>{_inline(m.group(1))}</li>")
            continue

        # Close any open list on non-list line
        if line.strip():
            _close_list()
            html_lines.append(f"<p>{_inline(line)}</p>")
        else:
            _close_list()

    _close_list()
    if in_pre and pre_buf:
        html_lines.append("<pre>" + "\n".join(pre_buf) + "</pre>")

    return "\n".join(html_lines)


# ── Private helpers ───────────────────────────────────────────────────────────


def _fig_to_html(fig: go.Figure) -> str:
    """Convert a Plotly figure to an HTML div fragment (no full_html, no plotlyjs CDN)."""
    return pio.to_html(fig, full_html=False, include_plotlyjs=False)


def _empty_figure(message: str) -> go.Figure:
    """Return a blank Plotly figure with a centred message."""
    fig = go.Figure()
    fig.update_layout(
        template=_PLOTLY_THEME,
        height=200,
        annotations=[{
            "text": message,
            "xref": "paper", "yref": "paper",
            "x": 0.5, "y": 0.5,
            "showarrow": False,
            "font": {"size": 14, "color": "#718096"},
        }],
        xaxis={"visible": False},
        yaxis={"visible": False},
    )
    return fig


def _compute_input_cost(tokens: int) -> float:
    """Compute USD cost for input tokens only (used for per-call estimates)."""
    return tokens * _INPUT_PRICE_PER_M / 1_000_000
