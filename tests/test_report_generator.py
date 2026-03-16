"""
tests/test_report_generator.py — Unit tests for core/report_generator.py.

All Bedrock calls are mocked — no AWS credentials required.
Tests cover HTML output structure, Plotly chart generation, Code Interpreter
narrative parsing, and file saving.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import plotly.graph_objects as go
import pytest

from core.report_generator import (
    _build_heatmap,
    _build_impact_chart,
    _build_pareto_chart,
    _build_tier_radar,
    _generate_code_interpreter_narrative,
    _markdown_to_html,
    generate_report,
    save_report,
)
from core.models import (
    AblationResults,
    ContextPayload,
    ContextSection,
    EvalQuery,
    SectionImpact,
    SectionType,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_section(id_: str, tokens: int = 200) -> ContextSection:
    return ContextSection(
        id=id_,
        label=id_,
        section_type=SectionType.RAG_DOCUMENT,
        content="Content here.",
        token_count=tokens,
    )


def _make_impact(
    section_id: str,
    avg_delta: float,
    token_count: int,
    classification: str,
    tiers: dict | None = None,
) -> SectionImpact:
    if tiers is None:
        tiers = {"disabled": avg_delta, "medium": avg_delta * 0.9}
    return SectionImpact(
        section_id=section_id,
        label=section_id,
        section_type="rag_document",
        token_count=token_count,
        avg_quality_delta=avg_delta,
        quality_delta_by_tier=tiers,
        tier_sensitivity=abs(avg_delta * 0.1),
        classification=classification,
        quality_per_token=avg_delta / max(token_count, 1),
    )


@pytest.fixture()
def minimal_payload() -> ContextPayload:
    return ContextPayload(
        sections=[
            _make_section("sys_001", 500),
            _make_section("faq_001", 1000),
            _make_section("faq_002", 800),
        ],
        evaluation_queries=[EvalQuery(query="Q0")],
        quality_criteria=["relevance", "accuracy", "completeness", "groundedness"],
        total_tokens=2300,
    )


@pytest.fixture()
def full_results() -> AblationResults:
    return AblationResults(
        baseline_scores={"disabled": {"0": {"relevance": {"score": 7, "justification": "ok"},
                                            "accuracy": {"score": 7, "justification": "ok"},
                                            "completeness": {"score": 7, "justification": "ok"},
                                            "groundedness": {"score": 7, "justification": "ok"}}}},
        section_impacts=[
            _make_impact("sys_001", avg_delta=3.5, token_count=500,  classification="essential"),
            _make_impact("faq_001", avg_delta=0.4, token_count=1000, classification="removable"),
            _make_impact("faq_002", avg_delta=0.1, token_count=800,  classification="removable"),
        ],
        lean_configuration=["sys_001"],
        lean_quality_retention=0.97,
        lean_token_reduction=0.78,
        ordering_recommendations=[
            {
                "section_id": "faq_001",
                "label": "faq_001",
                "best_position": "start",
                "quality_deltas": {"start": 0.3, "middle": 0.1, "end": -0.1},
                "quality_gain": 0.4,
            }
        ],
        redundancy_clusters=[("faq_001", "faq_002", 0.85)],
        pareto_configurations=[
            {"section_ids": ["sys_001", "faq_001"], "quality": 6.8, "tokens": 1500, "cost": 0.00045},
            {"section_ids": ["sys_001"],             "quality": 6.5, "tokens": 500,  "cost": 0.00015},
        ],
        total_api_calls=80,
        total_input_tokens=4_000_000,
        total_output_tokens=8_000,
        total_cost=1.24,
    )


# ── Tests: generate_report (HTML output) ──────────────────────────────────────


def test_generate_report_returns_html_string(full_results, minimal_payload):
    """generate_report must return a non-empty HTML string."""
    html = generate_report(
        results=full_results,
        payload=minimal_payload,
        diet_plan="# Plan\n\nRemove faq_001.",
        client=None,
    )
    assert isinstance(html, str)
    assert len(html) > 100
    assert "<html" in html.lower()


def test_generate_report_contains_plotly_script(full_results, minimal_payload):
    """The HTML must include a Plotly CDN script tag."""
    html = generate_report(
        results=full_results,
        payload=minimal_payload,
        diet_plan="# Plan",
        client=None,
    )
    assert "cdn.plot.ly" in html or "plotly" in html.lower()


def test_generate_report_contains_diet_plan(full_results, minimal_payload):
    """Diet plan content must be present in the rendered HTML."""
    diet_plan = "# My Diet Plan\n\nRemove the faq_001 section."
    html = generate_report(
        results=full_results,
        payload=minimal_payload,
        diet_plan=diet_plan,
        client=None,
    )
    assert "My Diet Plan" in html
    assert "faq_001" in html


def test_generate_report_contains_summary_stats(full_results, minimal_payload):
    """HTML must include token counts and quality retention from the results."""
    html = generate_report(
        results=full_results,
        payload=minimal_payload,
        diet_plan="# Plan",
        client=None,
    )
    # total_tokens = 2300
    assert "2,300" in html
    # lean_quality_retention = 97.0%
    assert "97.0" in html
    # lean_token_reduction = 78.0%
    assert "78.0" in html


def test_generate_report_with_empty_ordering(full_results, minimal_payload):
    """Report must render without error when ordering_recommendations is empty."""
    full_results.ordering_recommendations = []
    html = generate_report(
        results=full_results,
        payload=minimal_payload,
        diet_plan="# Plan",
        client=None,
    )
    assert isinstance(html, str)
    assert "Ordering experiment" in html or "ordering" in html.lower()


def test_generate_report_with_empty_redundancy(full_results, minimal_payload):
    """Report must render without error when redundancy_clusters is empty."""
    full_results.redundancy_clusters = []
    html = generate_report(
        results=full_results,
        payload=minimal_payload,
        diet_plan="# Plan",
        client=None,
    )
    assert isinstance(html, str)
    assert "No redundancy" in html


# ── Tests: save_report ─────────────────────────────────────────────────────────


def test_save_report_writes_file(full_results, minimal_payload):
    """save_report must write the HTML to the specified path."""
    html = generate_report(
        results=full_results,
        payload=minimal_payload,
        diet_plan="# Plan",
        client=None,
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "test_report.html"
        result_path = save_report(html, out_path)
        assert result_path.exists()
        content = result_path.read_text(encoding="utf-8")
        assert "<html" in content.lower()


def test_save_report_creates_parent_dirs():
    """save_report must create parent directories if they don't exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        nested = Path(tmpdir) / "subdir" / "deep" / "report.html"
        save_report("<html><body>Test</body></html>", nested)
        assert nested.exists()


# ── Tests: Plotly chart builders ──────────────────────────────────────────────


def test_build_heatmap_returns_figure(full_results):
    """_build_heatmap must return a Plotly Figure."""
    fig = _build_heatmap(full_results)
    assert isinstance(fig, go.Figure)


def test_build_heatmap_empty_impacts():
    """_build_heatmap must handle empty section_impacts gracefully."""
    results = AblationResults(
        baseline_scores={}, section_impacts=[], lean_configuration=[],
        lean_quality_retention=1.0, lean_token_reduction=0.0,
        ordering_recommendations=[], redundancy_clusters=[],
        pareto_configurations=[], total_api_calls=0,
        total_input_tokens=0, total_output_tokens=0, total_cost=0.0,
    )
    fig = _build_heatmap(results)
    assert isinstance(fig, go.Figure)


def test_build_impact_chart_returns_figure(full_results):
    """_build_impact_chart must return a Plotly Figure."""
    fig = _build_impact_chart(full_results)
    assert isinstance(fig, go.Figure)


def test_build_pareto_chart_returns_figure(full_results):
    """_build_pareto_chart must return a Plotly Figure."""
    fig = _build_pareto_chart(full_results)
    assert isinstance(fig, go.Figure)


def test_build_pareto_chart_empty_configs():
    """_build_pareto_chart must handle empty pareto_configurations gracefully."""
    results = AblationResults(
        baseline_scores={}, section_impacts=[], lean_configuration=[],
        lean_quality_retention=1.0, lean_token_reduction=0.0,
        ordering_recommendations=[], redundancy_clusters=[],
        pareto_configurations=[], total_api_calls=0,
        total_input_tokens=0, total_output_tokens=0, total_cost=0.0,
    )
    fig = _build_pareto_chart(results)
    assert isinstance(fig, go.Figure)


def test_build_tier_radar_returns_figure(full_results):
    """_build_tier_radar must return a Plotly Figure."""
    fig = _build_tier_radar(full_results)
    assert isinstance(fig, go.Figure)


# ── Tests: Code Interpreter narrative ─────────────────────────────────────────


def test_code_interpreter_narrative_with_tool_blocks(full_results):
    """With toolUse/toolResult blocks, the narrative HTML must include both."""
    client = MagicMock()
    client.invoke_raw.return_value = (
        [
            {"text": "The analysis confirms a 78% token reduction."},
            {"toolUse": {"name": "nova_code_interpreter",
                         "input": {"snippet": "print(0.78 * 100)"}}},
            {"toolResult": {"content": [{"json": {"stdOut": "78.0\n", "stdErr": "", "exitCode": 0}}]}},
            {"text": "The lean configuration retains 97% of quality."},
        ],
        {"input_tokens": 500, "output_tokens": 300, "total_tokens": 800},
    )

    html = _generate_code_interpreter_narrative(full_results, client)

    assert "78" in html
    assert "97" in html


def test_code_interpreter_narrative_text_only(full_results):
    """With text-only blocks (no Code Interpreter activation), return the text."""
    client = MagicMock()
    client.invoke_raw.return_value = (
        [{"text": "Here is my analysis of the ablation results."}],
        {"input_tokens": 200, "output_tokens": 100, "total_tokens": 300},
    )

    html = _generate_code_interpreter_narrative(full_results, client)

    assert "analysis" in html


def test_code_interpreter_narrative_error_fallback(full_results):
    """On API error, _generate_code_interpreter_narrative must return a fallback HTML string."""
    client = MagicMock()
    client.invoke_raw.side_effect = RuntimeError("Simulated API error")

    html = _generate_code_interpreter_narrative(full_results, client)

    assert isinstance(html, str)
    assert len(html) > 0


# ── Tests: _markdown_to_html ──────────────────────────────────────────────────


def test_markdown_to_html_headings():
    """h1–h3 headings must be converted to <h1>–<h3> tags."""
    md = "# Title\n## Subtitle\n### Section"
    html = _markdown_to_html(md)
    assert "<h1>Title</h1>" in html
    assert "<h2>Subtitle</h2>" in html
    assert "<h3>Section</h3>" in html


def test_markdown_to_html_bold_italic():
    """Bold (**text**) and italic (*text*) must be converted."""
    md = "**bold** and *italic*"
    html = _markdown_to_html(md)
    assert "<strong>bold</strong>" in html
    assert "<em>italic</em>" in html


def test_markdown_to_html_unordered_list():
    """Unordered lists must be wrapped in <ul><li> tags."""
    md = "- item one\n- item two\n- item three"
    html = _markdown_to_html(md)
    assert "<ul>" in html
    assert "<li>item one</li>" in html


def test_markdown_to_html_empty_string():
    """Empty input must return an empty string without raising."""
    assert _markdown_to_html("") == ""
