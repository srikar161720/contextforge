"""
tests/test_diet_planner.py — Unit tests for core/diet_planner.py.

All Bedrock calls are mocked — no AWS credentials required.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from core.diet_planner import (
    _build_diet_plan_prompt,
    _format_ordering_summary,
    _format_pareto_summary,
    _format_redundancy_summary,
    _format_section_table,
    generate_diet_plan,
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


def _make_section(id_: str, tokens: int = 100) -> ContextSection:
    return ContextSection(
        id=id_,
        label=id_,
        section_type=SectionType.RAG_DOCUMENT,
        content="Some content.",
        token_count=tokens,
    )


def _make_impact(
    section_id: str,
    avg_delta: float,
    token_count: int,
    classification: str,
) -> SectionImpact:
    return SectionImpact(
        section_id=section_id,
        label=section_id,
        section_type="rag_document",
        token_count=token_count,
        avg_quality_delta=avg_delta,
        quality_delta_by_tier={"disabled": avg_delta},
        tier_sensitivity=0.0,
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
def minimal_results() -> AblationResults:
    return AblationResults(
        baseline_scores={"disabled": {"0": {"relevance": {"score": 7, "justification": "ok"},
                                            "accuracy": {"score": 7, "justification": "ok"},
                                            "completeness": {"score": 7, "justification": "ok"},
                                            "groundedness": {"score": 7, "justification": "ok"}}}},
        section_impacts=[
            _make_impact("sys_001", avg_delta=3.5, token_count=500,  classification="essential"),
            _make_impact("faq_001", avg_delta=0.2, token_count=1000, classification="removable"),
            _make_impact("faq_002", avg_delta=0.1, token_count=800,  classification="removable"),
        ],
        lean_configuration=["sys_001"],
        lean_quality_retention=0.97,
        lean_token_reduction=0.78,
        ordering_recommendations=[],
        redundancy_clusters=[("faq_001", "faq_002", 0.85)],
        pareto_configurations=[
            {"section_ids": ["sys_001", "faq_001"], "quality": 6.8, "tokens": 1500, "cost": 0.00045},
            {"section_ids": ["sys_001"],             "quality": 6.5, "tokens": 500,  "cost": 0.00015},
        ],
        total_api_calls=100,
        total_input_tokens=5_000_000,
        total_output_tokens=10_000,
        total_cost=1.53,
    )


# ── Tests: _format_section_table ──────────────────────────────────────────────


def test_format_section_table_contains_headers():
    """Output must include the markdown table header row."""
    impacts = [_make_impact("sec_001", 2.5, 500, "essential")]
    table = _format_section_table(impacts)
    assert "| Section |" in table
    assert "| Type |" in table
    assert "| Tokens |" in table
    assert "| Avg Δ |" in table
    assert "| Classification |" in table


def test_format_section_table_sorted_descending():
    """Sections must appear sorted by avg_quality_delta descending."""
    impacts = [
        _make_impact("low_001",  0.1, 100, "removable"),
        _make_impact("high_001", 3.0, 200, "essential"),
        _make_impact("mid_001",  1.5, 150, "moderate"),
    ]
    table = _format_section_table(impacts)
    idx_high = table.index("high_001")
    idx_mid  = table.index("mid_001")
    idx_low  = table.index("low_001")
    assert idx_high < idx_mid < idx_low


def test_format_section_table_empty_returns_fallback():
    """Empty impacts list should return a fallback message, not an empty table."""
    result = _format_section_table([])
    assert "No section impact data available" in result


# ── Tests: _format_redundancy_summary ─────────────────────────────────────────


def test_format_redundancy_summary_empty():
    """Empty clusters should return a no-clusters message."""
    result = _format_redundancy_summary([])
    assert "No redundancy clusters" in result


def test_format_redundancy_summary_with_clusters():
    """Non-empty clusters should list section pairs and similarity scores."""
    clusters = [("faq_001", "faq_002", 0.90), ("faq_003", "faq_004", 0.75)]
    result = _format_redundancy_summary(clusters)
    assert "faq_001" in result
    assert "faq_002" in result
    assert "0.90" in result
    assert "2 redundant section pair" in result


def test_format_redundancy_summary_caps_at_ten():
    """More than 10 clusters should be capped with a trailing 'and N more' line."""
    clusters = [(f"sec_{i:03d}", f"sec_{i+1:03d}", 0.80) for i in range(12)]
    result = _format_redundancy_summary(clusters)
    assert "2 more" in result


# ── Tests: _format_ordering_summary ───────────────────────────────────────────


def test_format_ordering_summary_empty():
    """Empty recommendations should return a no-experiments message."""
    result = _format_ordering_summary([])
    assert "No ordering experiments" in result


def test_format_ordering_summary_with_data():
    """Non-empty recommendations should include section labels and positions."""
    recs = [
        {
            "section_id": "rag_001",
            "label":       "FAQ Section",
            "best_position": "start",
            "quality_deltas": {"start": 0.5, "middle": 0.1, "end": -0.2},
            "quality_gain": 0.7,
        }
    ]
    result = _format_ordering_summary(recs)
    assert "FAQ Section" in result
    assert "start" in result
    assert "0.70" in result or "0.7" in result


# ── Tests: _format_pareto_summary ─────────────────────────────────────────────


def test_format_pareto_summary_empty():
    """Empty Pareto list should return a no-configurations message."""
    result = _format_pareto_summary([])
    assert "No Pareto-optimal configurations" in result


def test_format_pareto_summary_with_configs():
    """Non-empty Pareto configs should list token counts and quality values."""
    configs = [
        {"section_ids": ["a", "b"], "quality": 7.5, "tokens": 1000, "cost": 0.0003},
        {"section_ids": ["a"],       "quality": 6.0, "tokens": 500,  "cost": 0.00015},
    ]
    result = _format_pareto_summary(configs)
    assert "1,000 tokens" in result or "1000 tokens" in result
    assert "7.50" in result or "7.5" in result


# ── Tests: _build_diet_plan_prompt ────────────────────────────────────────────


def test_build_diet_plan_prompt_contains_all_sections(minimal_results, minimal_payload):
    """Prompt must include all required template sections."""
    prompt = _build_diet_plan_prompt(minimal_results, minimal_payload)
    assert "ABLATION SUMMARY" in prompt
    assert "SECTION ANALYSIS" in prompt
    assert "REDUNDANCY CLUSTERS" in prompt
    assert "ORDERING FINDINGS" in prompt
    assert "PARETO-OPTIMAL CONFIGURATIONS" in prompt
    assert "PRICING REFERENCE" in prompt


def test_build_diet_plan_prompt_token_counts(minimal_results, minimal_payload):
    """Token counts and quality retention must appear in the prompt."""
    prompt = _build_diet_plan_prompt(minimal_results, minimal_payload)
    assert "2,300" in prompt        # total_tokens
    assert "97.0%" in prompt        # quality_retention
    assert "78.0%" in prompt        # token_reduction


# ── Tests: generate_diet_plan ─────────────────────────────────────────────────


def test_generate_diet_plan_calls_high_reasoning_tier(minimal_results, minimal_payload):
    """generate_diet_plan must call client.invoke with reasoning_tier='high'."""
    client = MagicMock()
    client.invoke.return_value = (
        "# Context Diet Plan\n\nExecutive summary here.",
        True,
        {"input_tokens": 5000, "output_tokens": 2000, "total_tokens": 7000},
    )

    generate_diet_plan(minimal_results, minimal_payload, client)

    call_kwargs = client.invoke.call_args.kwargs
    assert call_kwargs.get("reasoning_tier") == "high"


def test_generate_diet_plan_does_not_pass_max_tokens(minimal_results, minimal_payload):
    """HIGH tier constraint: max_tokens must not be passed to invoke()."""
    client = MagicMock()
    client.invoke.return_value = (
        "Diet plan text.",
        True,
        {"input_tokens": 5000, "output_tokens": 2000, "total_tokens": 7000},
    )

    generate_diet_plan(minimal_results, minimal_payload, client)

    call_kwargs = client.invoke.call_args.kwargs
    assert "max_tokens" not in call_kwargs
    assert "temperature" not in call_kwargs


def test_generate_diet_plan_returns_text(minimal_results, minimal_payload):
    """generate_diet_plan must return the text from the client invoke response."""
    expected_text = "# Diet Plan\n\n1. Remove FAQ sections."
    client = MagicMock()
    client.invoke.return_value = (
        expected_text,
        True,
        {"input_tokens": 5000, "output_tokens": 2000, "total_tokens": 7000},
    )

    result = generate_diet_plan(minimal_results, minimal_payload, client)

    assert result == expected_text


def test_generate_diet_plan_fallback_on_api_error(minimal_results, minimal_payload):
    """If the API call fails, generate_diet_plan should return a fallback plan string."""
    client = MagicMock()
    client.invoke.side_effect = RuntimeError("Simulated API failure")

    result = generate_diet_plan(minimal_results, minimal_payload, client)

    assert isinstance(result, str)
    assert len(result) > 0
    # Fallback plan should mention removable sections
    assert "faq_001" in result or "faq_002" in result or "Remove" in result
