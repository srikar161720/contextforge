"""
tests/test_analyzer.py — Unit tests for core/analyzer.py.

All tests are purely local (numpy/scipy only) — no API calls.
"""

from __future__ import annotations

import pytest

from core.analyzer import (
    build_pareto_candidates,
    compute_confidence_interval,
    compute_pareto_frontier,
    compute_section_impact,
    rank_sections,
    _classify,
)
from core.models import (
    ContextPayload,
    ContextSection,
    CriterionScore,
    EvalQuery,
    ScoringResult,
    SectionType,
)


# ── Shared helpers ────────────────────────────────────────────────────────────


def _criterion(score: int) -> CriterionScore:
    return CriterionScore(score=score, justification="Test.")


def _scoring_result(score: int) -> ScoringResult:
    c = _criterion(score)
    return ScoringResult(relevance=c, accuracy=c, completeness=c, groundedness=c)


def _make_section(
    id: str = "sec_001",
    token_count: int = 1000,
    section_type: SectionType = SectionType.RAG_DOCUMENT,
) -> ContextSection:
    return ContextSection(
        id=id,
        label=id,
        section_type=section_type,
        content="Some content.",
        token_count=token_count,
    )


def _make_payload(sections: list[ContextSection]) -> ContextPayload:
    return ContextPayload(
        sections=sections,
        evaluation_queries=[EvalQuery(query="Q1"), EvalQuery(query="Q2")],
        quality_criteria=["relevance", "accuracy", "completeness", "groundedness"],
        total_tokens=sum(s.token_count for s in sections),
    )


# ── _classify tests ───────────────────────────────────────────────────────────


def test_classify_harmful():
    """Negative delta → 'harmful'."""
    assert _classify(-0.5, 0.5, 2.0) == "harmful"


def test_classify_removable():
    """Zero to below impact_low → 'removable'."""
    assert _classify(0.0, 0.5, 2.0) == "removable"
    assert _classify(0.4, 0.5, 2.0) == "removable"


def test_classify_moderate():
    """impact_low ≤ delta < impact_high → 'moderate'."""
    assert _classify(0.5, 0.5, 2.0) == "moderate"
    assert _classify(1.5, 0.5, 2.0) == "moderate"


def test_classify_essential():
    """delta ≥ impact_high → 'essential'."""
    assert _classify(2.0, 0.5, 2.0) == "essential"
    assert _classify(5.0, 0.5, 2.0) == "essential"


# ── compute_section_impact tests ──────────────────────────────────────────────


def test_compute_section_impact_essential():
    """Section with large delta is classified 'essential'."""
    baseline = {"disabled": {0: _scoring_result(9), 1: _scoring_result(9)}}
    ablated  = {"disabled": {0: _scoring_result(5), 1: _scoring_result(5)}}
    section  = _make_section(token_count=500)

    impact = compute_section_impact(baseline, ablated, section, {})

    assert impact.classification == "essential"
    assert impact.avg_quality_delta == pytest.approx(4.0)


def test_compute_section_impact_moderate():
    """Section with delta in [0.5, 2.0) is classified 'moderate'."""
    baseline = {"disabled": {0: _scoring_result(8)}}
    ablated  = {"disabled": {0: _scoring_result(7)}}
    section  = _make_section(token_count=200)

    impact = compute_section_impact(baseline, ablated, section, {})

    assert impact.classification == "moderate"
    assert impact.avg_quality_delta == pytest.approx(1.0)


def test_compute_section_impact_removable():
    """Section with delta near 0 is classified 'removable'."""
    baseline = {"disabled": {0: _scoring_result(7)}}
    ablated  = {"disabled": {0: _scoring_result(7)}}
    section  = _make_section(token_count=300)

    impact = compute_section_impact(baseline, ablated, section, {})

    assert impact.classification == "removable"
    assert impact.avg_quality_delta == pytest.approx(0.0)


def test_compute_section_impact_harmful():
    """Section with negative delta is classified 'harmful'."""
    baseline = {"disabled": {0: _scoring_result(6)}}
    ablated  = {"disabled": {0: _scoring_result(8)}}
    section  = _make_section(token_count=400)

    impact = compute_section_impact(baseline, ablated, section, {})

    assert impact.classification == "harmful"
    assert impact.avg_quality_delta == pytest.approx(-2.0)


def test_compute_section_impact_tier_sensitivity_nonzero():
    """Different per-tier deltas produce nonzero tier_sensitivity (variance)."""
    baseline = {
        "disabled": {0: _scoring_result(9)},
        "medium":   {0: _scoring_result(9)},
    }
    ablated = {
        "disabled": {0: _scoring_result(7)},   # delta = 2
        "medium":   {0: _scoring_result(5)},   # delta = 4
    }
    section = _make_section(token_count=500)

    impact = compute_section_impact(baseline, ablated, section, {})

    assert impact.tier_sensitivity > 0.0
    # Variance of [2.0, 4.0] = 1.0
    assert impact.tier_sensitivity == pytest.approx(1.0)


def test_compute_section_impact_quality_per_token():
    """quality_per_token = avg_quality_delta / token_count."""
    baseline = {"disabled": {0: _scoring_result(8)}}
    ablated  = {"disabled": {0: _scoring_result(4)}}
    section  = _make_section(token_count=2000)

    impact = compute_section_impact(baseline, ablated, section, {})

    assert impact.quality_per_token == pytest.approx(4.0 / 2000)


def test_compute_section_impact_zero_token_count_no_error():
    """A section with token_count=0 must not raise ZeroDivisionError."""
    baseline = {"disabled": {0: _scoring_result(7)}}
    ablated  = {"disabled": {0: _scoring_result(5)}}
    section  = _make_section(token_count=0)

    impact = compute_section_impact(baseline, ablated, section, {})

    assert impact.quality_per_token == pytest.approx(2.0)   # uses max(0, 1) = 1


def test_compute_section_impact_missing_queries_in_ablated():
    """Only query indices common to both baseline and ablated are used."""
    baseline = {"disabled": {0: _scoring_result(8), 1: _scoring_result(8)}}
    # q_idx=1 is absent from ablated
    ablated  = {"disabled": {0: _scoring_result(6)}}
    section  = _make_section(token_count=100)

    impact = compute_section_impact(baseline, ablated, section, {})

    # Only q_idx=0 contributes: delta = 8-6 = 2
    assert impact.avg_quality_delta == pytest.approx(2.0)


def test_compute_section_impact_missing_tier_skipped():
    """Tiers in baseline but absent from ablated are skipped gracefully."""
    baseline = {
        "disabled": {0: _scoring_result(8)},
        "medium":   {0: _scoring_result(8)},
    }
    ablated = {"disabled": {0: _scoring_result(6)}}   # "medium" missing
    section = _make_section(token_count=100)

    impact = compute_section_impact(baseline, ablated, section, {})

    # Only "disabled" tier contributes: delta = 2
    assert impact.avg_quality_delta == pytest.approx(2.0)
    assert "disabled" in impact.quality_delta_by_tier
    assert "medium" not in impact.quality_delta_by_tier


def test_compute_section_impact_preserves_section_metadata():
    """SectionImpact fields are populated from the ContextSection object."""
    baseline = {"disabled": {0: _scoring_result(7)}}
    ablated  = {"disabled": {0: _scoring_result(6)}}
    section  = _make_section(id="rag_007", token_count=123, section_type=SectionType.RAG_DOCUMENT)

    impact = compute_section_impact(baseline, ablated, section, {})

    assert impact.section_id == "rag_007"
    assert impact.token_count == 123
    assert impact.section_type == "rag_document"


# ── rank_sections tests ───────────────────────────────────────────────────────


def test_rank_sections_sorted_descending():
    """rank_sections returns impacts sorted by avg_quality_delta descending."""
    def _make_impact(sid: str, delta: float):
        baseline = {"disabled": {0: _scoring_result(int(5 + delta))}}
        ablated  = {"disabled": {0: _scoring_result(5)}}
        return compute_section_impact(baseline, ablated, _make_section(id=sid), {})

    impacts = [_make_impact("a", 1.0), _make_impact("b", 3.0), _make_impact("c", 0.2)]
    ranked  = rank_sections(impacts)

    deltas = [r.avg_quality_delta for r in ranked]
    assert deltas == sorted(deltas, reverse=True)
    assert ranked[0].section_id == "b"


def test_rank_sections_empty_list():
    """rank_sections on an empty list returns an empty list."""
    assert rank_sections([]) == []


# ── compute_confidence_interval tests ────────────────────────────────────────


def test_confidence_interval_empty_list():
    """Empty list returns (0.0, 0.0)."""
    assert compute_confidence_interval([]) == (0.0, 0.0)


def test_confidence_interval_single_value():
    """Single value returns (value, value) — no spread to measure."""
    lo, hi = compute_confidence_interval([4.5])
    assert lo == pytest.approx(4.5)
    assert hi == pytest.approx(4.5)


def test_confidence_interval_identical_values():
    """All identical values → CI width is zero (SEM = 0)."""
    lo, hi = compute_confidence_interval([3.0, 3.0, 3.0])
    assert lo == pytest.approx(3.0)
    assert hi == pytest.approx(3.0)


def test_confidence_interval_contains_mean():
    """The 95% CI should bracket the sample mean for a moderate spread."""
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    import numpy as np
    mean   = float(np.mean(values))
    lo, hi = compute_confidence_interval(values)

    assert lo < mean < hi
    assert lo < hi


# ── build_pareto_candidates tests ─────────────────────────────────────────────


def test_build_pareto_candidates_includes_full_context():
    """First candidate always represents the full context (no exclusions)."""
    section  = _make_section("s1", token_count=1000)
    payload  = _make_payload([section])
    baseline = {"disabled": {0: _scoring_result(8)}}

    # No impacts to remove
    candidates = build_pareto_candidates(payload, [], 8.0)

    assert len(candidates) >= 1
    full = candidates[-1]   # sorted by tokens ascending; full context has most tokens
    assert full["tokens"] == 1000
    assert full["quality"] == pytest.approx(8.0)


def test_build_pareto_candidates_sorted_by_tokens():
    """Returned list is sorted by tokens ascending."""
    s1 = _make_section("s1", token_count=1000)
    s2 = _make_section("s2", token_count=2000)
    payload  = _make_payload([s1, s2])
    baseline = {"disabled": {0: _scoring_result(8)}}

    # One impact to drive progressive removal
    impact = compute_section_impact(
        {"disabled": {0: _scoring_result(8)}},
        {"disabled": {0: _scoring_result(7)}},
        s1, {},
    )

    candidates = build_pareto_candidates(payload, [impact], 8.0)
    token_counts = [c["tokens"] for c in candidates]
    assert token_counts == sorted(token_counts)


def test_build_pareto_candidates_quality_decreases():
    """Quality should decrease (or stay equal) as more sections are removed."""
    s1 = _make_section("s1", token_count=500)
    s2 = _make_section("s2", token_count=500)
    payload  = _make_payload([s1, s2])

    impact_s1 = compute_section_impact(
        {"disabled": {0: _scoring_result(8)}},
        {"disabled": {0: _scoring_result(5)}},
        s1, {},
    )
    impact_s2 = compute_section_impact(
        {"disabled": {0: _scoring_result(8)}},
        {"disabled": {0: _scoring_result(6)}},
        s2, {},
    )

    candidates = build_pareto_candidates(payload, [impact_s1, impact_s2], 8.0)

    # Full context (highest tokens) should have highest quality
    assert candidates[-1]["quality"] >= candidates[0]["quality"]


# ── compute_pareto_frontier tests ─────────────────────────────────────────────


def test_pareto_frontier_single_config():
    """A single configuration is always Pareto-optimal."""
    configs = [{"quality": 7.5, "tokens": 100_000}]
    frontier = compute_pareto_frontier(configs)
    assert len(frontier) == 1


def test_pareto_frontier_dominated_config_excluded():
    """A dominated configuration is excluded from the frontier."""
    # Config B dominates Config A (better quality, same tokens)
    configs = [
        {"quality": 6.0, "tokens": 100_000},   # A — dominated
        {"quality": 8.0, "tokens": 100_000},   # B — dominates A
    ]
    frontier = compute_pareto_frontier(configs)
    qualities = [c["quality"] for c in frontier]
    assert 6.0 not in qualities
    assert 8.0 in qualities


def test_pareto_frontier_two_non_dominated():
    """Two configs that each dominate the other in one dimension are both kept."""
    configs = [
        {"quality": 8.0, "tokens": 200_000},   # high quality, high tokens
        {"quality": 6.0, "tokens":  80_000},   # lower quality, fewer tokens
    ]
    frontier = compute_pareto_frontier(configs)
    assert len(frontier) == 2


def test_pareto_frontier_sorted_by_quality_descending():
    """Frontier is sorted by quality descending."""
    configs = [
        {"quality": 5.0, "tokens": 50_000},
        {"quality": 8.0, "tokens": 150_000},
        {"quality": 7.0, "tokens": 100_000},
    ]
    frontier = compute_pareto_frontier(configs)
    qualities = [c["quality"] for c in frontier]
    assert qualities == sorted(qualities, reverse=True)


def test_pareto_frontier_empty_input():
    """Empty input returns empty output."""
    assert compute_pareto_frontier([]) == []
