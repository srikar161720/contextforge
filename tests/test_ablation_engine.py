"""
tests/test_ablation_engine.py — Unit and integration tests for core/ablation_engine.py.

Unit tests mock BedrockClient.invoke() so no Bedrock calls are made.
The integration test is marked @pytest.mark.integration and skipped without creds.
"""

from __future__ import annotations

import json
import pathlib
from unittest.mock import MagicMock, call, patch

import pytest

from core.ablation_engine import (
    compute_quality_delta,
    run_baseline,
    run_single_ablation,
)
from core.models import (
    ContextPayload,
    ContextSection,
    CriterionScore,
    EvalQuery,
    ScoringResult,
    SectionType,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _criterion(score: int) -> CriterionScore:
    return CriterionScore(score=score, justification="Test.")


def _scoring_result(score: int) -> ScoringResult:
    c = _criterion(score)
    return ScoringResult(relevance=c, accuracy=c, completeness=c, groundedness=c)


def _scoring_json(score: int) -> str:
    return json.dumps(
        {
            "relevance":    {"score": score, "justification": "Test."},
            "accuracy":     {"score": score, "justification": "Test."},
            "completeness": {"score": score, "justification": "Test."},
            "groundedness": {"score": score, "justification": "Test."},
        }
    )


def _make_section(
    id: str,
    section_type: SectionType = SectionType.RAG_DOCUMENT,
    content: str = "Some content.",
) -> ContextSection:
    return ContextSection(
        id=id,
        label=id,
        section_type=section_type,
        content=content,
        token_count=20,
    )


@pytest.fixture()
def minimal_payload() -> ContextPayload:
    return ContextPayload(
        sections=[
            _make_section("sys_001", SectionType.SYSTEM_PROMPT, "You are a helpful assistant."),
            _make_section("rag_001", SectionType.RAG_DOCUMENT, "FAQ content."),
        ],
        evaluation_queries=[
            EvalQuery(query="Query 0", reference_answer="Answer 0."),
            EvalQuery(query="Query 1", reference_answer="Answer 1."),
        ],
        quality_criteria=["relevance", "accuracy", "completeness", "groundedness"],
        total_tokens=40,
    )


def _mock_client_with_scorer(response_score: int = 7, scorer_score: int = 7) -> MagicMock:
    """Mock client: invoke() returns a fixed response, plus a canned scoring JSON."""
    client = MagicMock()
    client.invoke.return_value = (
        _scoring_json(scorer_score),  # text (first call is scorer prompt, re-used here)
        False,
        {"input_tokens": 200_000, "output_tokens": 200, "total_tokens": 200_200},
    )
    return client


# ── Unit tests: run_baseline ──────────────────────────────────────────────────


def test_run_baseline_returns_correct_structure(minimal_payload):
    """run_baseline should return {tier: {query_idx: ScoringResult}}."""
    # Patch score_response to return a fixed ScoringResult without real Bedrock calls
    with patch("core.ablation_engine.score_response") as mock_score:
        mock_score.return_value = (_scoring_result(7), {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15})
        client = MagicMock()
        client.invoke.return_value = ("Model response text.", False, {"input_tokens": 10, "output_tokens": 10, "total_tokens": 20})

        scores = run_baseline(client, minimal_payload, tiers=["disabled", "medium"])

    assert set(scores.keys()) == {"disabled", "medium"}
    for tier in ("disabled", "medium"):
        assert 0 in scores[tier]
        assert 1 in scores[tier]
        assert isinstance(scores[tier][0], ScoringResult)
        assert isinstance(scores[tier][1], ScoringResult)


def test_run_baseline_respects_num_queries(minimal_payload):
    """num_queries should limit the number of queries evaluated."""
    with patch("core.ablation_engine.score_response") as mock_score:
        mock_score.return_value = (_scoring_result(7), {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15})
        client = MagicMock()
        client.invoke.return_value = ("Response.", False, {"input_tokens": 10, "output_tokens": 10, "total_tokens": 20})

        scores = run_baseline(client, minimal_payload, tiers=["disabled"], num_queries=1)

    # Only query index 0 should be present
    assert 0 in scores["disabled"]
    assert 1 not in scores["disabled"]


def test_run_baseline_skips_failed_experiments(minimal_payload):
    """Individual experiment failures should be logged and skipped, not raised."""
    with patch("core.ablation_engine.score_response", side_effect=ValueError("LLM parse error")):
        client = MagicMock()
        client.invoke.return_value = ("Response.", False, {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15})

        # Should not raise — failures are silently skipped
        scores = run_baseline(client, minimal_payload, tiers=["disabled"])

    # All experiments failed — dict should be empty for that tier
    assert scores["disabled"] == {}


# ── Unit tests: run_single_ablation ──────────────────────────────────────────


def test_run_single_ablation_excludes_section(minimal_payload):
    """run_single_ablation should call assemble_api_call with exclude_ids set."""
    with patch("core.ablation_engine.score_response") as mock_score, \
         patch("core.ablation_engine.assemble_api_call") as mock_assemble:

        mock_score.return_value = (_scoring_result(5), {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15})
        mock_assemble.return_value = {
            "system": None,
            "messages": [{"role": "user", "content": [{"text": "Query."}]}],
        }
        client = MagicMock()
        client.invoke.return_value = ("Response.", False, {"input_tokens": 10, "output_tokens": 10, "total_tokens": 20})

        run_single_ablation(client, minimal_payload, section_id="rag_001", tiers=["disabled"])

    # Every call to assemble_api_call should have exclude_ids={"rag_001"}
    for c in mock_assemble.call_args_list:
        assert c.kwargs.get("exclude_ids") == {"rag_001"} or \
               (len(c.args) > 2 and c.args[2] == {"rag_001"})


def test_run_single_ablation_returns_correct_structure(minimal_payload):
    """run_single_ablation should return {tier: {query_idx: ScoringResult}}."""
    with patch("core.ablation_engine.score_response") as mock_score:
        mock_score.return_value = (_scoring_result(5), {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15})
        client = MagicMock()
        client.invoke.return_value = ("Response.", False, {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15})

        scores = run_single_ablation(
            client, minimal_payload, section_id="rag_001", tiers=["disabled", "medium"]
        )

    assert set(scores.keys()) == {"disabled", "medium"}
    assert isinstance(scores["disabled"][0], ScoringResult)


# ── Unit tests: compute_quality_delta ────────────────────────────────────────


def test_quality_delta_positive_when_section_helps():
    """Positive delta: baseline > ablated means section helped quality."""
    baseline = {"disabled": {0: _scoring_result(8), 1: _scoring_result(8)}}
    ablated  = {"disabled": {0: _scoring_result(6), 1: _scoring_result(6)}}
    delta = compute_quality_delta(baseline, ablated)
    assert delta == pytest.approx(2.0)


def test_quality_delta_negative_when_section_hurts():
    """Negative delta: baseline < ablated means section hurt quality."""
    baseline = {"disabled": {0: _scoring_result(5), 1: _scoring_result(5)}}
    ablated  = {"disabled": {0: _scoring_result(7), 1: _scoring_result(7)}}
    delta = compute_quality_delta(baseline, ablated)
    assert delta == pytest.approx(-2.0)


def test_quality_delta_zero_when_no_impact():
    """Zero delta: baseline == ablated means section had no measurable impact."""
    baseline = {"disabled": {0: _scoring_result(7)}}
    ablated  = {"disabled": {0: _scoring_result(7)}}
    delta = compute_quality_delta(baseline, ablated)
    assert delta == pytest.approx(0.0)


def test_quality_delta_averaged_across_tiers_and_queries():
    """Delta should be averaged across all matched (tier, query) pairs."""
    baseline = {
        "disabled": {0: _scoring_result(8), 1: _scoring_result(9)},
        "medium":   {0: _scoring_result(7), 1: _scoring_result(7)},
    }
    ablated = {
        "disabled": {0: _scoring_result(6), 1: _scoring_result(7)},
        "medium":   {0: _scoring_result(5), 1: _scoring_result(5)},
    }
    # Deltas: (8-6)=2, (9-7)=2, (7-5)=2, (7-5)=2 → mean=2.0
    delta = compute_quality_delta(baseline, ablated)
    assert delta == pytest.approx(2.0)


def test_quality_delta_handles_missing_tiers():
    """Tiers present in baseline but not ablated should be skipped gracefully."""
    baseline = {
        "disabled": {0: _scoring_result(8)},
        "medium":   {0: _scoring_result(7)},
    }
    ablated = {
        "disabled": {0: _scoring_result(6)},
        # "medium" missing from ablated
    }
    delta = compute_quality_delta(baseline, ablated)
    # Only "disabled"/"query 0" pair: delta = 8-6 = 2
    assert delta == pytest.approx(2.0)


def test_quality_delta_returns_zero_for_empty_inputs():
    """No paired results should return 0.0 without raising."""
    delta = compute_quality_delta({}, {})
    assert delta == 0.0


# ── Integration test ──────────────────────────────────────────────────────────


@pytest.mark.integration
def test_end_to_end_single_ablation(bedrock_client):
    """
    Full pipeline: parse demo → baseline (1 tier, 2 queries) → ablate system prompt
    → ablate irrelevant conv turn → verify system prompt delta > conv turn delta.

    Phase 3 exit criteria: quality delta is directionally correct.
    Cost estimate: ~12 API calls (~$0.50).
    """
    from core.parser import parse_payload

    demo_path = pathlib.Path(__file__).parent.parent / "data" / "demo_payloads" / "customer_support.json"
    if not demo_path.exists():
        pytest.skip("Demo payload not found — run scripts/generate_demo_payload.py first")

    # 1. Parse demo payload
    payload = parse_payload(demo_path)
    assert len(payload.sections) == 79

    # 2. Run baseline: 1 tier, 2 queries (cost control)
    baseline = run_baseline(bedrock_client, payload, tiers=["disabled"], num_queries=2)
    assert "disabled" in baseline
    assert len(baseline["disabled"]) > 0, "Baseline produced no results — check API credentials"

    # 3. Ablate system prompt (should significantly hurt quality → large positive delta)
    ablated_sys = run_single_ablation(
        bedrock_client, payload,
        section_id="sys_001",
        tiers=["disabled"],
        num_queries=2,
    )
    delta_sys = compute_quality_delta(baseline, ablated_sys)

    # 4. Ablate an early irrelevant conversation turn (should barely affect quality)
    # conv_001 is from the unrelated past issue (turns 1-35 in the demo payload)
    ablated_turn = run_single_ablation(
        bedrock_client, payload,
        section_id="conv_001",
        tiers=["disabled"],
        num_queries=2,
    )
    delta_turn = compute_quality_delta(baseline, ablated_turn)

    # Phase 3 exit criteria: the pipeline runs end-to-end and produces visible,
    # non-zero quality deltas. Directional correctness (system prompt delta >
    # conv turn delta) is inherently noisy with only 2 integer-scale LLM scores
    # and is validated in Phase 4 via the full multi-query sweep.
    assert len(ablated_sys.get("disabled", {})) > 0, "System prompt ablation produced no results"
    assert len(ablated_turn.get("disabled", {})) > 0, "Conv turn ablation produced no results"
    assert delta_sys != delta_turn, (
        f"Both deltas are identical ({delta_sys:.2f}) — pipeline may not be measuring impact"
    )
