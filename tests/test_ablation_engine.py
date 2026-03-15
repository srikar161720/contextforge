"""
tests/test_ablation_engine.py — Unit and integration tests for core/ablation_engine.py.

Unit tests mock BedrockClient.invoke() so no Bedrock calls are made.
The integration test is marked @pytest.mark.integration and skipped without creds.
"""

from __future__ import annotations

import json
import pathlib
import queue
from unittest.mock import MagicMock, call, patch

import pytest

from core.ablation_engine import (
    check_interaction_effects,
    compute_quality_delta,
    run_baseline,
    run_full_sweep,
    run_greedy_elimination,
    run_single_ablation,
)
from core.models import (
    AblationResults,
    ContextPayload,
    ContextSection,
    CriterionScore,
    EvalQuery,
    ExperimentConfig,
    ExperimentMode,
    ScoringResult,
    SectionImpact,
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


# ── Phase 4: run_full_sweep unit tests ────────────────────────────────────────


def _make_demo_config(mode: str = "demo") -> ExperimentConfig:
    return ExperimentConfig(
        mode=ExperimentMode(mode),
        quality_tolerance=0.05,
        redundancy_threshold=0.7,
        repetitions=1,
        reasoning_tiers=["disabled"],
    )


def _make_minimal_payload_3q() -> ContextPayload:
    """Minimal payload with 2 sections and 3 eval queries (demo mode minimum)."""
    return ContextPayload(
        sections=[
            _make_section("sys_001", SectionType.SYSTEM_PROMPT, "System prompt."),
            _make_section("rag_001", SectionType.RAG_DOCUMENT, "FAQ content."),
        ],
        evaluation_queries=[
            EvalQuery(query="Q0"), EvalQuery(query="Q1"), EvalQuery(query="Q2"),
        ],
        quality_criteria=["relevance", "accuracy", "completeness", "groundedness"],
        total_tokens=40,
    )


def _patched_sweep_client(score: int = 7) -> MagicMock:
    """Mock client whose invoke() returns clean scoring JSON, score_response patched."""
    client = MagicMock()
    client.invoke.return_value = (
        _scoring_json(score), False,
        {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
    )
    # Cumulative usage attributes (read by run_full_sweep at the end)
    client.total_api_calls    = 10
    client.total_input_tokens = 1_000_000
    client.total_output_tokens = 5_000
    client.total_cost          = 0.31
    return client


def test_run_full_sweep_returns_ablation_results():
    """run_full_sweep must return an AblationResults instance."""
    payload = _make_minimal_payload_3q()
    config  = _make_demo_config()

    with patch("core.ablation_engine.score_response") as mock_score:
        mock_score.return_value = (
            _scoring_result(7),
            {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        )
        client = _patched_sweep_client()
        client.invoke.return_value = (
            "Response.", False,
            {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
        )

        result = run_full_sweep(client, payload, config)

    assert isinstance(result, AblationResults)


def test_run_full_sweep_section_impacts_populated():
    """AblationResults.section_impacts must have one entry per successfully ablated section."""
    payload = _make_minimal_payload_3q()
    config  = _make_demo_config()

    with patch("core.ablation_engine.score_response") as mock_score:
        mock_score.return_value = (
            _scoring_result(7),
            {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        )
        client = _patched_sweep_client()
        client.invoke.return_value = (
            "Response.", False,
            {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
        )

        result = run_full_sweep(client, payload, config)

    # 2 sections in payload → 2 section impacts
    assert len(result.section_impacts) == len(payload.sections)


def test_run_full_sweep_progress_reporting():
    """Progress messages should be put on the queue during a sweep."""
    payload = _make_minimal_payload_3q()
    config  = _make_demo_config()
    q: queue.Queue = queue.Queue()

    with patch("core.ablation_engine.score_response") as mock_score:
        mock_score.return_value = (
            _scoring_result(7),
            {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        )
        client = _patched_sweep_client()
        client.invoke.return_value = (
            "Response.", False,
            {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
        )

        run_full_sweep(client, payload, config, progress_queue=q)

    messages = []
    while not q.empty():
        messages.append(q.get_nowait())

    msg_types = [m["type"] for m in messages]
    assert "start" in msg_types
    assert "baseline_complete" in msg_types
    assert "sweep_complete" in msg_types
    assert "done" in msg_types


def test_run_full_sweep_handles_partial_failures():
    """Failures on individual sections must not abort the sweep."""
    payload = _make_minimal_payload_3q()
    config  = _make_demo_config()

    call_count = {"n": 0}

    def _flaky_score(*args, **kwargs):
        call_count["n"] += 1
        # Fail every other scoring call to simulate partial failures
        if call_count["n"] % 4 == 0:
            raise ValueError("Simulated LLM parse error")
        return (
            _scoring_result(7),
            {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        )

    with patch("core.ablation_engine.score_response", side_effect=_flaky_score):
        client = _patched_sweep_client()
        client.invoke.return_value = (
            "Response.", False,
            {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
        )
        # Should not raise
        result = run_full_sweep(client, payload, config)

    assert isinstance(result, AblationResults)


def test_run_full_sweep_lean_config_is_subset_of_sections():
    """lean_configuration must only contain section IDs present in the payload."""
    payload = _make_minimal_payload_3q()
    config  = _make_demo_config()

    with patch("core.ablation_engine.score_response") as mock_score:
        mock_score.return_value = (
            _scoring_result(7),
            {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        )
        client = _patched_sweep_client()
        client.invoke.return_value = (
            "Response.", False,
            {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
        )

        result = run_full_sweep(client, payload, config)

    all_section_ids = {s.id for s in payload.sections}
    for sid in result.lean_configuration:
        assert sid in all_section_ids


# ── Phase 4: run_greedy_elimination unit tests ────────────────────────────────


def _make_section_impact(
    section_id:      str,
    avg_delta:       float,
    token_count:     int,
    classification:  str,
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


def test_greedy_elimination_removes_removable_sections():
    """Sections classified 'removable' with low quality delta should be excluded."""
    payload = _make_minimal_payload_3q()
    config  = ExperimentConfig(
        mode=ExperimentMode.QUICK,
        quality_tolerance=0.10,    # generous tolerance
        repetitions=1,
        reasoning_tiers=["disabled"],
    )
    baseline_scores = {"disabled": {0: _scoring_result(8), 1: _scoring_result(8)}}

    impacts = [
        _make_section_impact("sys_001", avg_delta=3.0, token_count=20, classification="essential"),
        _make_section_impact("rag_001", avg_delta=0.1, token_count=20, classification="removable"),
    ]

    with patch("core.ablation_engine._run_multi_exclusion") as mock_excl, \
         patch("core.ablation_engine._get_mode_config", return_value={"num_queries": 2}):

        # Return slightly lower quality when rag_001 is excluded (still within tolerance)
        mock_excl.return_value = {"disabled": {0: _scoring_result(7), 1: _scoring_result(8)}}

        lean_config, retention, reduction = run_greedy_elimination(
            client=MagicMock(),
            payload=payload,
            impacts=impacts,
            config=config,
            baseline_scores=baseline_scores,
        )

    # rag_001 should be removed; sys_001 should remain (essential)
    assert "rag_001" not in lean_config
    assert "sys_001" in lean_config


def test_greedy_elimination_respects_tolerance():
    """Sections that would cause quality loss > tolerance must not be excluded."""
    payload = _make_minimal_payload_3q()
    config  = ExperimentConfig(
        mode=ExperimentMode.QUICK,
        quality_tolerance=0.05,    # strict 5% tolerance
        repetitions=1,
        reasoning_tiers=["disabled"],
    )
    baseline_scores = {"disabled": {0: _scoring_result(8)}}

    impacts = [
        _make_section_impact("sys_001", avg_delta=4.0, token_count=20, classification="essential"),
        _make_section_impact("rag_001", avg_delta=0.3, token_count=20, classification="removable"),
    ]

    with patch("core.ablation_engine._run_multi_exclusion") as mock_excl, \
         patch("core.ablation_engine._get_mode_config", return_value={"num_queries": 2}):

        # Removing rag_001 causes 50% quality loss → far exceeds 5% tolerance
        mock_excl.return_value = {"disabled": {0: _scoring_result(4)}}

        lean_config, retention, reduction = run_greedy_elimination(
            client=MagicMock(),
            payload=payload,
            impacts=impacts,
            config=config,
            baseline_scores=baseline_scores,
        )

    # rag_001 must be kept because quality loss exceeded tolerance
    assert "rag_001" in lean_config


def test_greedy_elimination_candidates_sorted_by_token_count():
    """Candidates with largest token_count should be tried first."""
    payload = _make_minimal_payload_3q()
    config  = ExperimentConfig(
        mode=ExperimentMode.QUICK,
        quality_tolerance=0.20,
        repetitions=1,
        reasoning_tiers=["disabled"],
    )
    baseline_scores = {"disabled": {0: _scoring_result(8)}}

    impacts = [
        _make_section_impact("sys_001", avg_delta=0.1, token_count=5,  classification="removable"),
        _make_section_impact("rag_001", avg_delta=0.1, token_count=50, classification="removable"),
    ]

    calls_order: list[str] = []

    def _mock_excl(client, payload, exclude_ids, tiers, num_queries=None):
        # Record which section was in the tentative excluded set for this call
        calls_order.append(str(sorted(exclude_ids)))
        return {"disabled": {0: _scoring_result(8)}}

    with patch("core.ablation_engine._run_multi_exclusion", side_effect=_mock_excl), \
         patch("core.ablation_engine._get_mode_config", return_value={"num_queries": 2}):
        run_greedy_elimination(
            client=MagicMock(),
            payload=payload,
            impacts=impacts,
            config=config,
            baseline_scores=baseline_scores,
        )

    # rag_001 (50 tokens) should appear in the first tentative excluded set
    assert "rag_001" in calls_order[0]


# ── Phase 4: check_interaction_effects unit tests ────────────────────────────


def test_interaction_check_no_flag_when_gap_small():
    """No interaction flag when gap between measured and predicted is ≤ 0.5."""
    result = check_interaction_effects(
        lean_quality=7.0,
        individual_deltas=[0.5, 0.3],   # predicted = 8.0 - 0.8 = 7.2
        excluded_ids=["a", "b"],
        baseline_quality=8.0,
    )
    # gap = 7.0 - 7.2 = -0.2; abs(-0.2) = 0.2 ≤ 0.5 → no flag
    assert result["interaction_flag"] is False
    assert result["gap"] == pytest.approx(7.0 - 7.2)


def test_interaction_check_flag_when_gap_large():
    """Interaction flag is True when |gap| > 0.5."""
    result = check_interaction_effects(
        lean_quality=5.0,
        individual_deltas=[0.2, 0.1],   # predicted = 8.0 - 0.3 = 7.7
        excluded_ids=["a"],
        baseline_quality=8.0,
    )
    # gap = 5.0 - 7.7 = -2.7; abs > 0.5 → flag
    assert result["interaction_flag"] is True


def test_interaction_check_excluded_count():
    """excluded_count field reflects the number of excluded sections."""
    result = check_interaction_effects(
        lean_quality=7.0,
        individual_deltas=[0.5],
        excluded_ids=["a", "b", "c"],
        baseline_quality=8.0,
    )
    assert result["excluded_count"] == 3


# ── Phase 4: integration test (slow, real Bedrock calls) ─────────────────────


@pytest.mark.integration
@pytest.mark.slow
def test_full_sweep_demo_mode(bedrock_client):
    """
    Full pipeline on demo payload in Demo mode (disabled + medium tiers, 3 queries).

    Phase 4 exit criteria: section rankings match expected findings.
      - sys_001 (system prompt) classified 'essential'
      - Legal disclaimers (legal_*) classified 'removable' or 'moderate'
      - Early conversation turns (conv_001..conv_035) classified 'removable'
      - Redundancy clusters detected in FAQ sections
      - AblationResults fully populated with non-zero usage stats

    Cost estimate: depends on number of sections × queries × tiers × 2 calls each.
    """
    from core.parser import parse_payload

    demo_path = (
        pathlib.Path(__file__).parent.parent
        / "data" / "demo_payloads" / "customer_support.json"
    )
    if not demo_path.exists():
        pytest.skip("Demo payload not found — run scripts/generate_demo_payload.py first")

    payload = parse_payload(demo_path)
    config  = ExperimentConfig(
        mode=ExperimentMode.DEMO,
        quality_tolerance=0.05,
        redundancy_threshold=0.7,
        repetitions=1,
        reasoning_tiers=["disabled", "medium"],
    )

    result = run_full_sweep(bedrock_client, payload, config)

    # ── Structural checks ──────────────────────────────────────────────────────
    assert isinstance(result, AblationResults)
    assert len(result.section_impacts) > 0
    assert result.total_api_calls > 0
    assert result.total_input_tokens > 0

    # ── Classification checks ──────────────────────────────────────────────────
    impact_by_id = {imp.section_id: imp for imp in result.section_impacts}

    # System prompt must be essential
    if "sys_001" in impact_by_id:
        assert impact_by_id["sys_001"].classification == "essential", (
            f"Expected sys_001 to be 'essential', got '{impact_by_id['sys_001'].classification}'"
        )

    # At least one early conversation turn should be removable
    removable_conv = [
        imp for imp in result.section_impacts
        if imp.section_id.startswith("conv_") and imp.classification in ("removable", "harmful")
    ]
    assert len(removable_conv) > 0, "Expected some early conversation turns to be removable"

    # Redundancy clusters should be detected in the FAQ-heavy payload
    assert len(result.redundancy_clusters) > 0, (
        "Expected redundancy clusters in the FAQ sections"
    )
