"""
core/ablation_engine.py — Minimal ablation experiment orchestrator (Phase 3).

Provides the core loop for:
  1. Baseline evaluation — score model responses on the full context.
  2. Single-section ablation — remove one section and re-score.
  3. Quality delta computation — measure impact of removing a section.

Phase 3 scope: baseline + single-section only. Full sweep, multi-section
elimination, ordering experiments, and AblationResults assembly are Phase 4.

Critical rules:
  - All Bedrock calls go through BedrockClient.invoke() — never directly.
  - Individual experiment failures are caught, logged, and skipped (not fatal).
  - Per-call token usage is tracked cumulatively via the client's own counters.

See context/architecture.md for the full pipeline design.
"""

from __future__ import annotations

import logging

from core.assembler import assemble_api_call
from core.models import ContextPayload, ScoringResult
from core.quality_scorer import score_response
from infra.bedrock_client import BedrockClient

logger = logging.getLogger(__name__)

# Max tokens for the response generation call (not scoring).
# 4096 gives Nova room for a substantive answer on large contexts.
_RESPONSE_MAX_TOKENS = 4096


def run_baseline(
    client: BedrockClient,
    payload: ContextPayload,
    tiers: list[str],
    num_queries: int | None = None,
) -> dict[str, dict[int, ScoringResult]]:
    """Run baseline evaluation: full context, across all specified tiers and queries.

    For each (tier, query) combination:
      1. Assemble full context (no exclusions).
      2. Invoke Nova to generate a response.
      3. Score the response via LLM-as-judge.

    Args:
        client:      BedrockClient instance.
        payload:     Parsed context payload (sections + queries).
        tiers:       Reasoning tiers to evaluate, e.g. ["disabled", "medium"].
        num_queries: Maximum number of queries to use. None = use all.

    Returns:
        Nested dict: {tier_name: {query_idx: ScoringResult}}.
        Missing entries indicate a failed experiment (logged and skipped).
    """
    queries = payload.evaluation_queries[:num_queries] if num_queries else payload.evaluation_queries
    scores: dict[str, dict[int, ScoringResult]] = {}

    for tier in tiers:
        scores[tier] = {}
        for q_idx, eval_query in enumerate(queries):
            try:
                api_params = assemble_api_call(payload.sections, eval_query.query)
                response_text, _, _usage = client.invoke(
                    system=api_params["system"],
                    messages=api_params["messages"],
                    reasoning_tier=tier,
                    max_tokens=_RESPONSE_MAX_TOKENS,
                )
                result, _score_usage = score_response(
                    client=client,
                    query=eval_query.query,
                    response_text=response_text,
                    reference_answer=eval_query.reference_answer,
                    criteria=payload.quality_criteria,
                )
                scores[tier][q_idx] = result
                logger.info(
                    "Baseline [tier=%s, query=%d] avg=%.2f",
                    tier, q_idx, result.avg_score(),
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Baseline experiment failed [tier=%s, query=%d]: %s",
                    tier, q_idx, exc,
                )

    return scores


def run_single_ablation(
    client: BedrockClient,
    payload: ContextPayload,
    section_id: str,
    tiers: list[str],
    num_queries: int | None = None,
) -> dict[str, dict[int, ScoringResult]]:
    """Run ablation with one section excluded, across all tiers and queries.

    Identical to run_baseline() except the named section is omitted from
    every assembled context.

    Args:
        client:     BedrockClient instance.
        payload:    Parsed context payload.
        section_id: ID of the section to exclude from all experiments.
        tiers:      Reasoning tiers to evaluate.
        num_queries: Maximum number of queries to use. None = use all.

    Returns:
        Nested dict: {tier_name: {query_idx: ScoringResult}}.
    """
    queries = payload.evaluation_queries[:num_queries] if num_queries else payload.evaluation_queries
    exclude = {section_id}
    scores: dict[str, dict[int, ScoringResult]] = {}

    for tier in tiers:
        scores[tier] = {}
        for q_idx, eval_query in enumerate(queries):
            try:
                api_params = assemble_api_call(
                    payload.sections,
                    eval_query.query,
                    exclude_ids=exclude,
                )
                response_text, _, _usage = client.invoke(
                    system=api_params["system"],
                    messages=api_params["messages"],
                    reasoning_tier=tier,
                    max_tokens=_RESPONSE_MAX_TOKENS,
                )
                result, _score_usage = score_response(
                    client=client,
                    query=eval_query.query,
                    response_text=response_text,
                    reference_answer=eval_query.reference_answer,
                    criteria=payload.quality_criteria,
                )
                scores[tier][q_idx] = result
                logger.info(
                    "Ablation [section=%s, tier=%s, query=%d] avg=%.2f",
                    section_id, tier, q_idx, result.avg_score(),
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Ablation experiment failed [section=%s, tier=%s, query=%d]: %s",
                    section_id, tier, q_idx, exc,
                )

    return scores


def compute_quality_delta(
    baseline_scores: dict[str, dict[int, ScoringResult]],
    ablated_scores:  dict[str, dict[int, ScoringResult]],
) -> float:
    """Compute the average quality delta between baseline and ablated scores.

    Delta = mean(baseline avg_scores) − mean(ablated avg_scores).

    Positive delta: the removed section helped quality.
    Negative delta: the removed section hurt quality (adding it back would lower quality).
    Zero delta:     the section had no measurable impact.

    Only query indices present in both baseline and ablated dicts (for a given
    tier) are included in the average, so partial experiment runs are handled
    gracefully.

    Args:
        baseline_scores: Output of run_baseline().
        ablated_scores:  Output of run_single_ablation().

    Returns:
        Average quality delta as a float. Returns 0.0 if no paired results
        are available for comparison.
    """
    deltas: list[float] = []

    for tier in baseline_scores:
        if tier not in ablated_scores:
            continue
        for q_idx in baseline_scores[tier]:
            if q_idx not in ablated_scores[tier]:
                continue
            b = baseline_scores[tier][q_idx].avg_score()
            a = ablated_scores[tier][q_idx].avg_score()
            deltas.append(b - a)

    if not deltas:
        return 0.0

    return sum(deltas) / len(deltas)
