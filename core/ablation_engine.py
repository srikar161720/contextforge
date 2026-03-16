"""
core/ablation_engine.py — Ablation experiment orchestrator (Phase 3 + Phase 4 + Phase 5).

Provides the full ablation pipeline:
  1. Baseline evaluation — score model responses on the full context.
  2. Single-section ablation — remove one section and re-score.
  3. Quality delta computation — measure impact of removing a section.
  4. Full sweep — baseline + single-section ablation for every section.
  5. Greedy backward elimination — find a lean multi-section configuration.
  6. Interaction effects check — compare predicted vs. measured lean quality.
  7. Ordering experiments — test top-N sections at 3 positions (Phase 5).

Critical rules:
  - All Bedrock calls go through BedrockClient.invoke() — never directly.
  - Individual experiment failures are caught, logged, and skipped (not fatal).
  - Per-call token usage is tracked cumulatively via the client's own counters.
  - Progress is reported via queue.Queue — never via st.* from background threads.
"""

from __future__ import annotations

import logging
import queue
from pathlib import Path

import numpy as np
import yaml

from core.assembler import assemble_api_call
from core.analyzer import (
    build_pareto_candidates,
    compute_pareto_frontier,
    compute_section_impact,
    rank_sections,
)
from core.models import (
    AblationResults,
    ContextPayload,
    ExperimentConfig,
    ScoringResult,
    SectionImpact,
)
from core.quality_scorer import score_response
from core.redundancy import detect_redundancy
from infra.bedrock_client import BedrockClient

logger = logging.getLogger(__name__)

# Max tokens for the response generation call (not scoring).
# 4096 gives Nova room for a substantive answer on large contexts.
_RESPONSE_MAX_TOKENS = 4096

# Path to config.yaml, resolved relative to this file's location.
_CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"


# ── Phase 3: core single-section functions ────────────────────────────────────


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
        client:      BedrockClient instance.
        payload:     Parsed context payload.
        section_id:  ID of the section to exclude from all experiments.
        tiers:       Reasoning tiers to evaluate.
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


# ── Phase 4: config loading helpers ──────────────────────────────────────────


def _load_config() -> dict:
    """Load config.yaml from the project root."""
    with open(_CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def _load_thresholds() -> dict:
    """Return classification thresholds dict from config.yaml."""
    return _load_config().get("thresholds", {})


def _get_mode_config(mode: str) -> dict:
    """Return mode-specific config dict for the given mode name."""
    return _load_config().get("modes", {}).get(mode, {})


# ── Phase 4: progress queue helper ───────────────────────────────────────────


def _send_progress(q: queue.Queue | None, msg: dict) -> None:
    """Put a progress message on the queue if one is provided.

    Worker threads call this instead of st.* to comply with the Streamlit
    threading constraint (never call st.* from background threads).

    Args:
        q:   Optional queue.Queue from st.session_state["experiment_queue"].
        msg: Dict with at least a "type" key describing the event.
    """
    if q is not None:
        q.put(msg)


# ── Phase 4: multi-section exclusion helper ───────────────────────────────────


def _run_multi_exclusion(
    client:      BedrockClient,
    payload:     ContextPayload,
    exclude_ids: set[str],
    tiers:       list[str],
    num_queries: int | None = None,
) -> dict[str, dict[int, ScoringResult]]:
    """Score the context with an arbitrary set of sections excluded.

    Equivalent to run_single_ablation() but removes a set of section IDs
    rather than just one. Delegates to assemble_api_call which already
    accepts exclude_ids as a set[str].

    Args:
        client:      BedrockClient instance.
        payload:     Full context payload.
        exclude_ids: Set of section IDs to omit from every assembled context.
        tiers:       Reasoning tiers to evaluate.
        num_queries: Max queries to use. None = use all.

    Returns:
        {tier: {query_idx: ScoringResult}} for the reduced configuration.
    """
    queries = payload.evaluation_queries[:num_queries] if num_queries else payload.evaluation_queries
    scores: dict[str, dict[int, ScoringResult]] = {}

    for tier in tiers:
        scores[tier] = {}
        for q_idx, eval_query in enumerate(queries):
            try:
                api_params = assemble_api_call(
                    payload.sections,
                    eval_query.query,
                    exclude_ids=exclude_ids,
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
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Multi-exclusion failed [exclude=%s, tier=%s, query=%d]: %s",
                    exclude_ids, tier, q_idx, exc,
                )

    return scores


# ── Phase 4: full sweep orchestrator ─────────────────────────────────────────


def run_full_sweep(
    client:         BedrockClient,
    payload:        ContextPayload,
    config:         ExperimentConfig,
    progress_queue: queue.Queue | None = None,
) -> AblationResults:
    """Orchestrate a full ablation sweep and return assembled AblationResults.

    Executes the complete pipeline:
      1. Baseline evaluation (×repetitions)
      2. Single-section ablation for every section (×repetitions)
      3. Statistical analysis: compute and rank SectionImpact per section
      4. Redundancy detection via TF-IDF
      5. Greedy backward elimination (if mode enables run_multi_section=true)
         or static derivation from classifications (demo mode)
      6. Pareto frontier computation
      7. Assemble AblationResults with cumulative client usage stats

    Progress messages are put on progress_queue (if provided) — never calls
    st.* directly (Streamlit threading constraint).

    Args:
        client:         BedrockClient instance (shared across the pipeline).
        payload:        Parsed ContextPayload.
        config:         ExperimentConfig with mode, tiers, repetitions, tolerances.
        progress_queue: Optional Queue for Streamlit progress polling.

    Returns:
        Fully populated AblationResults.
    """
    mode_cfg       = _get_mode_config(config.mode.value)
    thresholds     = _load_thresholds()
    tiers          = config.reasoning_tiers
    num_queries    = mode_cfg.get("num_queries")
    repetitions    = config.repetitions
    run_multi      = mode_cfg.get("run_multi_section", False)
    run_ordering   = mode_cfg.get("run_ordering", False)

    n_sections  = len(payload.sections)
    # Total progress steps: (1 baseline + n_sections ablations) × repetitions
    total_steps = (1 + n_sections) * repetitions
    _send_progress(progress_queue, {"type": "start", "total": total_steps})

    # ── 1. Baseline ───────────────────────────────────────────────────────────
    baseline_all_reps: list[dict[str, dict[int, ScoringResult]]] = []
    for rep in range(repetitions):
        rep_scores = run_baseline(client, payload, tiers=tiers, num_queries=num_queries)
        baseline_all_reps.append(rep_scores)
        _send_progress(
            progress_queue,
            {"type": "baseline_rep_complete", "rep": rep + 1, "total_reps": repetitions},
        )

    baseline_scores = _merge_rep_scores(baseline_all_reps)
    _send_progress(
        progress_queue,
        {"type": "baseline_complete", "completed": repetitions, "total": total_steps},
    )

    # ── 2. Single-section ablation sweep ─────────────────────────────────────
    # {section_id: merged ablated scores across repetitions}
    section_ablation_scores: dict[str, dict[str, dict[int, ScoringResult]]] = {}

    for s_idx, section in enumerate(payload.sections):
        ablation_reps: list[dict[str, dict[int, ScoringResult]]] = []
        for rep in range(repetitions):
            try:
                rep_ablated = run_single_ablation(
                    client=client,
                    payload=payload,
                    section_id=section.id,
                    tiers=tiers,
                    num_queries=num_queries,
                )
                ablation_reps.append(rep_ablated)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Full sweep: ablation failed [section=%s, rep=%d]: %s",
                    section.id, rep + 1, exc,
                )
                _send_progress(
                    progress_queue,
                    {"type": "error", "section_id": section.id, "message": str(exc)},
                )

        if ablation_reps:
            section_ablation_scores[section.id] = _merge_rep_scores(ablation_reps)

        completed = repetitions + (s_idx + 1) * repetitions
        _send_progress(
            progress_queue,
            {
                "type":       "section_complete",
                "section_id": section.id,
                "completed":  completed,
                "total":      total_steps,
            },
        )

    _send_progress(progress_queue, {"type": "sweep_complete"})

    # ── 3. Statistical analysis ───────────────────────────────────────────────
    section_impacts: list[SectionImpact] = []
    for section in payload.sections:
        if section.id not in section_ablation_scores:
            continue
        impact = compute_section_impact(
            baseline_scores=baseline_scores,
            ablated_scores=section_ablation_scores[section.id],
            section=section,
            thresholds=thresholds,
        )
        section_impacts.append(impact)

    section_impacts = rank_sections(section_impacts)

    # ── 4. Redundancy detection ───────────────────────────────────────────────
    redundancy_clusters = detect_redundancy(
        payload.sections,
        threshold=config.redundancy_threshold,
    )

    # ── 5. Lean configuration ─────────────────────────────────────────────────
    if run_multi and section_impacts:
        lean_config, lean_retention, lean_reduction = run_greedy_elimination(
            client=client,
            payload=payload,
            impacts=section_impacts,
            config=config,
            baseline_scores=baseline_scores,
            progress_queue=progress_queue,
        )
    else:
        # Demo mode (run_multi_section=false): derive lean config from impact
        # classifications without additional API calls.
        lean_config, lean_retention, lean_reduction = _derive_lean_from_impacts(
            payload=payload,
            impacts=section_impacts,
            baseline_scores=baseline_scores,
            quality_tolerance=config.quality_tolerance,
        )

    # ── 6. Pareto frontier ────────────────────────────────────────────────────
    baseline_quality = _compute_avg(baseline_scores)
    pareto_configs   = build_pareto_candidates(payload, section_impacts, baseline_quality)
    pareto_frontier  = compute_pareto_frontier(pareto_configs)

    # ── 7. Ordering experiments (Full mode only) ──────────────────────────────
    ordering_recs: list[dict] = []
    if run_ordering and section_impacts:
        ordering_recs = run_ordering_experiments(
            client=client,
            payload=payload,
            impacts=section_impacts,
            config=config,
            baseline_scores=baseline_scores,
            progress_queue=progress_queue,
        )

    # ── 8. Assemble AblationResults ───────────────────────────────────────────
    _send_progress(progress_queue, {"type": "done"})

    return AblationResults(
        baseline_scores=_serialize_baseline(baseline_scores),
        section_impacts=section_impacts,
        lean_configuration=lean_config,
        lean_quality_retention=lean_retention,
        lean_token_reduction=lean_reduction,
        ordering_recommendations=ordering_recs,
        redundancy_clusters=redundancy_clusters,
        pareto_configurations=pareto_frontier,
        total_api_calls=client.total_api_calls,
        total_input_tokens=client.total_input_tokens,
        total_output_tokens=client.total_output_tokens,
        total_cost=client.total_cost,
    )


# ── Phase 5: ordering experiments ────────────────────────────────────────────


def run_ordering_experiments(
    client:          BedrockClient,
    payload:         ContextPayload,
    impacts:         list[SectionImpact],
    config:          ExperimentConfig,
    baseline_scores: dict[str, dict[int, ScoringResult]],
    progress_queue:  queue.Queue | None = None,
    top_n:           int = 5,
) -> list[dict]:
    """Test the effect of section ordering on response quality.

    For the top ``top_n`` non-system sections by avg_quality_delta, evaluates
    placing each section at 3 positions (start, middle, end) of the non-system
    section list and measures quality delta relative to baseline at each position.

    Only system_prompt sections are excluded from ordering candidates because
    they are always routed to the Converse API ``system`` field by the assembler
    and cannot be repositioned.

    API cost: top_n × 3 positions × num_queries calls (uses first tier only).

    Args:
        client:          BedrockClient instance.
        payload:         Full context payload.
        impacts:         Ranked SectionImpact list (from rank_sections).
        config:          ExperimentConfig with reasoning_tiers and quality_tolerance.
        baseline_scores: {tier: {q_idx: ScoringResult}} from run_baseline().
        progress_queue:  Optional Queue for Streamlit progress reporting.
        top_n:           Number of top sections to test (default 5).

    Returns:
        List of ordering result dicts, one per candidate section::

            [
                {
                    "section_id":    str,
                    "label":         str,
                    "best_position": str,   # "start" | "middle" | "end"
                    "quality_deltas": {     # delta vs. baseline per position
                        "start":  float,
                        "middle": float,
                        "end":    float,
                    },
                    "quality_gain": float,  # best delta minus worst delta
                },
                ...
            ]
    """
    mode_cfg    = _get_mode_config(config.mode.value)
    num_queries = mode_cfg.get("num_queries")
    # Use only the first tier to minimise ordering experiment API cost.
    tiers       = config.reasoning_tiers[:1]

    # Non-system sections in their original payload order — these are reorderable.
    from core.models import SectionType  # local import to avoid circular at module level
    non_system_ids = [
        s.id for s in payload.sections
        if s.section_type != SectionType.SYSTEM_PROMPT
    ]
    n_non_sys = len(non_system_ids)

    # Candidates: top_n ranked non-system sections by avg_quality_delta.
    candidates = [
        imp for imp in impacts
        if imp.section_type != SectionType.SYSTEM_PROMPT.value
    ][:top_n]

    if not candidates:
        logger.info("Ordering experiments: no non-system candidates found — skipping.")
        return []

    baseline_quality = _compute_avg(baseline_scores)
    results: list[dict] = []

    _send_progress(
        progress_queue,
        {"type": "ordering_start", "candidates": len(candidates)},
    )

    for cand_idx, cand in enumerate(candidates):
        section_id = cand.section_id
        position_deltas: dict[str, float] = {}

        # Build the 3 orderings: place section at start, middle, or end of
        # the non-system section list. Remaining sections keep their original
        # relative order.
        remaining_ids = [sid for sid in non_system_ids if sid != section_id]
        mid_idx       = max(0, n_non_sys // 2 - 1)

        positions: dict[str, list[str]] = {
            "start":  [section_id] + remaining_ids,
            "middle": remaining_ids[:mid_idx] + [section_id] + remaining_ids[mid_idx:],
            "end":    remaining_ids + [section_id],
        }

        for pos_name, ordering in positions.items():
            queries = (
                payload.evaluation_queries[:num_queries]
                if num_queries
                else payload.evaluation_queries
            )
            pos_scores: dict[str, dict[int, ScoringResult]] = {}

            for tier in tiers:
                pos_scores[tier] = {}
                for q_idx, eval_query in enumerate(queries):
                    try:
                        api_params = assemble_api_call(
                            payload.sections,
                            eval_query.query,
                            ordering=ordering,
                        )
                        response_text, _, _usage = client.invoke(
                            system=api_params["system"],
                            messages=api_params["messages"],
                            reasoning_tier=tier,
                            max_tokens=_RESPONSE_MAX_TOKENS,
                        )
                        score_result, _score_usage = score_response(
                            client=client,
                            query=eval_query.query,
                            response_text=response_text,
                            reference_answer=eval_query.reference_answer,
                            criteria=payload.quality_criteria,
                        )
                        pos_scores[tier][q_idx] = score_result
                        logger.info(
                            "Ordering [section=%s, pos=%s, tier=%s, q=%d] avg=%.2f",
                            section_id, pos_name, tier, q_idx, score_result.avg_score(),
                        )
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            "Ordering experiment failed [section=%s, pos=%s, tier=%s, q=%d]: %s",
                            section_id, pos_name, tier, q_idx, exc,
                        )

            pos_quality = _compute_avg(pos_scores)
            # Delta: positive = this position is better than baseline.
            position_deltas[pos_name] = pos_quality - baseline_quality

        best_position = max(position_deltas, key=lambda p: position_deltas[p])
        delta_values  = list(position_deltas.values())
        quality_gain  = max(delta_values) - min(delta_values) if delta_values else 0.0

        results.append({
            "section_id":     section_id,
            "label":          cand.label,
            "best_position":  best_position,
            "quality_deltas": position_deltas,
            "quality_gain":   quality_gain,
        })

        _send_progress(
            progress_queue,
            {
                "type":          "ordering_progress",
                "section_id":    section_id,
                "completed":     cand_idx + 1,
                "total":         len(candidates),
                "best_position": best_position,
                "quality_gain":  quality_gain,
            },
        )
        logger.info(
            "Ordering complete [section=%s]: best=%s, gain=%.3f",
            section_id, best_position, quality_gain,
        )

    _send_progress(progress_queue, {"type": "ordering_complete", "results": len(results)})
    return results


# ── Phase 4: greedy backward elimination ─────────────────────────────────────


def run_greedy_elimination(
    client:          BedrockClient,
    payload:         ContextPayload,
    impacts:         list[SectionImpact],
    config:          ExperimentConfig,
    baseline_scores: dict[str, dict[int, ScoringResult]],
    progress_queue:  queue.Queue | None = None,
) -> tuple[list[str], float, float]:
    """Greedy backward elimination to find a lean context configuration.

    Evaluates each removable/harmful section via a real API call and permanently
    removes it when the cumulative quality loss stays within the tolerance
    threshold. Sections are ordered by token_count descending to maximise
    token savings per elimination step.

    Args:
        client:          BedrockClient instance.
        payload:         Full context payload.
        impacts:         Ranked SectionImpact list from the single-section sweep.
        config:          ExperimentConfig with quality_tolerance and reasoning_tiers.
        baseline_scores: {tier: {q_idx: ScoringResult}} from run_baseline().
        progress_queue:  Optional Queue for Streamlit progress reporting.

    Returns:
        (lean_config, lean_retention, lean_reduction)
        lean_config:    Ordered list of section IDs kept in the lean context.
        lean_retention: Fraction of baseline quality preserved (e.g. 0.97).
        lean_reduction: Fraction of tokens removed (e.g. 0.58).
    """
    mode_cfg    = _get_mode_config(config.mode.value)
    num_queries = mode_cfg.get("num_queries")
    tiers       = config.reasoning_tiers[:1]  # First tier only to save API costs

    baseline_quality = _compute_avg(baseline_scores)

    # Candidates: removable + harmful sections, sorted by token_count descending
    # (largest first for maximum token savings per accepted removal)
    candidates = [
        imp for imp in impacts
        if imp.classification in ("removable", "harmful")
    ]
    candidates.sort(key=lambda x: x.token_count, reverse=True)

    excluded_ids: set[str] = set()
    _send_progress(
        progress_queue,
        {"type": "elimination_start", "candidates": len(candidates)},
    )

    for cand in candidates:
        tentative = excluded_ids | {cand.section_id}
        try:
            lean_scores      = _run_multi_exclusion(
                client=client,
                payload=payload,
                exclude_ids=tentative,
                tiers=tiers,
                num_queries=num_queries,
            )
            lean_quality     = _compute_avg(lean_scores)
            quality_loss_pct = (
                (baseline_quality - lean_quality) / baseline_quality
                if baseline_quality > 0 else 0.0
            )
            if quality_loss_pct <= config.quality_tolerance:
                excluded_ids = tentative
                logger.info(
                    "Greedy: accepted removal of '%s' (loss=%.3f ≤ tol=%.3f)",
                    cand.section_id, quality_loss_pct, config.quality_tolerance,
                )
            else:
                logger.info(
                    "Greedy: rejected removal of '%s' (loss=%.3f > tol=%.3f)",
                    cand.section_id, quality_loss_pct, config.quality_tolerance,
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Greedy: evaluation failed for '%s': %s — skipping.",
                cand.section_id, exc,
            )

    # Build lean config preserving original section order
    lean_config  = [s.id for s in payload.sections if s.id not in excluded_ids]
    total_tokens = payload.total_tokens
    lean_tokens  = sum(s.token_count for s in payload.sections if s.id not in excluded_ids)
    lean_reduction = 1.0 - lean_tokens / total_tokens if total_tokens > 0 else 0.0

    # Final lean quality measurement
    if excluded_ids:
        try:
            final_scores = _run_multi_exclusion(
                client=client,
                payload=payload,
                exclude_ids=excluded_ids,
                tiers=tiers,
                num_queries=num_queries,
            )
            lean_quality = _compute_avg(final_scores)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Greedy: final lean evaluation failed: %s", exc)
            lean_quality = baseline_quality * (1.0 - lean_reduction)

        lean_retention = lean_quality / baseline_quality if baseline_quality > 0 else 1.0
        lean_retention = max(0.0, min(1.0, lean_retention))
    else:
        lean_retention = 1.0

    _send_progress(
        progress_queue,
        {
            "type":           "elimination_complete",
            "lean_config":    lean_config,
            "lean_retention": lean_retention,
            "lean_reduction": lean_reduction,
        },
    )
    logger.info(
        "Greedy elimination complete: removed %d sections, retention=%.3f, reduction=%.3f",
        len(excluded_ids), lean_retention, lean_reduction,
    )

    return lean_config, lean_retention, lean_reduction


# ── Phase 4: interaction effects check ───────────────────────────────────────


def check_interaction_effects(
    lean_quality:      float,
    individual_deltas: list[float],
    excluded_ids:      list[str],
    baseline_quality:  float,
) -> dict:
    """Compare measured lean-set quality against the sum-of-individual-deltas prediction.

    Flags a significant interaction effect when the gap between the measured and
    the predicted lean quality exceeds 0.5 points on the 1–10 scale.

    Args:
        lean_quality:      Measured quality of the lean configuration.
        individual_deltas: Per-section avg_quality_delta values for excluded sections.
        excluded_ids:      IDs of the sections that were excluded.
        baseline_quality:  Measured quality of the full context.

    Returns:
        Dict with keys:
          "measured_quality":  float — actual lean quality.
          "predicted_quality": float — baseline minus sum of individual deltas.
          "gap":               float — (measured − predicted), positive = better.
          "interaction_flag":  bool  — True if |gap| > 0.5.
          "excluded_count":    int   — number of excluded sections.
    """
    predicted_quality = baseline_quality - sum(individual_deltas)
    gap               = lean_quality - predicted_quality
    return {
        "measured_quality":  lean_quality,
        "predicted_quality": predicted_quality,
        "gap":               gap,
        "interaction_flag":  abs(gap) > 0.5,
        "excluded_count":    len(excluded_ids),
    }


# ── Private helpers ───────────────────────────────────────────────────────────


def _merge_rep_scores(
    reps: list[dict[str, dict[int, ScoringResult]]],
) -> dict[str, dict[int, ScoringResult]]:
    """Merge multiple repetition score dicts into one.

    For each (tier, query_idx) pair, keeps the ScoringResult from the first
    successful repetition. This is sufficient for Phase 4; Phase 4 confidence
    intervals are computed from per-query deltas within a single rep rather
    than from cross-rep variance.

    Args:
        reps: List of {tier: {q_idx: ScoringResult}} dicts, one per repetition.

    Returns:
        Merged {tier: {q_idx: ScoringResult}}.
    """
    if not reps:
        return {}
    if len(reps) == 1:
        return reps[0]

    all_tiers = {tier for rep in reps for tier in rep}
    merged: dict[str, dict[int, ScoringResult]] = {}
    for tier in all_tiers:
        merged[tier] = {}
        all_q_idxs = {
            q_idx
            for rep in reps
            if tier in rep
            for q_idx in rep[tier]
        }
        for q_idx in all_q_idxs:
            for rep in reps:
                if tier in rep and q_idx in rep[tier]:
                    merged[tier][q_idx] = rep[tier][q_idx]
                    break

    return merged


def _compute_avg(scores: dict[str, dict[int, ScoringResult]]) -> float:
    """Compute mean avg_score() across all tiers and queries in a score dict."""
    all_scores = [
        result.avg_score()
        for tier_scores in scores.values()
        for result in tier_scores.values()
    ]
    return float(np.mean(all_scores)) if all_scores else 0.0


def _derive_lean_from_impacts(
    payload:           ContextPayload,
    impacts:           list[SectionImpact],
    baseline_scores:   dict[str, dict[int, ScoringResult]],
    quality_tolerance: float,
) -> tuple[list[str], float, float]:
    """Derive a lean config from section classifications without extra API calls.

    Used when run_multi_section=false (demo mode). Greedily removes harmful
    sections first (they improve quality), then removable sections by token_count
    descending, stopping when the cumulative projected quality drop would exceed
    the tolerance. No additional API calls are made.

    Returns:
        (lean_config, lean_retention, lean_reduction)
    """
    baseline_quality = _compute_avg(baseline_scores)
    max_drop         = quality_tolerance * baseline_quality

    # Harmful first (negative delta = quality improvement), then removable largest first
    candidates = sorted(
        [imp for imp in impacts if imp.classification in ("removable", "harmful")],
        key=lambda x: (0 if x.avg_quality_delta < 0 else 1, -x.token_count),
    )

    excluded_ids:    set[str] = set()
    cumulative_delta = 0.0

    for cand in candidates:
        projected = cumulative_delta + cand.avg_quality_delta
        if projected <= max_drop:
            excluded_ids.add(cand.section_id)
            cumulative_delta = projected

    lean_config    = [s.id for s in payload.sections if s.id not in excluded_ids]
    total_tokens   = payload.total_tokens
    lean_tokens    = sum(s.token_count for s in payload.sections if s.id not in excluded_ids)
    lean_reduction = 1.0 - lean_tokens / total_tokens if total_tokens > 0 else 0.0

    if baseline_quality > 0:
        lean_quality   = max(0.0, baseline_quality - cumulative_delta)
        lean_retention = max(0.0, min(1.0, lean_quality / baseline_quality))
    else:
        lean_retention = 1.0

    return lean_config, lean_retention, lean_reduction


def _serialize_baseline(
    scores: dict[str, dict[int, ScoringResult]],
) -> dict:
    """Serialize {tier: {q_idx: ScoringResult}} to JSON-safe plain dicts.

    JSON object keys must be strings; q_idx (int) is converted to str.
    ScoringResult is serialised via model_dump() for AblationResults storage.
    """
    return {
        tier: {str(q_idx): sr.model_dump() for q_idx, sr in tier_scores.items()}
        for tier, tier_scores in scores.items()
    }
