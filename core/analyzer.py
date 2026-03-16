"""
core/analyzer.py — Local statistical analysis of ablation experiment results.

All statistics are computed locally using numpy and scipy — no API calls.
Provides section impact ranking, classification, confidence intervals,
tier sensitivity scores, quality-per-token efficiency metrics, and Pareto
frontier computation.
"""

from __future__ import annotations

import logging

import numpy as np
import scipy.stats

from core.models import (
    ContextPayload,
    ContextSection,
    ScoringResult,
    SectionImpact,
)

logger = logging.getLogger(__name__)

# ── Classification thresholds ─────────────────────────────────────────────────
# Defaults are overridden by config.yaml thresholds when supplied by callers.
_DEFAULT_IMPACT_LOW  = 0.5   # avg_delta below this → "removable"
_DEFAULT_IMPACT_HIGH = 2.0   # avg_delta at or above this → "essential"


# ── Public interface ──────────────────────────────────────────────────────────


def compute_section_impact(
    baseline_scores: dict[str, dict[int, ScoringResult]],
    ablated_scores:  dict[str, dict[int, ScoringResult]],
    section:         ContextSection,
    thresholds:      dict,
) -> SectionImpact:
    """Compute a SectionImpact for a single ablated section.

    Calculates per-tier quality deltas, overall average delta, tier sensitivity
    (variance of per-tier deltas), section classification, and quality-per-token
    efficiency metric.

    Args:
        baseline_scores: {tier: {query_idx: ScoringResult}} from run_baseline().
        ablated_scores:  {tier: {query_idx: ScoringResult}} from run_single_ablation().
        section:         The ContextSection being evaluated.
        thresholds:      Dict with optional "section_impact_low" and
                         "section_impact_high" keys (from config.yaml). Uses
                         defaults 0.5 / 2.0 when keys are absent.

    Returns:
        Fully populated SectionImpact instance.
    """
    impact_low  = float(thresholds.get("section_impact_low",  _DEFAULT_IMPACT_LOW))
    impact_high = float(thresholds.get("section_impact_high", _DEFAULT_IMPACT_HIGH))

    quality_delta_by_tier: dict[str, float] = {}

    for tier in baseline_scores:
        if tier not in ablated_scores:
            continue
        tier_deltas: list[float] = []
        for q_idx in baseline_scores[tier]:
            if q_idx not in ablated_scores[tier]:
                continue
            b = baseline_scores[tier][q_idx].avg_score()
            a = ablated_scores[tier][q_idx].avg_score()
            tier_deltas.append(b - a)
        if tier_deltas:
            quality_delta_by_tier[tier] = float(np.mean(tier_deltas))

    all_deltas        = list(quality_delta_by_tier.values())
    avg_quality_delta = float(np.mean(all_deltas)) if all_deltas else 0.0
    # Tier sensitivity: variance of per-tier deltas (0.0 when ≤1 tier)
    tier_sensitivity  = float(np.var(all_deltas)) if len(all_deltas) > 1 else 0.0
    classification    = _classify(avg_quality_delta, impact_low, impact_high)
    quality_per_token = avg_quality_delta / max(section.token_count, 1)

    return SectionImpact(
        section_id=section.id,
        label=section.label,
        section_type=section.section_type.value,
        token_count=section.token_count,
        avg_quality_delta=avg_quality_delta,
        quality_delta_by_tier=quality_delta_by_tier,
        tier_sensitivity=tier_sensitivity,
        classification=classification,
        quality_per_token=quality_per_token,
    )


def rank_sections(
    impacts: list[SectionImpact],
) -> list[SectionImpact]:
    """Sort SectionImpact list by avg_quality_delta descending (most essential first).

    Args:
        impacts: List of SectionImpact objects.

    Returns:
        New list sorted by avg_quality_delta descending.
    """
    return sorted(impacts, key=lambda s: s.avg_quality_delta, reverse=True)


def compute_confidence_interval(
    values:     list[float],
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Compute a two-sided confidence interval for a list of measurements.

    Uses scipy.stats.sem (standard error of the mean) and the t-distribution,
    which is appropriate for small sample sizes (n < 30) common in ablation
    experiments where API cost limits the number of repetitions.

    Args:
        values:     List of numeric measurements (e.g. per-query quality deltas).
        confidence: Confidence level (default 0.95 for a 95% interval).

    Returns:
        (lower, upper) CI bounds.
        Returns (mean, mean) for empty or single-element lists (no spread to measure).
    """
    if len(values) <= 1:
        mean = float(np.mean(values)) if values else 0.0
        return mean, mean

    mean = float(np.mean(values))
    sem  = scipy.stats.sem(values)
    h    = sem * scipy.stats.t.ppf((1 + confidence) / 2, df=len(values) - 1)
    return float(mean - h), float(mean + h)


def build_pareto_candidates(
    payload:          ContextPayload,
    impacts:          list[SectionImpact],
    baseline_quality: float,
) -> list[dict]:
    """Generate Pareto candidate configurations by progressively removing sections.

    Starts from the full context and greedily removes sections from least to most
    impactful (ascending avg_quality_delta), producing a (quality, tokens) sweep
    through the design space suitable for Pareto frontier computation.

    Args:
        payload:          The full ContextPayload (for token counts and ordering).
        impacts:          SectionImpact list — ordering determines removal sequence.
        baseline_quality: Measured quality of the full context.

    Returns:
        List of dicts with keys "section_ids", "quality", "tokens",
        sorted by tokens ascending. Each dict represents one candidate config.
    """
    section_by_id = {s.id: s for s in payload.sections}
    total_tokens  = payload.total_tokens

    # Full-context starting point
    configs: list[dict] = [
        {
            "section_ids": [s.id for s in payload.sections],
            "quality":     baseline_quality,
            "tokens":      total_tokens,
        }
    ]

    # Sort least → most impactful so we remove the safest sections first
    sorted_impacts = sorted(impacts, key=lambda x: x.avg_quality_delta)

    excluded_ids:    set[str] = set()
    cumulative_delta = 0.0

    for impact in sorted_impacts:
        excluded_ids.add(impact.section_id)
        cumulative_delta += impact.avg_quality_delta
        remaining = [s.id for s in payload.sections if s.id not in excluded_ids]
        rem_tokens = sum(
            section_by_id[sid].token_count
            for sid in remaining
            if sid in section_by_id
        )
        configs.append({
            "section_ids": remaining,
            "quality":     max(0.0, baseline_quality - cumulative_delta),
            "tokens":      rem_tokens,
        })

    return sorted(configs, key=lambda x: x["tokens"])


def compute_pareto_frontier(
    configurations: list[dict],
) -> list[dict]:
    """Filter configurations to keep only Pareto-optimal ones.

    A configuration C is Pareto-optimal if no other D simultaneously has
    quality >= C.quality AND tokens <= C.tokens with strict improvement in ≥1
    dimension.

    Args:
        configurations: List of dicts with required keys "quality" (float) and
                        "tokens" (int). All other keys are preserved unchanged.

    Returns:
        Subset of non-dominated configurations, sorted by quality descending.
        Returns an empty list when input is empty.
    """
    if not configurations:
        return []

    pareto: list[dict] = []
    for c in configurations:
        dominated = False
        for d in configurations:
            if d is c:
                continue
            # d dominates c: weakly better in both dimensions, strictly in ≥1
            if d["quality"] >= c["quality"] and d["tokens"] <= c["tokens"]:
                if d["quality"] > c["quality"] or d["tokens"] < c["tokens"]:
                    dominated = True
                    break
        if not dominated:
            pareto.append(c)

    return sorted(pareto, key=lambda x: x["quality"], reverse=True)


# ── Private helpers ───────────────────────────────────────────────────────────


def _classify(
    avg_delta:   float,
    impact_low:  float,
    impact_high: float,
) -> str:
    """Map an average quality delta to a section classification string.

    Thresholds:
      - harmful:   avg_delta < 0                          (removing improves quality)
      - removable: 0 ≤ avg_delta < impact_low             (minimal impact, safe to remove)
      - moderate:  impact_low ≤ avg_delta < impact_high   (context-dependent)
      - essential: avg_delta ≥ impact_high                (significant loss if removed)
    """
    if avg_delta < 0:
        return "harmful"
    if avg_delta < impact_low:
        return "removable"
    if avg_delta < impact_high:
        return "moderate"
    return "essential"
