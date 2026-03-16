"""
core/diet_planner.py — Context Diet Plan generator.

Uses Nova extended thinking (HIGH tier) to produce a natural-language
optimization guide from ablation results. The plan is structured, prioritised,
and quantified — it names specific sections to remove, projects token and cost
savings, and explains each recommendation with ablation evidence.

Critical rules:
  - HIGH tier: do NOT pass max_tokens, temperature, topP, or topK.
    Passing any of these raises ValidationException.
  - All Bedrock calls go through BedrockClient.invoke().
"""

from __future__ import annotations

import logging

from core.models import AblationResults, ContextPayload, SectionImpact
from infra.bedrock_client import BedrockClient

logger = logging.getLogger(__name__)

# ── Pricing constants ────────────────────────────────────────────────────────
_INPUT_PRICE_PER_M  = 0.30   # USD per 1M input tokens
_OUTPUT_PRICE_PER_M = 2.50   # USD per 1M output tokens (includes reasoning)


# ── Prompt templates ─────────────────────────────────────────────────────────

_DIET_PLANNER_SYSTEM_PROMPT = """\
You are an expert LLM context engineering consultant. You analyze the results of \
context ablation experiments and produce clear, actionable optimization recommendations.

Your recommendations must:
1. Be specific and actionable — name exact sections to remove, trim, or reorder.
2. Quantify projected savings — token counts, cost reductions at provided pricing.
3. Preserve quality — only recommend removals within the stated quality tolerance.
4. Explain reasoning — reference specific ablation findings (quality deltas, tier \
sensitivity) to support each recommendation.
5. Prioritize by ROI — lead with the highest token-savings recommendations that \
have the lowest quality cost.

Frame all findings as behavioral observations from the ablation data. Do not make \
mechanistic claims about how the model processes context internally."""

_DIET_PLAN_USER_TEMPLATE = """\
Here are the ablation results for a context optimization analysis. Produce a \
"Context Diet Plan" — a structured, prioritized set of recommendations for \
optimizing this context payload.

ABLATION SUMMARY:
- Baseline context: {total_tokens:,} tokens | ${baseline_cost_per_call:.4f} per call
- Lean configuration: {lean_tokens:,} tokens | ${lean_cost_per_call:.4f} per call
- Quality retention: {quality_retention:.1%} of baseline
- Token reduction: {token_reduction:.1%}

SECTION ANALYSIS (ranked by avg quality delta when removed):
{section_table}

REDUNDANCY CLUSTERS:
{redundancy_summary}

ORDERING FINDINGS:
{ordering_summary}

PARETO-OPTIMAL CONFIGURATIONS:
{pareto_summary}

PRICING REFERENCE:
- Input: $0.30 / 1M tokens
- Output: $2.50 / 1M tokens
- Quality tolerance: {quality_tolerance:.0%} max loss

Produce the Context Diet Plan with these sections:
1. Executive Summary (2-3 sentences: before vs. after, key savings)
2. Priority Recommendations (ordered by ROI, each with: action, projected savings, \
quality impact, rationale)
3. Redundancy Consolidations (which sections overlap and how to merge)
4. Ordering Optimizations (which sections to move and why)
5. What NOT to Remove (essential sections and why they matter)
6. Implementation Checklist (step-by-step actions the developer should take)
"""


# ── Public interface ──────────────────────────────────────────────────────────


def generate_diet_plan(
    results: AblationResults,
    payload: ContextPayload,
    client:  BedrockClient,
) -> str:
    """Generate a Context Diet Plan using Nova extended thinking (HIGH).

    Builds a structured prompt from the ablation results and calls Nova with
    HIGH reasoning tier to produce a natural-language optimization guide.

    Args:
        results: Fully populated AblationResults from the ablation sweep.
        payload: Original ContextPayload (used for total token counts).
        client:  BedrockClient instance — must be initialised before calling.

    Returns:
        Markdown-formatted diet plan string from Nova.
        Falls back to a locally-generated summary if the API call fails.
    """
    prompt = _build_diet_plan_prompt(results, payload)

    try:
        text, _, usage = client.invoke(
            system=_DIET_PLANNER_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            reasoning_tier="high",
            # HIGH tier: no max_tokens, no temperature — model manages output length.
        )
        logger.info(
            "Diet plan generated — input_tokens=%d, output_tokens=%d",
            usage.get("input_tokens", 0),
            usage.get("output_tokens", 0),
        )
        return text
    except Exception as exc:  # noqa: BLE001
        logger.warning("Diet plan API call failed: %s — returning fallback.", exc)
        return _generate_fallback_plan(results, payload)


# ── Prompt building helpers ───────────────────────────────────────────────────


def _build_diet_plan_prompt(results: AblationResults, payload: ContextPayload) -> str:
    """Populate the diet plan user prompt template with ablation data."""
    total_tokens = payload.total_tokens

    # Token counts for the lean configuration
    lean_section_ids = set(results.lean_configuration)
    lean_tokens = sum(
        s.token_count for s in payload.sections if s.id in lean_section_ids
    )

    # Cost projections: input tokens × price (output assumed proportional)
    baseline_cost  = _compute_input_cost(total_tokens)
    lean_cost      = _compute_input_cost(lean_tokens)

    return _DIET_PLAN_USER_TEMPLATE.format(
        total_tokens=total_tokens,
        baseline_cost_per_call=baseline_cost,
        lean_tokens=lean_tokens,
        lean_cost_per_call=lean_cost,
        quality_retention=results.lean_quality_retention,
        token_reduction=results.lean_token_reduction,
        section_table=_format_section_table(results.section_impacts),
        redundancy_summary=_format_redundancy_summary(results.redundancy_clusters),
        ordering_summary=_format_ordering_summary(results.ordering_recommendations),
        pareto_summary=_format_pareto_summary(results.pareto_configurations),
        quality_tolerance=0.05,  # from config default; hardcoded here for prompt clarity
    )


def _format_section_table(impacts: list[SectionImpact]) -> str:
    """Format section impacts as a markdown table, sorted by avg_quality_delta desc."""
    rows = [
        "| Section | Type | Tokens | Avg Δ | Classification |",
        "|---------|------|--------|-------|----------------|",
    ]
    sorted_impacts = sorted(impacts, key=lambda x: x.avg_quality_delta, reverse=True)
    for imp in sorted_impacts:
        rows.append(
            f"| {imp.label} | {imp.section_type} | {imp.token_count:,} "
            f"| {imp.avg_quality_delta:+.2f} | {imp.classification} |"
        )
    return "\n".join(rows) if len(rows) > 2 else "No section impact data available."


def _format_redundancy_summary(clusters: list[tuple]) -> str:
    """Format redundancy cluster tuples as a readable summary."""
    if not clusters:
        return "No redundancy clusters detected above the configured threshold."

    lines = [f"Detected {len(clusters)} redundant section pair(s):"]
    for id1, id2, similarity in clusters[:10]:   # cap at 10 for prompt length
        lines.append(f"  - {id1} ↔ {id2} (similarity: {similarity:.2f})")
    if len(clusters) > 10:
        lines.append(f"  ... and {len(clusters) - 10} more pairs.")
    return "\n".join(lines)


def _format_ordering_summary(recommendations: list[dict]) -> str:
    """Format ordering experiment results as a readable summary."""
    if not recommendations:
        return "No ordering experiments were run (Full mode only)."

    lines = [f"Tested {len(recommendations)} section(s) at 3 positions:"]
    for rec in recommendations:
        gain = rec.get("quality_gain", 0.0)
        best = rec.get("best_position", "unknown")
        label = rec.get("label", rec.get("section_id", "?"))
        deltas = rec.get("quality_deltas", {})
        delta_str = ", ".join(
            f"{pos}: {d:+.2f}" for pos, d in deltas.items()
        )
        lines.append(
            f"  - {label}: best position = {best} (gain={gain:.2f}; {delta_str})"
        )
    return "\n".join(lines)


def _format_pareto_summary(configurations: list[dict]) -> str:
    """Format Pareto-optimal configurations as a readable summary."""
    if not configurations:
        return "No Pareto-optimal configurations computed."

    lines = [f"{len(configurations)} Pareto-optimal configuration(s):"]
    for i, cfg in enumerate(configurations[:5], start=1):   # top 5
        quality = cfg.get("quality", 0.0)
        tokens  = cfg.get("tokens", 0)
        cost    = cfg.get("cost", _compute_input_cost(tokens))
        n_secs  = len(cfg.get("section_ids", []))
        lines.append(
            f"  {i}. {n_secs} sections | {tokens:,} tokens | "
            f"quality={quality:.2f} | ~${cost:.4f}/call"
        )
    if len(configurations) > 5:
        lines.append(f"  ... and {len(configurations) - 5} more.")
    return "\n".join(lines)


# ── Fallback plan ─────────────────────────────────────────────────────────────


def _generate_fallback_plan(results: AblationResults, payload: ContextPayload) -> str:
    """Generate a minimal local diet plan when the Nova API call fails."""
    removable = [
        imp for imp in results.section_impacts
        if imp.classification in ("removable", "harmful")
    ]
    essential = [
        imp for imp in results.section_impacts
        if imp.classification == "essential"
    ]
    saved_tokens = sum(imp.token_count for imp in removable)

    lines = [
        "# Context Diet Plan (Fallback — Nova API unavailable)\n",
        "## Executive Summary",
        f"This context payload contains {payload.total_tokens:,} tokens. "
        f"Based on ablation data, {len(removable)} section(s) can be removed, "
        f"saving approximately {saved_tokens:,} tokens "
        f"({saved_tokens / max(payload.total_tokens, 1):.1%} of total).",
        "",
        "## Priority Recommendations",
    ]
    for imp in sorted(removable, key=lambda x: x.token_count, reverse=True):
        lines.append(
            f"- **Remove `{imp.label}`** ({imp.token_count:,} tokens, "
            f"avg quality delta: {imp.avg_quality_delta:+.2f})"
        )

    lines += [
        "",
        "## What NOT to Remove",
    ]
    for imp in essential:
        lines.append(
            f"- **Keep `{imp.label}`** (avg quality delta: {imp.avg_quality_delta:+.2f} — essential)"
        )

    return "\n".join(lines)


# ── Cost helper ───────────────────────────────────────────────────────────────


def _compute_input_cost(tokens: int) -> float:
    """Compute USD cost for input tokens only (used for per-call estimates)."""
    return tokens * _INPUT_PRICE_PER_M / 1_000_000
