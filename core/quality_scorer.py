"""
core/quality_scorer.py — LLM-as-judge response quality scoring.

Uses Amazon Nova with medium reasoning to score a model response against
a set of quality criteria, returning a validated ScoringResult.

Prompt templates are taken verbatim from context/prompt-templates.md.
Call pattern: reasoning_tier="medium", temperature=0, max_tokens=8000.

Critical rules:
  - All Bedrock calls go through BedrockClient.invoke() — never call
    bedrock.converse() directly.
  - Always parse LLM output with parse_llm_json() — never raw json.loads().
  - Token counts come from usage returned by invoke() — not tiktoken.
"""

from __future__ import annotations

import json
import logging

from infra.bedrock_client import BedrockClient
from infra.json_parser import parse_llm_json
from core.models import ScoringResult

logger = logging.getLogger(__name__)

# ── Prompt constants (from context/prompt-templates.md) ──────────────────────

SCORING_SYSTEM_PROMPT = """\
You are a rigorous and consistent response quality evaluator. Your task is to score AI-generated responses against a set of quality criteria on a 1-10 integer scale.

Scoring guidelines:
- Be consistent: identical response quality must receive identical scores across evaluations.
- Be discriminating: avoid clustering all scores at 7-8. Reserve 9-10 for exceptional responses, 1-2 for completely inadequate ones.
- Base scores strictly on the provided context — do not use outside knowledge to fill gaps.
- One-sentence justifications must reference specific evidence from the response.

Return ONLY valid JSON. No markdown, no explanation, no preamble."""

SCORING_USER_TEMPLATE = """\
QUERY:
{query}

RESPONSE TO EVALUATE:
{response_text}
{reference_block}\
Score each criterion on an integer scale from 1 (very poor) to 10 (excellent), with a one-sentence justification citing specific evidence.

Criteria definitions:
- relevance: Does the response directly address the query?
- accuracy: Are factual claims correct relative to the provided context?
- completeness: Does the response cover all key aspects the query requires?
- groundedness: Is the response grounded in the provided context rather than fabricated?
{extra_criteria}\
Return ONLY a valid JSON object matching this schema. No markdown fences, no explanation, no preamble.
Schema: {schema}
"""

# Inserted when a reference answer is provided
_REFERENCE_BLOCK_TEMPLATE = "REFERENCE ANSWER (ground truth):\n{reference_answer}\n\n"

# Schema shown to the LLM — matches ScoringResult field structure
_SCORING_SCHEMA = json.dumps(
    {
        "relevance":    {"score": "int 1-10", "justification": "str"},
        "accuracy":     {"score": "int 1-10", "justification": "str"},
        "completeness": {"score": "int 1-10", "justification": "str"},
        "groundedness": {"score": "int 1-10", "justification": "str"},
    },
    indent=2,
)

# Default criteria descriptions (used for extra criteria beyond the 4 defaults)
_DEFAULT_CRITERIA = {"relevance", "accuracy", "completeness", "groundedness"}


# ── Public interface ──────────────────────────────────────────────────────────


def score_response(
    client: BedrockClient,
    query: str,
    response_text: str,
    reference_answer: str | None = None,
    criteria: list[str] | None = None,
) -> tuple[ScoringResult, dict]:
    """Score a model response using LLM-as-judge via Nova medium reasoning.

    Args:
        client:           BedrockClient instance (shared across the pipeline).
        query:            The evaluation query that was answered.
        response_text:    The model's response text to evaluate.
        reference_answer: Optional ground-truth answer for comparison.
        criteria:         Quality criteria list. If None or omitting defaults,
                          uses the standard 4: relevance, accuracy,
                          completeness, groundedness. Extra criteria beyond
                          the default 4 are appended to the prompt.

    Returns:
        result (ScoringResult): Validated per-criterion scores with
                                justifications.
        usage (dict):           API-reported token usage for this scoring
                                call — {"input_tokens", "output_tokens",
                                "total_tokens"}.

    Raises:
        ValueError: If the LLM output cannot be parsed into a ScoringResult
                    after all fallback strategies are exhausted.
    """
    scoring_prompt = _build_scoring_prompt(
        query=query,
        response_text=response_text,
        reference_answer=reference_answer,
        criteria=criteria,
    )

    # max_tokens=8000: medium reasoning tokens consume the output budget first,
    # leaving too little room for the JSON output at 4000. 8000 gives ample
    # headroom for both reasoning tokens and the ~200-token scoring JSON.
    # See CLAUDE.md gotchas: "medium reasoning can consume entire output budget".
    text, _, usage = client.invoke(
        system=SCORING_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": [{"text": scoring_prompt}]}],
        reasoning_tier="medium",
        max_tokens=8000,
        temperature=0,
    )

    result = parse_llm_json(text, ScoringResult)
    logger.debug(
        "Scored response for query '%s...' → avg=%.2f",
        query[:60],
        result.avg_score(),
    )
    return result, usage


# ── Private helpers ───────────────────────────────────────────────────────────


def _build_scoring_prompt(
    query: str,
    response_text: str,
    reference_answer: str | None,
    criteria: list[str] | None,
) -> str:
    """Construct the scoring user prompt from template."""
    reference_block = ""
    if reference_answer:
        reference_block = _REFERENCE_BLOCK_TEMPLATE.format(
            reference_answer=reference_answer
        )

    extra_criteria = ""
    if criteria:
        for name in criteria:
            if name not in _DEFAULT_CRITERIA:
                extra_criteria += f"- {name}: Evaluate the response on the '{name}' dimension.\n"

    return SCORING_USER_TEMPLATE.format(
        query=query,
        response_text=response_text,
        reference_block=reference_block,
        extra_criteria=extra_criteria,
        schema=_SCORING_SCHEMA,
    )
