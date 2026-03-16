"""
tests/test_json_parser.py — Unit tests for infra/json_parser.py.

Tests the 4-strategy fallback chain against all documented edge cases.
No Bedrock API calls — purely local parsing logic.
"""

import pytest

from infra.json_parser import parse_llm_json
from core.models import CriterionScore, ScoringResult


# ── Helpers ────────────────────────────────────────────────────────────────────

def _valid_criterion_json(**overrides) -> dict:
    """Return a dict that satisfies CriterionScore fields."""
    base = {"score": 8, "justification": "The response is accurate and relevant."}
    base.update(overrides)
    return base


def _valid_scoring_result_dict() -> dict:
    """Return a dict that satisfies all ScoringResult fields."""
    criterion = {"score": 7, "justification": "Adequate coverage."}
    return {
        "relevance":    criterion,
        "accuracy":     criterion,
        "completeness": criterion,
        "groundedness": criterion,
    }


# ── Strategy 1: Clean JSON ─────────────────────────────────────────────────────

def test_clean_json_parses_successfully():
    """Strategy 1: direct json.loads() on valid JSON."""
    raw = '{"score": 9, "justification": "Excellent accuracy."}'
    result = parse_llm_json(raw, CriterionScore)
    assert result.score == 9
    assert result.justification == "Excellent accuracy."


# ── Strategy 2: JSON in code fences ───────────────────────────────────────────

def test_json_in_backtick_fences():
    """Strategy 2: extract JSON from ```json ... ``` fences."""
    raw = '```json\n{"score": 7, "justification": "Mostly correct."}\n```'
    result = parse_llm_json(raw, CriterionScore)
    assert result.score == 7


def test_json_in_plain_fences():
    """Strategy 2: extract JSON from plain ``` ... ``` fences (no language tag)."""
    raw = '```\n{"score": 5, "justification": "Partial answer."}\n```'
    result = parse_llm_json(raw, CriterionScore)
    assert result.score == 5


# ── Strategy 3/4: Preamble text and repair ────────────────────────────────────

def test_json_with_preamble_text():
    """Strategy 3/4: JSON preceded by explanatory text."""
    raw = 'Here is my evaluation:\n{"score": 6, "justification": "Acceptable."}'
    result = parse_llm_json(raw, CriterionScore)
    assert result.score == 6


def test_json_with_trailing_comma():
    """Strategy 3: json_repair handles trailing commas."""
    raw = '{"score": 8, "justification": "Good response.",}'
    result = parse_llm_json(raw, CriterionScore)
    assert result.score == 8


def test_json_with_single_quotes():
    """Strategy 3: json_repair handles single-quoted strings."""
    raw = "{'score': 4, 'justification': 'Incomplete answer.'}"
    result = parse_llm_json(raw, CriterionScore)
    assert result.score == 4


def test_json_with_python_booleans():
    """Strategy 3: json_repair handles Python-style True/False/None."""
    # Use ScoringResult which has nested dicts; embed Python booleans in preamble
    # Test that repair handles the canonical Python bool case
    raw = '{"score": 10, "justification": "Perfect.", "flagged": True}'
    # CriterionScore only has score/justification — extra fields are ignored by Pydantic v2
    result = parse_llm_json(raw, CriterionScore)
    assert result.score == 10


# ── Error cases ────────────────────────────────────────────────────────────────

def test_empty_string_raises_value_error():
    """All strategies fail on empty input — must raise ValueError."""
    with pytest.raises(ValueError, match="All JSON extraction strategies failed"):
        parse_llm_json("", CriterionScore)


def test_valid_json_failing_pydantic_validation_raises_value_error():
    """Valid JSON that doesn't match the model schema must raise ValueError."""
    # score is required; justification is required — omitting both should fail
    raw = '{"wrong_field": "unexpected"}'
    with pytest.raises(ValueError, match="All JSON extraction strategies failed"):
        parse_llm_json(raw, CriterionScore)


# ── Multi-field model (ScoringResult) ─────────────────────────────────────────

def test_nested_model_parses_correctly():
    """Validate that a more complex nested model (ScoringResult) parses end-to-end."""
    import json
    raw = json.dumps(_valid_scoring_result_dict())
    result = parse_llm_json(raw, ScoringResult)
    assert result.relevance.score == 7
    assert result.accuracy.justification == "Adequate coverage."
    assert isinstance(result.avg_score(), float)


def test_nested_model_with_fence():
    """ScoringResult inside code fences."""
    import json
    inner = json.dumps(_valid_scoring_result_dict())
    raw = f"```json\n{inner}\n```"
    result = parse_llm_json(raw, ScoringResult)
    assert result.completeness.score == 7
