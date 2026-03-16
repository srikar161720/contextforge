"""
tests/test_quality_scorer.py — Unit and integration tests for core/quality_scorer.py.

Unit tests mock BedrockClient.invoke() so no Bedrock calls are made.
Integration tests are marked @pytest.mark.integration and skipped without creds.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from core.models import CriterionScore, ScoringResult
from core.quality_scorer import (
    SCORING_SYSTEM_PROMPT,
    SCORING_USER_TEMPLATE,
    score_response,
    _build_scoring_prompt,
)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_scoring_result(score: int = 7) -> ScoringResult:
    criterion = CriterionScore(score=score, justification="Test justification.")
    return ScoringResult(
        relevance=criterion,
        accuracy=criterion,
        completeness=criterion,
        groundedness=criterion,
    )


def _make_scoring_json(score: int = 7) -> str:
    """Return a valid JSON string matching ScoringResult schema."""
    return json.dumps(
        {
            "relevance":    {"score": score, "justification": "The response addresses the query."},
            "accuracy":     {"score": score, "justification": "Facts are correct."},
            "completeness": {"score": score, "justification": "All aspects covered."},
            "groundedness": {"score": score, "justification": "Grounded in the context."},
        }
    )


def _mock_client(response_text: str) -> MagicMock:
    """Create a mock BedrockClient whose invoke() returns response_text."""
    client = MagicMock()
    client.invoke.return_value = (
        response_text,   # text
        False,           # reasoning_active
        {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
    )
    return client


# ── Unit tests ────────────────────────────────────────────────────────────────


def test_score_response_returns_scoring_result():
    """score_response should return a valid ScoringResult for clean JSON output."""
    client = _mock_client(_make_scoring_json(8))
    result, usage = score_response(client, "What is the refund policy?", "You can get a refund in 30 days.")

    assert isinstance(result, ScoringResult)
    assert result.relevance.score == 8
    assert result.accuracy.score == 8
    assert result.completeness.score == 8
    assert result.groundedness.score == 8


def test_score_response_avg_score():
    """avg_score() should return the mean of all criterion scores."""
    client = _mock_client(_make_scoring_json(6))
    result, _ = score_response(client, "Query?", "Response.")
    assert result.avg_score() == pytest.approx(6.0)


def test_score_response_returns_usage():
    """score_response should return the usage dict from BedrockClient."""
    client = _mock_client(_make_scoring_json(7))
    _, usage = score_response(client, "Query?", "Response.")

    assert "input_tokens" in usage
    assert "output_tokens" in usage
    assert "total_tokens" in usage
    assert usage["total_tokens"] == 150


def test_score_response_calls_medium_reasoning():
    """score_response must always call invoke() with reasoning_tier='medium'."""
    client = _mock_client(_make_scoring_json(7))
    score_response(client, "Query?", "Response.")

    call_kwargs = client.invoke.call_args
    assert call_kwargs.kwargs["reasoning_tier"] == "medium"


def test_score_response_calls_correct_params():
    """score_response must call invoke() with temperature=0, max_tokens=16000."""
    client = _mock_client(_make_scoring_json(7))
    score_response(client, "Query?", "Response.")

    call_kwargs = client.invoke.call_args
    assert call_kwargs.kwargs["temperature"] == 0
    assert call_kwargs.kwargs["max_tokens"] == 16000


def test_score_response_uses_system_prompt():
    """invoke() should receive the scoring system prompt."""
    client = _mock_client(_make_scoring_json(7))
    score_response(client, "Query?", "Response.")

    call_kwargs = client.invoke.call_args
    assert call_kwargs.kwargs["system"] == SCORING_SYSTEM_PROMPT


def test_score_response_with_reference_answer():
    """Reference answer should appear in the user prompt sent to invoke()."""
    client = _mock_client(_make_scoring_json(9))
    score_response(
        client,
        "What is the refund window?",
        "You have 30 days to request a refund.",
        reference_answer="Refunds are available within 30 days of purchase.",
    )

    call_kwargs = client.invoke.call_args
    messages = call_kwargs.kwargs["messages"]
    user_text = messages[0]["content"][0]["text"]
    assert "REFERENCE ANSWER" in user_text
    assert "Refunds are available within 30 days" in user_text


def test_score_response_without_reference_answer():
    """Without a reference answer, the REFERENCE ANSWER block should be absent."""
    client = _mock_client(_make_scoring_json(7))
    score_response(client, "Query?", "Response.", reference_answer=None)

    call_kwargs = client.invoke.call_args
    messages = call_kwargs.kwargs["messages"]
    user_text = messages[0]["content"][0]["text"]
    assert "REFERENCE ANSWER" not in user_text


def test_score_response_invalid_json_raises():
    """score_response must raise ValueError if both medium and disabled retries fail."""
    # Both calls return unparseable text — medium attempt + disabled fallback
    client = MagicMock()
    bad_return = (
        "This is not JSON at all.",
        False,
        {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
    )
    client.invoke.return_value = bad_return
    with pytest.raises(ValueError):
        score_response(client, "Query?", "Response.")


def test_score_response_extra_criteria_in_prompt():
    """Extra criteria beyond the default 4 should appear in the scoring prompt."""
    client = _mock_client(_make_scoring_json(7))
    score_response(
        client,
        "Query?",
        "Response.",
        criteria=["relevance", "accuracy", "completeness", "groundedness", "tone"],
    )

    call_kwargs = client.invoke.call_args
    messages = call_kwargs.kwargs["messages"]
    user_text = messages[0]["content"][0]["text"]
    assert "tone" in user_text


def test_score_response_retries_with_disabled_on_parse_failure():
    """If medium reasoning produces unparseable output, retry with disabled tier."""
    valid_json = _make_scoring_json(7)
    usage_dict = {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150}

    client = MagicMock()
    # First call (medium) returns empty text; second call (disabled) returns valid JSON
    client.invoke.side_effect = [
        ("", False, usage_dict),          # medium — empty output, triggers fallback
        (valid_json, False, usage_dict),  # disabled — valid JSON
    ]

    result, usage = score_response(client, "Query?", "Response.")

    assert isinstance(result, ScoringResult)
    assert result.avg_score() == pytest.approx(7.0)

    # Verify two calls were made: first with medium, then with disabled
    assert client.invoke.call_count == 2
    first_call = client.invoke.call_args_list[0]
    second_call = client.invoke.call_args_list[1]
    assert first_call.kwargs["reasoning_tier"] == "medium"
    assert first_call.kwargs["max_tokens"] == 16000
    assert second_call.kwargs["reasoning_tier"] == "disabled"
    assert second_call.kwargs["max_tokens"] == 4000


def test_score_response_no_retry_on_success():
    """If medium reasoning produces valid output, no retry should occur."""
    client = _mock_client(_make_scoring_json(8))
    result, _ = score_response(client, "Query?", "Response.")

    assert isinstance(result, ScoringResult)
    assert client.invoke.call_count == 1


# ── _build_scoring_prompt helpers ─────────────────────────────────────────────


def test_build_scoring_prompt_contains_query():
    prompt = _build_scoring_prompt("Test query?", "Test response.", None, None)
    assert "Test query?" in prompt


def test_build_scoring_prompt_contains_response():
    prompt = _build_scoring_prompt("Query?", "The model said XYZ.", None, None)
    assert "The model said XYZ." in prompt


def test_build_scoring_prompt_schema_present():
    prompt = _build_scoring_prompt("Q?", "R.", None, None)
    assert "relevance" in prompt
    assert "groundedness" in prompt


# ── Integration tests ─────────────────────────────────────────────────────────


@pytest.mark.integration
def test_score_response_live(bedrock_client):
    """Call Nova and verify we get back a valid ScoringResult with scores in 1-10."""
    result, usage = score_response(
        bedrock_client,
        query="What services does Acme Cloud Platform offer?",
        response_text=(
            "Acme Cloud Platform offers cloud storage, compute instances, "
            "database services, and monitoring tools for enterprise customers."
        ),
        reference_answer="Acme offers storage, compute, databases, and monitoring.",
    )

    assert isinstance(result, ScoringResult)
    for field_name in ("relevance", "accuracy", "completeness", "groundedness"):
        criterion = getattr(result, field_name)
        assert 1 <= criterion.score <= 10, (
            f"{field_name}.score={criterion.score} out of valid 1-10 range"
        )
        assert isinstance(criterion.justification, str)
        assert len(criterion.justification) > 0

    assert usage["input_tokens"] > 0
    assert usage["output_tokens"] > 0
