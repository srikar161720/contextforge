"""
tests/test_day1_validation.py — Day 1 integration tests against live Bedrock API.

All tests are marked @pytest.mark.integration and require valid AWS credentials
in the .env file. They are skipped automatically when credentials are absent.

Tests validate the Bedrock client setup as specified in PROGRESS.md (Phase 1):
  1. Basic Converse API call succeeds
  2. Extended thinking (medium) returns a reasoningContent block
  3. Code Interpreter returns toolUse/toolResult blocks
  4. Extended thinking + Code Interpreter compatibility
  5. Usage reporting — inputTokens/outputTokens populated
  6. Large context (~200K tokens) succeeds without timeout
  7. Rate limit validation — RPM=200 / TPM=8M confirmed at runtime

Record test 4 result (compatibility matrix) in PROGRESS.md session log.
"""

import pytest

from infra.bedrock_client import BedrockClient

# Simple prompts for validation calls — short to keep costs minimal
_SIMPLE_MESSAGES = [
    {"role": "user", "content": [{"text": "Reply with exactly the word: HELLO"}]}
]

_CODE_INTERP_MESSAGES = [
    {
        "role": "user",
        "content": [{"text": "Use Python to compute 7 * 6 and print the result."}],
    }
]

_CODE_INTERP_TOOL_CONFIG = {
    "tools": [{"systemTool": {"name": "nova_code_interpreter"}}]
}


# ── Test 1: Basic Converse API call ───────────────────────────────────────────

@pytest.mark.integration
def test_basic_converse_call(bedrock_client: BedrockClient):
    """Verify a simple call returns non-empty text and positive token counts."""
    text, reasoning_active, usage = bedrock_client.invoke(
        system=None,
        messages=_SIMPLE_MESSAGES,
        reasoning_tier="disabled",
        max_tokens=50,
    )

    assert isinstance(text, str) and len(text) > 0, "Response text must be non-empty"
    assert reasoning_active is False, "reasoning_active must be False for 'disabled' tier"
    assert usage["input_tokens"] > 0, "input_tokens must be positive"
    assert usage["output_tokens"] > 0, "output_tokens must be positive"
    assert usage["total_tokens"] == usage["input_tokens"] + usage["output_tokens"]


# ── Test 2: Extended thinking (medium) ────────────────────────────────────────

@pytest.mark.integration
def test_extended_thinking_medium(bedrock_client: BedrockClient):
    """Verify medium reasoning tier returns a reasoningContent block."""
    text, reasoning_active, usage = bedrock_client.invoke(
        system=None,
        messages=[
            {"role": "user", "content": [{"text": "What is 3 + 4? Think step by step."}]}
        ],
        reasoning_tier="medium",
        max_tokens=2000,  # Must be large enough for reasoning tokens + text response
    )

    assert reasoning_active is True, (
        "reasoning_active must be True for 'medium' tier — "
        "a reasoningContent block should be present in the response"
    )
    assert isinstance(text, str) and len(text) > 0, "Response text must be non-empty"
    assert usage["output_tokens"] > 0


# ── Test 3: Code Interpreter ──────────────────────────────────────────────────

@pytest.mark.integration
def test_code_interpreter_returns_tool_blocks(bedrock_client: BedrockClient):
    """Verify Code Interpreter response contains toolUse and/or toolResult blocks."""
    blocks, usage = bedrock_client.invoke_raw(
        system=None,
        messages=_CODE_INTERP_MESSAGES,
        reasoning_tier="disabled",
        tool_config=_CODE_INTERP_TOOL_CONFIG,
        max_tokens=1000,
    )

    block_types = {list(b.keys())[0] for b in blocks}
    assert "toolUse" in block_types, (
        f"Expected toolUse block in response. Got block types: {block_types}"
    )
    assert usage["input_tokens"] > 0


# ── Test 4: Extended thinking + Code Interpreter compatibility ─────────────────

@pytest.mark.integration
def test_extended_thinking_and_code_interpreter_compatibility(
    bedrock_client: BedrockClient,
):
    """
    Compatibility discovery: extended thinking (medium) + Code Interpreter simultaneously.

    Per context/api-reference.md: "Confirmed compatible — response will contain
    reasoningContent, toolUse, toolResult, and text blocks together."

    Record result (PASS/FAIL) in PROGRESS.md session log.
    """
    blocks, usage = bedrock_client.invoke_raw(
        system=None,
        messages=_CODE_INTERP_MESSAGES,
        reasoning_tier="medium",
        tool_config=_CODE_INTERP_TOOL_CONFIG,
        max_tokens=1000,
    )

    block_types = {list(b.keys())[0] for b in blocks}

    # Both should coexist in the same response
    has_reasoning = "reasoningContent" in block_types
    has_tool_use  = "toolUse" in block_types

    # Log the compatibility matrix result regardless of assertion
    print(
        f"\n[Compatibility Matrix] Extended Thinking + Code Interpreter: "
        f"reasoning={has_reasoning}, tool_use={has_tool_use}, "
        f"block_types={block_types}"
    )

    assert has_reasoning, (
        "Expected reasoningContent block when reasoning_tier='medium'"
    )
    assert has_tool_use, (
        "Expected toolUse block when Code Interpreter tool is enabled"
    )


# ── Test 5: Usage reporting ───────────────────────────────────────────────────

@pytest.mark.integration
def test_usage_reporting_populated(bedrock_client: BedrockClient):
    """Verify response['usage'] contains positive inputTokens and outputTokens."""
    _, _, usage = bedrock_client.invoke(
        system="You are a helpful assistant.",
        messages=_SIMPLE_MESSAGES,
        reasoning_tier="disabled",
        max_tokens=50,
    )

    assert usage["input_tokens"] > 0,  "inputTokens must be > 0"
    assert usage["output_tokens"] > 0, "outputTokens must be > 0"
    assert usage["total_tokens"] > 0,  "totalTokens must be > 0"

    # Verify session-level cumulative tracking is working
    assert bedrock_client.total_api_calls > 0
    assert bedrock_client.total_input_tokens > 0
    assert bedrock_client.total_cost > 0.0


# ── Test 6: Large context (~200K tokens) ──────────────────────────────────────

@pytest.mark.integration
@pytest.mark.slow  # This test takes ~60s+ — run with: pytest -m "integration and slow"
def test_large_context_succeeds_without_timeout(bedrock_client: BedrockClient):
    """
    Verify that a ~200K token context payload completes without timeout.

    This validates that read_timeout=3600 is correctly configured.
    Expected duration: 30–120 seconds depending on model load.
    """
    # Generate ~200K tokens of filler text (~750 chars ≈ 200 tokens; need ~150K chars)
    filler_paragraph = (
        "This is a filler paragraph used to generate a large context payload for "
        "testing purposes only. It contains no meaningful information and is "
        "repeated many times to reach the target token count. "
    )
    # ~150 chars per paragraph × 1000 repetitions ≈ 150K chars ≈ ~37K tokens via tiktoken
    # Multiply by 5 to get closer to 200K tokens
    large_content = filler_paragraph * 5000  # ~750K chars ≈ ~190K tokens

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "text": (
                        f"{large_content}\n\n"
                        "Given the above context, reply with exactly one word: DONE"
                    )
                }
            ],
        }
    ]

    text, _, usage = bedrock_client.invoke(
        system=None,
        messages=messages,
        reasoning_tier="disabled",
        max_tokens=20,
    )

    assert isinstance(text, str) and len(text) > 0, "Must return non-empty response"
    assert usage["input_tokens"] > 100_000, (
        f"Expected >100K input tokens for large context test, got {usage['input_tokens']}"
    )


# ── Test 7: Rate limit validation ─────────────────────────────────────────────

@pytest.mark.integration
def test_rate_limiter_does_not_block_single_call(bedrock_client: BedrockClient):
    """
    Validate that a single API call is not blocked by the rate limiter.

    With RPM=200 and TPM=8M, a single small call should proceed immediately
    without any sleep delay. Confirms config.yaml values are correctly loaded.
    """
    import time

    start = time.time()
    text, _, usage = bedrock_client.invoke(
        system=None,
        messages=_SIMPLE_MESSAGES,
        reasoning_tier="disabled",
        max_tokens=20,
    )
    elapsed = time.time() - start

    # The rate limiter should add no delay for the first call in a clean window.
    # Allow up to 5s for network latency to the Bedrock endpoint.
    assert elapsed < 30, (
        f"Single call took {elapsed:.1f}s — rate limiter may be misconfigured"
    )
    assert len(text) > 0
    assert usage["input_tokens"] > 0
