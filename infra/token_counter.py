"""
infra/token_counter.py — Token counting utilities.

Two distinct token sources are used throughout ContextForge:
  - tiktoken (cl100k_base): Pre-flight estimates for rate limiting and UI display.
  - API-reported usage:     Authoritative counts used in all reports and cost calculations.

Never use tiktoken counts in cost reports. See context/data-models.md for the
token counting convention.
"""

import tiktoken


# Load encoding once at module level to avoid repeated initialization cost
_enc = tiktoken.get_encoding("cl100k_base")


def estimate_tokens(text: str) -> int:
    """Approximate token count for pre-flight planning and UI display.

    Uses tiktoken cl100k_base encoding as a reasonable approximation for Nova.
    NOT used in reports or cost calculations — API-reported counts are authoritative.

    Args:
        text: The text to estimate token count for.

    Returns:
        Estimated token count.
    """
    return len(_enc.encode(text))


def extract_usage(response: dict) -> dict:
    """Extract API-reported token usage from a Bedrock Converse response.

    Use this for all reports and cost calculations. outputTokens includes
    reasoning tokens — they are billed at the same output rate.

    Args:
        response: Raw response dict from bedrock.converse().

    Returns:
        Dict with keys: input_tokens, output_tokens, total_tokens.
    """
    usage = response.get("usage", {})
    return {
        "input_tokens":  usage.get("inputTokens", 0),
        "output_tokens": usage.get("outputTokens", 0),  # Includes reasoning tokens
        "total_tokens":  usage.get("totalTokens", 0),
    }


def compute_cost(input_tokens: int, output_tokens: int) -> float:
    """Compute USD cost using US CRIS pricing for us.amazon.nova-2-lite-v1:0.

    Pricing: $0.30 / 1M input tokens, $2.50 / 1M output tokens.
    Output tokens include reasoning tokens — no separate reasoning token price.

    Args:
        input_tokens:  API-reported input token count.
        output_tokens: API-reported output token count (includes reasoning).

    Returns:
        Estimated cost in USD.
    """
    return (input_tokens * 0.30 + output_tokens * 2.50) / 1_000_000
