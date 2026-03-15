"""
infra/bedrock_client.py — Single gateway for all AWS Bedrock API calls.

ALL business logic must call BedrockClient.invoke() or BedrockClient.invoke_raw().
Never call bedrock.converse() directly from outside this module.

Critical rules enforced here:
  - Only bedrock.converse() is used — never invoke_model() (except embeddings stretch goal)
  - HIGH reasoning tier: inferenceConfig is omitted entirely (no temperature/topP/maxTokens)
  - Reasoning content is [REDACTED]; only its presence is checked, never parsed
  - Every call goes through rate limiting and retry logic

See context/api-reference.md and context/implementation-patterns.md for the
reference implementation and API call shape.
"""

import logging
import time
from pathlib import Path

import boto3
import yaml
from botocore.config import Config
from botocore.exceptions import ClientError

from infra.rate_limiter import RateLimiter
from infra import token_counter

logger = logging.getLogger(__name__)

# ── Error classification ───────────────────────────────────────────────────────

RETRYABLE_ERRORS = {
    "ThrottlingException",
    "ServiceUnavailableException",
    "ModelTimeoutException",
    "RequestTimeoutException",
    "ModelNotReadyException",
}

FATAL_ERRORS = {
    "AccessDeniedException",
    "ResourceNotFoundException",
}

# ── Reasoning tier configurations ──────────────────────────────────────────────

REASONING_TIERS = {
    "disabled": None,  # Omit additionalModelRequestFields entirely
    "low":      {"type": "enabled", "maxReasoningEffort": "low"},
    "medium":   {"type": "enabled", "maxReasoningEffort": "medium"},
    "high":     {"type": "enabled", "maxReasoningEffort": "high"},
}


def _load_config(config_path: Path) -> dict:
    """Load and return config.yaml as a dict."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _build_inference_params(
    tier_name: str,
    max_tokens: int | None = None,
    temperature: float | None = None,
) -> dict:
    """Build additionalModelRequestFields and inferenceConfig for a reasoning tier.

    HIGH tier constraint: inferenceConfig is omitted entirely. Passing temperature,
    topP, topK, or maxTokens with HIGH tier causes a ValidationException.

    Args:
        tier_name:   Key from REASONING_TIERS ("disabled", "low", "medium", "high").
        max_tokens:  Ignored for "high" tier.
        temperature: Ignored for "high" tier.

    Returns:
        Dict ready to unpack with ** into bedrock.converse().
    """
    params: dict = {}
    tier_config = REASONING_TIERS[tier_name]

    if tier_config is not None:
        params["additionalModelRequestFields"] = {"reasoningConfig": tier_config}

    if tier_name != "high" and (max_tokens is not None or temperature is not None):
        params["inferenceConfig"] = {}
        if max_tokens is not None:
            params["inferenceConfig"]["maxTokens"] = max_tokens
        if temperature is not None:
            params["inferenceConfig"]["temperature"] = temperature

    return params


class BedrockClient:
    """Converse API wrapper with rate limiting, retry logic, and usage tracking."""

    def __init__(self, config: dict | None = None) -> None:
        """Initialise the Bedrock client.

        Args:
            config: Optional config dict (for testing). If None, loads config.yaml
                    from the project root relative to this file's location.
        """
        if config is None:
            config_path = Path(__file__).parent.parent / "config.yaml"
            config = _load_config(config_path)

        self._config = config
        self._model_id: str = config["model"]["id"]
        self._region: str = config["model"]["region"]

        # boto3 client — read_timeout=3600 required for extended thinking on large contexts
        self._bedrock = boto3.client(
            "bedrock-runtime",
            region_name=self._region,
            config=Config(
                read_timeout=3600,
                connect_timeout=60,
                retries={"max_attempts": 3, "mode": "adaptive"},
            ),
        )

        # Rate limiter — uses confirmed AWS quota values from config.yaml
        rpm: int = config["rate_limits"]["rpm"]
        tpm: int = config["rate_limits"]["tpm"]
        self._rate_limiter = RateLimiter(rpm=rpm, tpm=tpm)

        # Cumulative usage tracking across the session
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0
        self.total_api_calls: int = 0
        self.total_cost: float = 0.0

    # ── Public interface ───────────────────────────────────────────────────────

    def invoke(
        self,
        system: str | None,
        messages: list[dict],
        reasoning_tier: str = "disabled",
        tool_config: dict | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> tuple[str, bool, dict]:
        """Call the Converse API and return concatenated text output.

        Args:
            system:         System prompt text, or None to omit.
            messages:       Converse API messages list (role/content dicts).
            reasoning_tier: One of "disabled", "low", "medium", "high".
            tool_config:    System tool config dict, e.g. {"tools": [...]}, or None.
            max_tokens:     Max output tokens. Ignored for "high" tier.
            temperature:    Sampling temperature. Ignored for "high" tier.

        Returns:
            text (str):             Concatenated text blocks from the response.
            reasoning_active (bool): True if a reasoningContent block was present.
            usage (dict):           {"input_tokens": int, "output_tokens": int, "total_tokens": int}

        Raises:
            ClientError: AccessDeniedException or ResourceNotFoundException (fatal, not retried).
        """
        kwargs = self._build_converse_kwargs(
            system, messages, reasoning_tier, tool_config, max_tokens, temperature
        )
        estimated = self._estimate_request_tokens(system, messages)
        self._rate_limiter.wait_if_needed(estimated)

        response = self._invoke_with_retry(**kwargs)

        text, reasoning_active, _ = self._parse_response(response)
        usage = token_counter.extract_usage(response)

        self._rate_limiter.log_usage(usage["total_tokens"])
        self._update_cumulative_usage(usage)

        return text, reasoning_active, usage

    def invoke_raw(
        self,
        system: str | None,
        messages: list[dict],
        reasoning_tier: str = "disabled",
        tool_config: dict | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> tuple[list[dict], dict]:
        """Call the Converse API and return raw content blocks.

        Used for Code Interpreter responses where toolUse/toolResult blocks
        must be inspected individually.

        Args:
            system, messages, reasoning_tier, tool_config, max_tokens, temperature:
                Same as invoke().

        Returns:
            blocks (list[dict]): All content blocks from response["output"]["message"]["content"].
            usage (dict):        {"input_tokens": int, "output_tokens": int, "total_tokens": int}
        """
        kwargs = self._build_converse_kwargs(
            system, messages, reasoning_tier, tool_config, max_tokens, temperature
        )
        estimated = self._estimate_request_tokens(system, messages)
        self._rate_limiter.wait_if_needed(estimated)

        response = self._invoke_with_retry(**kwargs)

        _, _, blocks = self._parse_response(response)
        usage = token_counter.extract_usage(response)

        self._rate_limiter.log_usage(usage["total_tokens"])
        self._update_cumulative_usage(usage)

        return blocks, usage

    # ── Private helpers ────────────────────────────────────────────────────────

    def _build_converse_kwargs(
        self,
        system: str | None,
        messages: list[dict],
        reasoning_tier: str,
        tool_config: dict | None,
        max_tokens: int | None,
        temperature: float | None,
    ) -> dict:
        """Construct the kwargs dict for bedrock.converse()."""
        kwargs: dict = {"modelId": self._model_id, "messages": messages}

        if system:
            kwargs["system"] = [{"text": system}]

        kwargs.update(
            _build_inference_params(reasoning_tier, max_tokens, temperature)
        )

        if tool_config is not None:
            kwargs["toolConfig"] = tool_config

        return kwargs

    def _invoke_with_retry(self, **kwargs) -> dict:
        """Call bedrock.converse() with exponential backoff retry.

        Retry schedule: 5s → 10s → 20s (3 attempts total).
        ModelNotReadyException gets a flat 30s wait.
        Fatal errors (AccessDeniedException, ResourceNotFoundException) are
        re-raised immediately without retry.
        Non-retryable errors (ValidationException, etc.) are also re-raised immediately.

        Returns:
            Raw Bedrock Converse API response dict.

        Raises:
            ClientError: On fatal errors or after exhausting all retries.
        """
        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                return self._bedrock.converse(**kwargs)
            except ClientError as e:
                code = e.response["Error"]["Code"]
                if code in FATAL_ERRORS:
                    logger.error("Fatal Bedrock error (%s) — not retrying.", code)
                    raise
                if code not in RETRYABLE_ERRORS:
                    logger.warning("Non-retryable Bedrock error (%s) — skipping.", code)
                    raise
                last_exc = e
                wait = 30 if code == "ModelNotReadyException" else (2 ** attempt) * 5
                logger.warning(
                    "Retryable Bedrock error (%s) on attempt %d/%d — waiting %ds.",
                    code, attempt + 1, 3, wait,
                )
                time.sleep(wait)

        raise last_exc  # type: ignore[misc]

    def _parse_response(
        self, response: dict
    ) -> tuple[str, bool, list[dict]]:
        """Parse Converse API response content blocks.

        Iterates all blocks in response["output"]["message"]["content"]:
          - reasoningContent: sets reasoning_active=True; content is [REDACTED], never parsed
          - text:             appended to text_parts
          - toolUse:          included in blocks list
          - toolResult:       included in blocks list

        Returns:
            text (str):              Concatenated text from all text blocks.
            reasoning_active (bool): True if any reasoningContent block was present.
            blocks (list[dict]):     All raw content blocks (for invoke_raw callers).
        """
        content_blocks: list[dict] = response["output"]["message"]["content"]

        text_parts: list[str] = []
        reasoning_active: bool = False

        for block in content_blocks:
            if "reasoningContent" in block:
                reasoning_active = True  # Presence confirmed; content is always [REDACTED]
            elif "text" in block:
                text_parts.append(block["text"])

        return "\n".join(text_parts), reasoning_active, content_blocks

    def _estimate_request_tokens(
        self, system: str | None, messages: list[dict]
    ) -> int:
        """Pre-flight token estimate for rate limiter (tiktoken, not authoritative)."""
        parts: list[str] = []
        if system:
            parts.append(system)
        for msg in messages:
            for content_block in msg.get("content", []):
                if "text" in content_block:
                    parts.append(content_block["text"])
        return token_counter.estimate_tokens(" ".join(parts))

    def _update_cumulative_usage(self, usage: dict) -> None:
        """Update session-level usage counters after a successful API call."""
        self.total_input_tokens  += usage["input_tokens"]
        self.total_output_tokens += usage["output_tokens"]
        self.total_api_calls     += 1
        self.total_cost          += token_counter.compute_cost(
            usage["input_tokens"], usage["output_tokens"]
        )
