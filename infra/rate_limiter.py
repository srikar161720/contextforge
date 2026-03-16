"""
infra/rate_limiter.py — Adaptive RPM + TPM rate limiter.

Uses sliding 60-second windows to enforce requests-per-minute and
tokens-per-minute limits. Instantiated once by BedrockClient and shared
across all API calls.
"""

import time
from collections import deque


class RateLimiter:
    def __init__(self, rpm: int = 200, tpm: int = 8_000_000) -> None:
        """
        Args:
            rpm: Max requests per minute (default: confirmed AWS quota).
            tpm: Max tokens per minute (default: confirmed AWS quota).
        """
        self.rpm = rpm
        self.tpm = tpm
        self.request_times: deque = deque()          # Timestamps of recent requests
        self.token_log: deque = deque()              # (timestamp, token_count) pairs

    def wait_if_needed(self, estimated_tokens: int = 50_000) -> None:
        """Block until the next request can proceed within rate limits.

        Args:
            estimated_tokens: Pre-flight token estimate (tiktoken); used for TPM check.
        """
        now = time.time()

        # Evict entries older than 60 seconds from both windows
        while self.request_times and now - self.request_times[0] > 60:
            self.request_times.popleft()
        while self.token_log and now - self.token_log[0][0] > 60:
            self.token_log.popleft()

        # Enforce RPM — sleep until the oldest request falls out of the window
        if len(self.request_times) >= self.rpm:
            sleep_time = 60 - (now - self.request_times[0]) + 0.1
            time.sleep(max(sleep_time, 0))
            now = time.time()

        # Enforce TPM — sleep until enough tokens roll out of the window
        current_tpm = sum(t for _, t in self.token_log)
        if current_tpm + estimated_tokens > self.tpm:
            sleep_time = 60 - (now - self.token_log[0][0]) + 0.1
            time.sleep(max(sleep_time, 0))

        self.request_times.append(time.time())

    def log_usage(self, actual_tokens: int) -> None:
        """Record actual token usage after an API call.

        Must be called after every successful API call with the API-reported
        total token count (inputTokens + outputTokens).

        Args:
            actual_tokens: API-reported total tokens for the completed call.
        """
        self.token_log.append((time.time(), actual_tokens))
