"""
infra/json_parser.py — LLM JSON output parser with 4-strategy fallback chain.

Always use parse_llm_json() to parse LLM responses. Never use raw json.loads()
on model outputs — they frequently contain markdown fences, preambles, trailing
commas, or other syntax issues that this module handles gracefully.

See context/implementation-patterns.md for the reference implementation and
test cases to cover in tests/test_json_parser.py.
"""

import json
import re
from typing import Type, TypeVar

import json_repair
from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


def parse_llm_json(raw: str, model_class: Type[T]) -> T:
    """Parse LLM output into a validated Pydantic model using a 4-strategy fallback chain.

    Strategies (tried in order):
      1. Direct json.loads() — succeeds when the model returns clean JSON.
      2. Extract from ```json ... ``` code fences, then json.loads().
      3. json_repair.loads() — fixes trailing commas, single quotes, Python booleans.
      4. Bracket-match to find outermost {...}, then json_repair.loads().

    Args:
        raw:         Raw string output from the LLM.
        model_class: Pydantic model class to validate and deserialize into.

    Returns:
        A validated instance of model_class.

    Raises:
        ValueError: If all four strategies fail or the parsed dict fails Pydantic validation.
    """
    strategies = [
        # Strategy 1: Direct parse — succeeds if model returned clean JSON
        lambda s: json.loads(s),
        # Strategy 2: Extract from ```json ... ``` code fences
        lambda s: json.loads(
            re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", s, re.DOTALL).group(1).strip()
        ),
        # Strategy 3: json_repair — fixes trailing commas, single quotes, Python booleans
        lambda s: json_repair.loads(s),
        # Strategy 4: Bracket-match to outermost {...}, then repair
        lambda s: json_repair.loads(s[s.index("{") : s.rindex("}") + 1]),
    ]

    for strategy in strategies:
        try:
            parsed = strategy(raw)
            if isinstance(parsed, dict):
                return model_class(**parsed)
        except (json.JSONDecodeError, ValidationError, ValueError, AttributeError):
            continue

    raise ValueError(
        f"All JSON extraction strategies failed for {model_class.__name__}. "
        f"Raw output (first 500 chars): {raw[:500]}"
    )
