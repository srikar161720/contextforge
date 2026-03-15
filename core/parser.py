"""
core/parser.py — Context payload parser and validator.

Parses a JSON payload (file path or dict) into a validated ContextPayload.
Recomputes token counts for all sections via tiktoken (user-provided values
are always overwritten). Validates structural requirements before returning.

See context/data-models.md for the expected input JSON schema.
"""

from __future__ import annotations

import json
from pathlib import Path

from core.models import ContextPayload, ContextSection, EvalQuery, SectionType
from infra import token_counter


def parse_payload(source: str | Path | dict) -> ContextPayload:
    """Parse a JSON payload into a validated ContextPayload with token counts.

    Args:
        source: File path (str or Path) to a JSON file, or a pre-loaded dict.

    Returns:
        ContextPayload with token_count populated on every section and
        total_tokens computed as the sum of all section token counts.

    Raises:
        ValueError: On validation failures — duplicate IDs, empty sections,
                    fewer than 3 evaluation queries, or invalid section types.
        FileNotFoundError: If source is a path that does not exist.
        json.JSONDecodeError: If source is a path to an invalid JSON file.
    """
    raw = _load_raw(source)

    _validate_sections_list(raw)
    _validate_eval_queries(raw)
    _validate_unique_ids(raw)

    sections = _build_sections(raw["sections"])
    queries  = _build_queries(raw["evaluation_queries"])

    criteria: list[str] = raw.get(
        "quality_criteria",
        ["relevance", "accuracy", "completeness", "groundedness"],
    )

    total_tokens = sum(s.token_count for s in sections)

    return ContextPayload(
        sections=sections,
        evaluation_queries=queries,
        quality_criteria=criteria,
        total_tokens=total_tokens,
    )


# ── Private helpers ──────────────────────────────────────────────────────────


def _load_raw(source: str | Path | dict) -> dict:
    """Load raw payload from a file path or return the dict directly."""
    if isinstance(source, dict):
        return source
    path = Path(source)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _validate_sections_list(raw: dict) -> None:
    """Raise ValueError if sections list is missing or empty."""
    sections = raw.get("sections")
    if not sections:
        raise ValueError(
            "Payload must contain a non-empty 'sections' list."
        )


def _validate_eval_queries(raw: dict) -> None:
    """Raise ValueError if fewer than 3 evaluation queries are provided."""
    queries = raw.get("evaluation_queries", [])
    if len(queries) < 3:
        raise ValueError(
            f"Payload must contain at least 3 evaluation queries; "
            f"found {len(queries)}."
        )


def _validate_unique_ids(raw: dict) -> None:
    """Raise ValueError if any section IDs are duplicated."""
    ids: list[str] = [s.get("id", "") for s in raw.get("sections", [])]
    seen: set[str] = set()
    duplicates: list[str] = []
    for sid in ids:
        if sid in seen:
            duplicates.append(sid)
        seen.add(sid)
    if duplicates:
        raise ValueError(
            f"Duplicate section IDs found: {duplicates}. "
            "All section IDs must be unique."
        )


def _build_sections(raw_sections: list[dict]) -> list[ContextSection]:
    """Construct ContextSection objects with recomputed token counts.

    Token counts are always recomputed via tiktoken — user-provided values
    are ignored. This ensures consistency with how tokens are tracked
    throughout the pipeline.
    """
    sections: list[ContextSection] = []
    for raw_s in raw_sections:
        # Validate section_type before constructing the model
        section_type_str = raw_s.get("section_type", "")
        try:
            SectionType(section_type_str)
        except ValueError:
            valid = [t.value for t in SectionType]
            raise ValueError(
                f"Invalid section_type '{section_type_str}' for section "
                f"'{raw_s.get('id', '?')}'. Valid values: {valid}"
            )

        content = raw_s.get("content", "")
        computed_tokens = token_counter.estimate_tokens(content)

        section = ContextSection(
            id=raw_s["id"],
            label=raw_s.get("label", raw_s["id"]),
            section_type=SectionType(section_type_str),
            content=content,
            token_count=computed_tokens,
            metadata=raw_s.get("metadata", {}),
        )
        sections.append(section)

    return sections


def _build_queries(raw_queries: list[dict]) -> list[EvalQuery]:
    """Construct EvalQuery objects from raw dicts."""
    return [
        EvalQuery(
            query=q["query"],
            reference_answer=q.get("reference_answer"),
        )
        for q in raw_queries
    ]
