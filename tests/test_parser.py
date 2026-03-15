"""
tests/test_parser.py — Unit tests for core/parser.py.

All tests are purely local — no Bedrock calls. Token counts use tiktoken
estimates (same as the parser itself).
"""

from __future__ import annotations

import json
import pathlib

import pytest

from core.models import SectionType
from core.parser import parse_payload


# ── Fixtures ─────────────────────────────────────────────────────────────────


def _make_section(
    id: str = "sec_001",
    label: str = "Test Section",
    section_type: str = "rag_document",
    content: str = "This is some test content for the section.",
    token_count: int = 0,
) -> dict:
    """Helper to build a minimal section dict."""
    return {
        "id": id,
        "label": label,
        "section_type": section_type,
        "content": content,
        "token_count": token_count,
    }


def _make_query(query: str = "What is the refund policy?") -> dict:
    return {"query": query, "reference_answer": "You can request a refund within 30 days."}


def _minimal_payload(
    sections: list[dict] | None = None,
    queries: list[dict] | None = None,
) -> dict:
    """Build a minimal valid payload dict."""
    if sections is None:
        sections = [
            _make_section("sys_001", "System Prompt", "system_prompt", "You are a helpful assistant."),
            _make_section("rag_001", "Company FAQ", "rag_document", "Q: What is this? A: A test service."),
        ]
    if queries is None:
        queries = [_make_query(f"Query {i}") for i in range(3)]
    return {"sections": sections, "evaluation_queries": queries}


# ── Test 1: Valid minimal payload ─────────────────────────────────────────────


def test_valid_minimal_payload():
    payload = parse_payload(_minimal_payload())
    assert len(payload.sections) == 2
    assert payload.total_tokens > 0
    for section in payload.sections:
        assert section.token_count > 0


# ── Test 2: Token count overwrite ─────────────────────────────────────────────


def test_token_count_overwrite():
    """Parser must overwrite user-provided token_count with tiktoken estimate."""
    sections = [
        _make_section("sys_001", "Sys", "system_prompt", "You are a helpful assistant.", token_count=999),
        _make_section("rag_001", "FAQ", "rag_document", "Q: What is this? A: A service.", token_count=999),
    ]
    payload = parse_payload(_minimal_payload(sections=sections))
    for section in payload.sections:
        assert section.token_count != 999, (
            f"Section {section.id} still has user-provided token_count=999"
        )
        assert section.token_count > 0


# ── Test 3: Duplicate IDs ─────────────────────────────────────────────────────


def test_duplicate_ids_raises():
    sections = [
        _make_section("dup_001", "Sec A", "rag_document", "Content A"),
        _make_section("dup_001", "Sec B", "rag_document", "Content B"),  # duplicate
    ]
    with pytest.raises(ValueError, match="Duplicate section IDs"):
        parse_payload(_minimal_payload(sections=sections))


# ── Test 4: Empty sections list ───────────────────────────────────────────────


def test_empty_sections_raises():
    with pytest.raises(ValueError, match="non-empty 'sections'"):
        parse_payload({"sections": [], "evaluation_queries": [_make_query() for _ in range(3)]})


# ── Test 5: Too few queries ───────────────────────────────────────────────────


def test_too_few_queries_raises():
    with pytest.raises(ValueError, match="at least 3 evaluation queries"):
        parse_payload({
            "sections": [_make_section()],
            "evaluation_queries": [_make_query(), _make_query()],  # only 2
        })


# ── Test 6: All section types ─────────────────────────────────────────────────


def test_all_section_types():
    sections = [
        _make_section(f"sec_{i:03d}", f"Section {t.value}", t.value, f"Content for {t.value}.")
        for i, t in enumerate(SectionType)
    ]
    payload = parse_payload(_minimal_payload(sections=sections))
    returned_types = {s.section_type for s in payload.sections}
    assert returned_types == set(SectionType)


# ── Test 7: Invalid section type ──────────────────────────────────────────────


def test_invalid_section_type_raises():
    sections = [_make_section("bad_001", "Bad", "nonexistent_type", "Content")]
    with pytest.raises(ValueError, match="Invalid section_type"):
        parse_payload(_minimal_payload(sections=sections))


# ── Test 8: File path input ───────────────────────────────────────────────────


def test_file_path_input(tmp_path: pathlib.Path):
    payload_dict = _minimal_payload()
    payload_file = tmp_path / "test_payload.json"
    payload_file.write_text(json.dumps(payload_dict), encoding="utf-8")

    # str path
    payload = parse_payload(str(payload_file))
    assert len(payload.sections) == 2

    # pathlib.Path
    payload = parse_payload(payload_file)
    assert len(payload.sections) == 2


# ── Test 9: Default quality_criteria ──────────────────────────────────────────


def test_default_quality_criteria():
    """quality_criteria should default to the 4 standard criteria when omitted."""
    payload = parse_payload(_minimal_payload())
    assert payload.quality_criteria == ["relevance", "accuracy", "completeness", "groundedness"]


def test_custom_quality_criteria():
    """User-supplied quality_criteria should be preserved."""
    raw = _minimal_payload()
    raw["quality_criteria"] = ["relevance", "tone"]
    payload = parse_payload(raw)
    assert payload.quality_criteria == ["relevance", "tone"]


# ── Test 10: Demo payload smoke test ─────────────────────────────────────────


def test_demo_payload_smoke():
    """Load the real demo payload and verify structural invariants."""
    demo_path = pathlib.Path(__file__).parent.parent / "data" / "demo_payloads" / "customer_support.json"
    if not demo_path.exists():
        pytest.skip("Demo payload not found — run scripts/generate_demo_payload.py first")

    payload = parse_payload(demo_path)

    assert len(payload.sections) == 79, f"Expected 79 sections, got {len(payload.sections)}"
    assert len(payload.evaluation_queries) == 10, (
        f"Expected 10 queries, got {len(payload.evaluation_queries)}"
    )
    assert 200_000 <= payload.total_tokens <= 220_000, (
        f"Expected total_tokens in [200K, 220K], got {payload.total_tokens}"
    )
    assert all(s.token_count > 0 for s in payload.sections), (
        "All sections should have non-zero token counts"
    )


# ── Test 11: total_tokens equals sum of section token_counts ──────────────────


def test_total_tokens_is_sum():
    payload = parse_payload(_minimal_payload())
    expected = sum(s.token_count for s in payload.sections)
    assert payload.total_tokens == expected


# ── Test 12: Missing sections key ─────────────────────────────────────────────


def test_missing_sections_key_raises():
    with pytest.raises((ValueError, KeyError)):
        parse_payload({"evaluation_queries": [_make_query() for _ in range(3)]})
