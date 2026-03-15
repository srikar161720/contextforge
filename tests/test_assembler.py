"""
tests/test_assembler.py — Unit tests for core/assembler.py.

All tests are purely local — no Bedrock calls.
"""

from __future__ import annotations

import pytest

from core.assembler import assemble_api_call
from core.models import ContextSection, SectionType


# ── Fixtures ─────────────────────────────────────────────────────────────────


def _make_section(
    id: str,
    label: str,
    section_type: SectionType,
    content: str,
) -> ContextSection:
    return ContextSection(
        id=id,
        label=label,
        section_type=section_type,
        content=content,
        token_count=10,
    )


@pytest.fixture()
def sections() -> list[ContextSection]:
    """A standard set of sections covering common types."""
    return [
        _make_section("sys_001", "System Prompt", SectionType.SYSTEM_PROMPT, "You are a helpful assistant."),
        _make_section("faq_001", "Company FAQ", SectionType.RAG_DOCUMENT, "Q: What is this?\nA: A test service."),
        _make_section("tool_001", "refund_processor", SectionType.TOOL_DEFINITION, '{"name": "refund_processor"}'),
        _make_section("conv_001", "Conversation Turn 1", SectionType.CONVERSATION_TURN, "Customer: Hello."),
    ]


QUERY = "How do I get a refund?"


# ── Test 1: Baseline assembly ─────────────────────────────────────────────────


def test_baseline_assembly(sections):
    result = assemble_api_call(sections, QUERY)

    # System field should contain the system prompt content
    assert result["system"] == "You are a helpful assistant."

    # Messages should be a single user turn
    assert len(result["messages"]) == 1
    assert result["messages"][0]["role"] == "user"

    user_text = result["messages"][0]["content"][0]["text"]

    # All non-system sections should appear
    assert "Company FAQ" in user_text
    assert "refund_processor" in user_text
    assert "Conversation Turn 1" in user_text

    # Query should appear at the end
    assert user_text.endswith(f"Query: {QUERY}")


# ── Test 2: Exclude one section ───────────────────────────────────────────────


def test_exclude_one_section(sections):
    result = assemble_api_call(sections, QUERY, exclude_ids={"faq_001"})
    user_text = result["messages"][0]["content"][0]["text"]

    assert "Company FAQ" not in user_text
    # Other sections still present
    assert "refund_processor" in user_text
    assert "Conversation Turn 1" in user_text


# ── Test 3: Exclude system prompt ─────────────────────────────────────────────


def test_exclude_system_prompt(sections):
    result = assemble_api_call(sections, QUERY, exclude_ids={"sys_001"})
    assert result["system"] is None
    # Non-system sections still present
    user_text = result["messages"][0]["content"][0]["text"]
    assert "Company FAQ" in user_text


# ── Test 4: Custom ordering ───────────────────────────────────────────────────


def test_custom_ordering(sections):
    # Reverse order: tool_001 first, then conv_001, then faq_001
    result = assemble_api_call(
        sections,
        QUERY,
        ordering=["tool_001", "conv_001", "faq_001"],
    )
    user_text = result["messages"][0]["content"][0]["text"]

    pos_tool = user_text.index("refund_processor")
    pos_conv = user_text.index("Conversation Turn 1")
    pos_faq  = user_text.index("Company FAQ")

    assert pos_tool < pos_conv < pos_faq, (
        "Sections should appear in the specified order: tool → conv → faq"
    )


def test_ordering_unlisted_sections_go_last():
    """Sections not in the ordering list should appear after listed ones."""
    secs = [
        _make_section("rag_a", "RAG A", SectionType.RAG_DOCUMENT, "Content A"),
        _make_section("rag_b", "RAG B", SectionType.RAG_DOCUMENT, "Content B"),
        _make_section("rag_c", "RAG C", SectionType.RAG_DOCUMENT, "Content C"),
    ]
    result = assemble_api_call(secs, QUERY, ordering=["rag_c"])
    user_text = result["messages"][0]["content"][0]["text"]

    pos_c = user_text.index("RAG C")
    pos_a = user_text.index("RAG A")
    pos_b = user_text.index("RAG B")

    # rag_c first (in ordering list), rag_a and rag_b after (not in list)
    assert pos_c < pos_a
    assert pos_c < pos_b


# ── Test 5: All sections excluded ─────────────────────────────────────────────


def test_all_sections_excluded(sections):
    all_ids = {s.id for s in sections}
    result = assemble_api_call(sections, QUERY, exclude_ids=all_ids)

    assert result["system"] is None

    user_text = result["messages"][0]["content"][0]["text"]
    # Only the query separator should remain
    assert user_text == f"---\n\nQuery: {QUERY}"


# ── Test 6: Section tag format ────────────────────────────────────────────────


def test_section_tag_format():
    """Each non-system section must be wrapped in [{type}: {label}]...[/{type}] tags."""
    secs = [
        _make_section("rag_001", "My FAQ", SectionType.RAG_DOCUMENT, "FAQ content here."),
    ]
    result = assemble_api_call(secs, QUERY)
    user_text = result["messages"][0]["content"][0]["text"]

    expected_open  = "[rag_document: My FAQ]"
    expected_close = "[/rag_document]"
    assert expected_open  in user_text, f"Expected opening tag '{expected_open}' in output"
    assert expected_close in user_text, f"Expected closing tag '{expected_close}' in output"
    # Content between tags
    assert "FAQ content here." in user_text


def test_section_tag_format_all_non_system_types():
    """All non-system section types should use their enum value in the tag."""
    non_system_types = [t for t in SectionType if t != SectionType.SYSTEM_PROMPT]
    for stype in non_system_types:
        secs = [_make_section("sec_001", "Label", stype, "Some content.")]
        result = assemble_api_call(secs, QUERY)
        user_text = result["messages"][0]["content"][0]["text"]
        assert f"[{stype.value}: Label]" in user_text
        assert f"[/{stype.value}]" in user_text


# ── Test 7: Multiple system prompts ──────────────────────────────────────────


def test_multiple_system_prompts():
    """Multiple system_prompt sections should be concatenated with \\n\\n."""
    secs = [
        _make_section("sys_001", "Sys A", SectionType.SYSTEM_PROMPT, "Part one of system prompt."),
        _make_section("sys_002", "Sys B", SectionType.SYSTEM_PROMPT, "Part two of system prompt."),
        _make_section("rag_001", "FAQ",   SectionType.RAG_DOCUMENT,   "Some FAQ content."),
    ]
    result = assemble_api_call(secs, QUERY)
    assert result["system"] == "Part one of system prompt.\n\nPart two of system prompt."


# ── Test 8: Query separator and position ─────────────────────────────────────


def test_query_at_end(sections):
    result = assemble_api_call(sections, QUERY)
    user_text = result["messages"][0]["content"][0]["text"]
    # The "---" separator followed by the query must appear at the end
    assert "---\n\nQuery: " + QUERY in user_text
    # Verify nothing follows the query
    assert user_text.endswith("Query: " + QUERY)


# ── Test 9: Return structure keys ─────────────────────────────────────────────


def test_return_structure(sections):
    result = assemble_api_call(sections, QUERY)
    assert "system" in result
    assert "messages" in result
    assert isinstance(result["messages"], list)
    msg = result["messages"][0]
    assert msg["role"] == "user"
    assert isinstance(msg["content"], list)
    assert "text" in msg["content"][0]


# ── Test 10: Empty exclude_ids (default) uses all sections ────────────────────


def test_empty_exclude_ids_default(sections):
    result_default = assemble_api_call(sections, QUERY)
    result_empty   = assemble_api_call(sections, QUERY, exclude_ids=set())
    assert result_default == result_empty
