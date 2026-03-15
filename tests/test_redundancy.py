"""
tests/test_redundancy.py — Unit tests for core/redundancy.py.

All tests are purely local — no API calls.
"""

from __future__ import annotations

import pytest

from core.models import ContextSection, SectionType
from core.redundancy import detect_redundancy


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_section(id: str, content: str) -> ContextSection:
    return ContextSection(
        id=id,
        label=id,
        section_type=SectionType.RAG_DOCUMENT,
        content=content,
        token_count=len(content.split()),
    )


# ── Tests ─────────────────────────────────────────────────────────────────────


def test_detect_redundancy_empty_list():
    """Empty section list returns no pairs."""
    assert detect_redundancy([]) == []


def test_detect_redundancy_single_section():
    """Single section returns no pairs (nothing to compare against)."""
    section = _make_section("s1", "Cloud storage for enterprise customers.")
    assert detect_redundancy([section]) == []


def test_detect_redundancy_identical_sections():
    """Two identical sections produce a pair with cosine similarity near 1.0."""
    text = "Acme Cloud Platform provides storage compute and database services."
    s1 = _make_section("s1", text)
    s2 = _make_section("s2", text)

    pairs = detect_redundancy([s1, s2], threshold=0.5)

    assert len(pairs) == 1
    section_ids = {pairs[0][0], pairs[0][1]}
    assert section_ids == {"s1", "s2"}
    assert pairs[0][2] == pytest.approx(1.0, abs=1e-6)


def test_detect_redundancy_different_sections_empty():
    """Very different sections produce no pairs above the default 0.7 threshold."""
    s1 = _make_section("s1", "Virtual machine compute instance cloud storage bucket.")
    s2 = _make_section("s2", "Refund policy thirty days money back guarantee billing invoice.")

    pairs = detect_redundancy([s1, s2], threshold=0.7)

    assert pairs == []


def test_detect_redundancy_threshold_respected():
    """Pairs below threshold are excluded; pairs at or above are included."""
    text_a = "The system processes refund requests within two business days."
    text_b = "Refund requests are processed within two business days by the system."
    s1 = _make_section("s1", text_a)
    s2 = _make_section("s2", text_b)

    # High threshold — no pairs expected
    assert detect_redundancy([s1, s2], threshold=0.99) == []

    # Low threshold — should detect at least one pair
    low_pairs = detect_redundancy([s1, s2], threshold=0.1)
    assert len(low_pairs) >= 1


def test_detect_redundancy_sorted_by_similarity_descending():
    """Output pairs are sorted by similarity descending (highest first)."""
    # s1/s2 are near-identical; s3/s4 are similar but less so
    s1 = _make_section("s1", "Cloud storage service for enterprise.")
    s2 = _make_section("s2", "Cloud storage service for enterprise customers.")
    s3 = _make_section("s3", "Monthly billing and invoicing cycle.")
    s4 = _make_section("s4", "Monthly invoices and billing.")

    pairs = detect_redundancy([s1, s2, s3, s4], threshold=0.1)

    if len(pairs) >= 2:
        sims = [p[2] for p in pairs]
        assert sims == sorted(sims, reverse=True)


def test_detect_redundancy_no_duplicate_pairs():
    """Each (i, j) pair appears exactly once — no (j, i) duplicate."""
    text = "Cloud storage compute database."
    s1 = _make_section("s1", text)
    s2 = _make_section("s2", text)
    s3 = _make_section("s3", text)

    pairs = detect_redundancy([s1, s2, s3], threshold=0.5)

    # Collect as frozensets so (a,b) and (b,a) would be equal
    pair_sets = [frozenset({p[0], p[1]}) for p in pairs]
    # No duplicate unordered pairs
    assert len(pair_sets) == len(set(pair_sets))

    # Each raw tuple should not have its reverse also present
    pair_tuples = [(p[0], p[1]) for p in pairs]
    for a, b in pair_tuples:
        assert (b, a) not in pair_tuples


def test_detect_redundancy_section_ids_in_output():
    """Output tuples contain the correct section IDs from the input list."""
    s1 = _make_section("faq_001", "Refund policy is thirty days.")
    s2 = _make_section("faq_002", "Refund policy is thirty days from purchase.")

    pairs = detect_redundancy([s1, s2], threshold=0.1)

    assert len(pairs) >= 1
    ids_in_pair = {pairs[0][0], pairs[0][1]}
    assert ids_in_pair == {"faq_001", "faq_002"}
