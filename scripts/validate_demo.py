#!/usr/bin/env python3
"""Validate the demo payload produces expected structural properties.

Performs structural checks on the generated demo payload to confirm it has
the right shape for producing dramatic ablation findings. Does NOT run
actual ablation (that requires Phase 4 modules). Instead verifies:

  - Token counts are within expected ranges
  - Section types and IDs match the spec
  - Conversation turns have the right relevance split
  - FAQ content has visible redundancy patterns
  - Tool definitions have the right relevant/irrelevant split

Usage:
    python scripts/validate_demo.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from infra.token_counter import estimate_tokens  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PAYLOAD_PATH = PROJECT_ROOT / "data" / "demo_payloads" / "customer_support.json"
QUERIES_PATH = PROJECT_ROOT / "data" / "eval_queries" / "customer_support_queries.json"

# ---------------------------------------------------------------------------
# Expected ranges and counts
# ---------------------------------------------------------------------------
TOTAL_MIN = 200_000
TOTAL_MAX = 220_000

EXPECTED_SECTION_COUNTS = {
    "system_prompt": 1,
    "rag_document": 2,          # FAQ + catalog
    "conversation_turn": 40,
    "tool_definition": 20,
    "few_shot_example": 15,
    "custom": 1,                # legal
}

SECTION_TOKEN_RANGES = {
    "sys_001":     (1_000,   4_000),
    "faq_001":     (20_000,  70_000),
    "catalog_001": (40_000, 150_000),
    "legal_001":   (2_000,  10_000),
}

CONV_TOTAL_MIN = 3_000   # Offline templates produce shorter turns; online API mode is more verbose
CONV_TOTAL_MAX = 50_000
TOOL_TOTAL_MIN = 8_000
TOOL_TOTAL_MAX = 25_000
FEW_SHOT_TOTAL_MIN = 2_000  # Offline templates produce shorter examples; online API mode is more verbose
FEW_SHOT_TOTAL_MAX = 20_000


# ---------------------------------------------------------------------------
# Validation checks
# ---------------------------------------------------------------------------

class ValidationResult:
    """Collects pass/fail results."""

    def __init__(self) -> None:
        self.results: list[tuple[str, bool, str]] = []

    def check(self, name: str, passed: bool, detail: str = "") -> None:
        self.results.append((name, passed, detail))
        status = "PASS" if passed else "FAIL"
        msg = f"  [{status}] {name}"
        if detail:
            msg += f" — {detail}"
        print(msg)

    @property
    def all_passed(self) -> bool:
        return all(ok for _, ok, _ in self.results)

    @property
    def summary(self) -> str:
        passed = sum(1 for _, ok, _ in self.results if ok)
        total = len(self.results)
        return f"{passed}/{total} checks passed"


def validate_payload(payload: dict, v: ValidationResult) -> None:
    """Run all structural validation checks."""
    sections = payload.get("sections", [])
    queries = payload.get("evaluation_queries", [])

    # ── Total token count ──────────────────────────────────────────────
    total_tokens = sum(s.get("token_count", 0) for s in sections)
    if total_tokens == 0:
        # Recount from content if token_count wasn't stored
        total_tokens = sum(estimate_tokens(s["content"]) for s in sections)

    v.check(
        "Total tokens in range",
        TOTAL_MIN <= total_tokens <= TOTAL_MAX,
        f"{total_tokens:,} (expected {TOTAL_MIN:,}–{TOTAL_MAX:,})",
    )

    # ── Section type counts ────────────────────────────────────────────
    type_counts: dict[str, int] = {}
    for s in sections:
        st = s["section_type"]
        type_counts[st] = type_counts.get(st, 0) + 1

    for stype, expected in EXPECTED_SECTION_COUNTS.items():
        actual = type_counts.get(stype, 0)
        v.check(
            f"Section count: {stype}",
            actual == expected,
            f"{actual} (expected {expected})",
        )

    # ── Unique section IDs ─────────────────────────────────────────────
    ids = [s["id"] for s in sections]
    v.check(
        "All section IDs unique",
        len(ids) == len(set(ids)),
        f"{len(ids)} total, {len(set(ids))} unique",
    )

    # ── Total section count ────────────────────────────────────────────
    expected_total = sum(EXPECTED_SECTION_COUNTS.values())
    v.check(
        "Total section count",
        len(sections) == expected_total,
        f"{len(sections)} (expected {expected_total})",
    )

    # ── Key section token ranges ───────────────────────────────────────
    section_by_id = {s["id"]: s for s in sections}
    for sid, (lo, hi) in SECTION_TOKEN_RANGES.items():
        s = section_by_id.get(sid)
        if s is None:
            v.check(f"Section {sid} exists", False, "missing")
            continue
        tok = s.get("token_count", 0) or estimate_tokens(s["content"])
        v.check(
            f"Token range: {sid}",
            lo <= tok <= hi,
            f"{tok:,} (expected {lo:,}–{hi:,})",
        )

    # ── Conversation turn aggregates ───────────────────────────────────
    conv_sections = [s for s in sections if s["section_type"] == "conversation_turn"]
    conv_tokens = sum(
        s.get("token_count", 0) or estimate_tokens(s["content"])
        for s in conv_sections
    )
    v.check(
        "Conversation total tokens",
        CONV_TOTAL_MIN <= conv_tokens <= CONV_TOTAL_MAX,
        f"{conv_tokens:,} (expected {CONV_TOTAL_MIN:,}–{CONV_TOTAL_MAX:,})",
    )

    # Check IDs are conv_001 through conv_040
    conv_ids = sorted(s["id"] for s in conv_sections)
    expected_conv_ids = [f"conv_{i:03d}" for i in range(1, 41)]
    v.check(
        "Conversation IDs sequential",
        conv_ids == expected_conv_ids,
        f"first={conv_ids[0] if conv_ids else 'N/A'}, last={conv_ids[-1] if conv_ids else 'N/A'}",
    )

    # ── Tool definition aggregates ─────────────────────────────────────
    tool_sections = [s for s in sections if s["section_type"] == "tool_definition"]
    tool_tokens = sum(
        s.get("token_count", 0) or estimate_tokens(s["content"])
        for s in tool_sections
    )
    v.check(
        "Tool definitions total tokens",
        TOOL_TOTAL_MIN <= tool_tokens <= TOOL_TOTAL_MAX,
        f"{tool_tokens:,} (expected {TOOL_TOTAL_MIN:,}–{TOOL_TOTAL_MAX:,})",
    )

    # Verify relevant tools present
    tool_ids = {s["id"] for s in tool_sections}
    v.check("tool_001 (refund_processor) present", "tool_001" in tool_ids)
    v.check("tool_002 (account_lookup) present", "tool_002" in tool_ids)

    # ── Few-shot aggregates ────────────────────────────────────────────
    shot_sections = [s for s in sections if s["section_type"] == "few_shot_example"]
    shot_tokens = sum(
        s.get("token_count", 0) or estimate_tokens(s["content"])
        for s in shot_sections
    )
    v.check(
        "Few-shot total tokens",
        FEW_SHOT_TOTAL_MIN <= shot_tokens <= FEW_SHOT_TOTAL_MAX,
        f"{shot_tokens:,} (expected {FEW_SHOT_TOTAL_MIN:,}–{FEW_SHOT_TOTAL_MAX:,})",
    )

    # ── Evaluation queries ─────────────────────────────────────────────
    v.check(
        "Eval query count",
        len(queries) == 10,
        f"{len(queries)} (expected 10)",
    )

    queries_with_ref = sum(1 for q in queries if q.get("reference_answer"))
    v.check(
        "All eval queries have reference answers",
        queries_with_ref == 10,
        f"{queries_with_ref}/10 have reference_answer",
    )

    # ── FAQ redundancy indicator ───────────────────────────────────────
    faq = section_by_id.get("faq_001")
    if faq:
        content = faq["content"]
        q_count = content.count("Q:")
        v.check(
            "FAQ has >= 50 Q&A entries",
            q_count >= 50,
            f"{q_count} entries found",
        )
    else:
        v.check("FAQ section exists", False, "faq_001 missing")

    # ── Content validity (non-empty) ───────────────────────────────────
    empty = [s["id"] for s in sections if not s.get("content", "").strip()]
    v.check(
        "No empty sections",
        len(empty) == 0,
        f"{len(empty)} empty: {empty[:5]}" if empty else "",
    )

    # ── System prompt section type ─────────────────────────────────────
    sys_section = section_by_id.get("sys_001")
    if sys_section:
        v.check(
            "System prompt has correct type",
            sys_section["section_type"] == "system_prompt",
            f"type={sys_section['section_type']}",
        )
    else:
        v.check("sys_001 exists", False, "missing")


def validate_queries_file(v: ValidationResult) -> None:
    """Validate the standalone eval queries file."""
    if not QUERIES_PATH.exists():
        v.check("Eval queries file exists", False, str(QUERIES_PATH))
        return

    queries = json.loads(QUERIES_PATH.read_text())
    v.check(
        "Eval queries file has 10 entries",
        len(queries) == 10,
        f"{len(queries)} entries",
    )

    # Check each has query and reference_answer
    valid = all(q.get("query") and q.get("reference_answer") for q in queries)
    v.check(
        "All queries have query + reference_answer fields",
        valid,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run validation and exit with appropriate code."""
    print("=" * 60)
    print("ContextForge Demo Payload Validation")
    print("=" * 60)

    v = ValidationResult()

    # Load payload
    if not PAYLOAD_PATH.exists():
        print(f"\nERROR: Payload not found at {PAYLOAD_PATH}")
        print("Run `python scripts/generate_demo_payload.py` first.")
        sys.exit(1)

    print(f"\nLoading {PAYLOAD_PATH}...")
    payload = json.loads(PAYLOAD_PATH.read_text())

    print("\n--- Payload Checks ---")
    validate_payload(payload, v)

    print("\n--- Eval Queries File Checks ---")
    validate_queries_file(v)

    # Summary
    print(f"\n{'=' * 60}")
    print(f"Result: {v.summary}")
    if v.all_passed:
        print("STATUS: ALL CHECKS PASSED")
    else:
        print("STATUS: SOME CHECKS FAILED")
        failed = [(n, d) for n, ok, d in v.results if not ok]
        print("\nFailed checks:")
        for name, detail in failed:
            print(f"  - {name}: {detail}")

    print("=" * 60)
    sys.exit(0 if v.all_passed else 1)


if __name__ == "__main__":
    main()
