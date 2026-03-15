"""
core/assembler.py — Context assembly for Converse API calls.

Converts a list of ContextSection objects into Converse API parameters
(system prompt + user message) ready to unpack into BedrockClient.invoke().

Supports:
  - Ablation: exclude sections by ID via exclude_ids
  - Custom ordering: reorder non-system sections via an ordered list of IDs

See context/implementation-patterns.md for the reference implementation
and section tag format.
"""

from __future__ import annotations

from core.models import ContextSection, SectionType


def assemble_api_call(
    sections: list[ContextSection],
    eval_query: str,
    exclude_ids: set[str] = frozenset(),
    ordering: list[str] | None = None,
) -> dict:
    """Build Converse API parameters from context sections.

    System prompt sections go into the ``system`` field; all other sections
    are wrapped in typed tags and placed in the user message, followed by
    the evaluation query.

    Section tag format::

        [{section_type}: {label}]
        {content}
        [/{section_type}]

    Args:
        sections:    All sections from a ContextPayload.
        eval_query:  The evaluation question to append at the end of the
                     user message.
        exclude_ids: Section IDs to omit (used during ablation experiments).
        ordering:    If provided, non-system sections are sorted by their
                     position in this list. Sections not present in the
                     list are appended at the end, preserving their original
                     relative order among themselves.

    Returns:
        Dict with keys:
          - ``"system"``:   System prompt text (str), or ``None`` if no
                            system_prompt sections are active.
          - ``"messages"``: Converse API messages list — a single user turn
                            containing all context blocks and the query.
        Ready to unpack as ``**kwargs`` into ``BedrockClient.invoke()``.
    """
    # 1. Filter excluded sections
    active = [s for s in sections if s.id not in exclude_ids]

    # 2. Apply custom ordering to non-system sections
    if ordering:
        order_map = {sid: i for i, sid in enumerate(ordering)}
        active = sorted(
            active,
            key=lambda s: (
                # System prompts always sort before everything else
                0 if s.section_type == SectionType.SYSTEM_PROMPT else 1,
                # Non-system sections sorted by order_map; unlisted go last
                order_map.get(s.id, len(order_map)),
            ),
        )

    # 3. Split into system and non-system sections
    system_sections = [s for s in active if s.section_type == SectionType.SYSTEM_PROMPT]
    other_sections  = [s for s in active if s.section_type != SectionType.SYSTEM_PROMPT]

    # 4. Build system prompt text (None if no system sections)
    system_text: str | None = None
    if system_sections:
        system_text = "\n\n".join(s.content for s in system_sections)

    # 5. Build labeled context blocks for non-system sections
    context_blocks = [
        f"[{s.section_type.value}: {s.label}]\n{s.content}\n[/{s.section_type.value}]"
        for s in other_sections
    ]

    # 6. Assemble user message: context blocks + query separator
    if context_blocks:
        user_content = "\n\n".join(context_blocks) + f"\n\n---\n\nQuery: {eval_query}"
    else:
        user_content = f"---\n\nQuery: {eval_query}"

    return {
        "system": system_text,
        "messages": [
            {
                "role": "user",
                "content": [{"text": user_content}],
            }
        ],
    }
