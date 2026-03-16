"""
core/redundancy.py — Context section redundancy detection via TF-IDF cosine similarity.

Default implementation uses scikit-learn TF-IDF for local computation — no API calls.
A stretch-goal path using Nova Multimodal Embeddings is provided in
detect_redundancy_embeddings(); it falls back to TF-IDF until fully implemented.

Critical rules:
  - detect_redundancy() is purely local — never calls the Bedrock API.
  - detect_redundancy_embeddings() is the ONE permitted use of invoke_model() in
    the codebase (embeddings model does not support the Converse API).
"""

from __future__ import annotations

import logging

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from core.models import ContextSection

logger = logging.getLogger(__name__)

_DEFAULT_REDUNDANCY_THRESHOLD = 0.7


def detect_redundancy(
    sections:  list[ContextSection],
    threshold: float = _DEFAULT_REDUNDANCY_THRESHOLD,
) -> list[tuple[str, str, float]]:
    """Detect redundant section pairs using TF-IDF cosine similarity.

    Builds a TF-IDF matrix from section content and computes pairwise cosine
    similarity. Returns section pairs whose similarity meets or exceeds the
    threshold.

    All computation is local — no API calls.

    Args:
        sections:  List of context sections to compare.
        threshold: Cosine similarity cutoff (default 0.7). Pairs at or above
                   this value are returned as redundant.

    Returns:
        List of (section_id_1, section_id_2, similarity) tuples for pairs
        meeting the threshold, sorted by similarity descending.
        Returns an empty list if fewer than 2 sections are provided.
    """
    if len(sections) < 2:
        return []

    texts = [s.content for s in sections]
    ids   = [s.id for s in sections]

    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    try:
        tfidf_matrix = vectorizer.fit_transform(texts)
    except ValueError:
        # Raised when vocabulary is empty (e.g. all content is stop words or blank)
        logger.warning(
            "TF-IDF vectorization failed (empty vocabulary) — no redundancy detected."
        )
        return []

    sim_matrix = cosine_similarity(tfidf_matrix)

    redundant: list[tuple[str, str, float]] = []
    n = len(sections)
    for i in range(n):
        for j in range(i + 1, n):
            sim = float(sim_matrix[i, j])
            if sim >= threshold:
                redundant.append((ids[i], ids[j], sim))

    return sorted(redundant, key=lambda x: x[2], reverse=True)


def detect_redundancy_embeddings(
    sections:       list[ContextSection],
    threshold:      float = _DEFAULT_REDUNDANCY_THRESHOLD,
    bedrock_client: object | None = None,
) -> list[tuple[str, str, float]]:
    """[STRETCH GOAL] Detect redundancy using Nova Multimodal Embeddings.

    Uses amazon.nova-2-multimodal-embeddings-v1:0 via invoke_model() — the one
    permitted use of invoke_model() in the codebase. Only activated when
    config.yaml stretch_goals.nova_embeddings = true.

    Falls back to detect_redundancy() (TF-IDF) if bedrock_client is None or
    the stretch goal implementation is not yet active.

    Args:
        sections:       Context sections to compare.
        threshold:      Similarity threshold (default 0.7).
        bedrock_client: BedrockClient instance. If None, falls back to TF-IDF.

    Returns:
        Same format as detect_redundancy(): [(id1, id2, similarity), ...].
    """
    if bedrock_client is None:
        logger.info(
            "Nova Embeddings: bedrock_client is None — falling back to TF-IDF."
        )
        return detect_redundancy(sections, threshold)

    # Stretch goal placeholder — full Nova Embeddings implementation goes here.
    logger.info(
        "Nova Embeddings stretch goal not yet implemented — falling back to TF-IDF."
    )
    return detect_redundancy(sections, threshold)
