"""
core/models.py — Single source of truth for all Pydantic models.

All modules import from here. Never define Pydantic models elsewhere.
See context/data-models.md for full field documentation.
"""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ── Enumerations ───────────────────────────────────────────────────────────────

class SectionType(str, Enum):
    SYSTEM_PROMPT     = "system_prompt"
    FEW_SHOT_EXAMPLE  = "few_shot_example"
    RAG_DOCUMENT      = "rag_document"
    CONVERSATION_TURN = "conversation_turn"
    TOOL_DEFINITION   = "tool_definition"
    CUSTOM            = "custom"


class ExperimentMode(str, Enum):
    DEMO  = "demo"   # ~35 API calls,  2–4 min,   2 tiers, 1 rep, single-section only
    QUICK = "quick"  # ~150 API calls, 5–10 min,  3 tiers, 1 rep, + multi-section
    FULL  = "full"   # ~800 API calls, 15–30 min, 4 tiers, 3 reps, + ordering


# ── Input Models ───────────────────────────────────────────────────────────────

class ContextSection(BaseModel):
    id: str                    # Unique identifier, e.g. "faq_001", "turn_038"
    label: str                 # Human-readable name shown in UI
    section_type: SectionType
    content: str               # Raw text content
    token_count: int = 0       # Populated by parser via tiktoken; overwritten on parse
    metadata: dict = {}        # Optional extra data (source, priority, etc.)


class EvalQuery(BaseModel):
    query: str
    reference_answer: Optional[str] = None  # Enables binary correctness metric


class ContextPayload(BaseModel):
    sections: list[ContextSection]
    evaluation_queries: list[EvalQuery]
    quality_criteria: list[str] = ["relevance", "accuracy", "completeness", "groundedness"]
    total_tokens: int = 0  # Populated by parser; sum of all section token_counts


# ── Experiment Configuration ───────────────────────────────────────────────────

class ExperimentConfig(BaseModel):
    mode: ExperimentMode = ExperimentMode.QUICK
    quality_tolerance: float = 0.05    # Max acceptable quality loss fraction (5%)
    redundancy_threshold: float = 0.7  # TF-IDF cosine similarity cutoff
    repetitions: int = 1               # Overridden per mode from config.yaml
    reasoning_tiers: list[str] = ["disabled", "medium"]


# ── Scoring Models ─────────────────────────────────────────────────────────────

class CriterionScore(BaseModel):
    score: int          # 1–10 integer
    justification: str  # One-sentence rationale


class ScoringResult(BaseModel):
    """
    Returned by quality_scorer.py for each (response, query) pair.
    Field names match the quality_criteria list from ContextPayload.
    Default criteria fields shown below; additional criteria are added dynamically.
    """
    relevance:    CriterionScore = Field(description="Does the response address the query?")
    accuracy:     CriterionScore = Field(description="Are facts correct relative to the context?")
    completeness: CriterionScore = Field(description="Does it cover all aspects of the query?")
    groundedness: CriterionScore = Field(description="Is it grounded in the provided context?")

    def avg_score(self) -> float:
        """Average score across all criteria."""
        scores = [
            v["score"]
            for v in self.model_dump().values()
            if isinstance(v, dict) and "score" in v
        ]
        return sum(scores) / len(scores) if scores else 0.0


# ── Ablation Result Models ─────────────────────────────────────────────────────

class SectionImpact(BaseModel):
    section_id: str
    label: str
    section_type: str
    token_count: int
    avg_quality_delta: float       # Mean quality drop when this section is removed;
                                   # positive = section helps quality, negative = hurts
    quality_delta_by_tier: dict    # {"disabled": float, "low": float, ...}
    tier_sensitivity: float        # Variance of delta across tiers; higher = more tier-dependent
    classification: str            # "essential" | "moderate" | "removable" | "harmful"
    quality_per_token: float       # avg_quality_delta / token_count — efficiency metric


class AblationResults(BaseModel):
    baseline_scores: dict                  # {tier_name: {query_idx: ScoringResult}}
    section_impacts: list[SectionImpact]
    lean_configuration: list[str]          # Ordered section IDs in optimized context
    lean_quality_retention: float          # Fraction of baseline quality preserved (e.g. 0.97)
    lean_token_reduction: float            # Fraction of tokens removed (e.g. 0.58)
    ordering_recommendations: list[dict]   # [{"section_id": str, "best_position": int, "quality_gain": float}]
    redundancy_clusters: list[tuple]       # [(section_id_1, section_id_2, cosine_similarity)]
    pareto_configurations: list[dict]      # [{"section_ids": list, "quality": float, "tokens": int, "cost": float}]
    total_api_calls: int
    total_input_tokens: int
    total_output_tokens: int               # Includes reasoning tokens
    total_cost: float                      # USD, from API-reported token counts
