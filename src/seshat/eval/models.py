from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field, computed_field

from seshat.eval.thresholds import (
    GROUPING_GROUP_HIT_RATE,
    IDENTIFICATION_PRECISION,
    IDENTIFICATION_RECALL,
    IDENTIFICATION_SPURIOUS_RATE,
    RESOLUTION_PRECISION,
    RESOLUTION_RECALL,
    RETRIEVAL_RECALL_AT_5,
    VERIFICATION_PRECISION,
    VERIFICATION_RECALL,
)
from seshat.models.enums import ConceptType, RelationshipType

# ── Identification corpus ────────────────────────────────────────────────────


class IdentificationCorpusNode(BaseModel):
    quote: str  # ground-truth quote used by span-overlap matcher
    type: ConceptType
    title: str
    description: str
    extra_fields: dict[str, Any] = Field(
        default_factory=dict,
        description="Type-specific expected field values (assignee, due, rationale, etc.).",
    )


class IdentificationCorpusExample(BaseModel):
    corpus_id: str
    transcript: str
    expected_nodes: list[IdentificationCorpusNode]
    tags: dict[str, Any] = Field(default_factory=dict)


# ── Resolution corpus ────────────────────────────────────────────────────────


class ResolutionCorpusNode(BaseModel):
    id: str  # human-readable slug — local cross-reference key only
    type: ConceptType
    title: str
    description: str
    quote: str


class ResolutionCorpusRelation(BaseModel):
    source: str  # slug
    target: str  # slug
    rel_type: RelationshipType


class ResolutionCorpusExample(BaseModel):
    corpus_id: str
    description: str
    source_nodes: list[ResolutionCorpusNode]
    kb_nodes: list[ResolutionCorpusNode]
    expected_relations: list[ResolutionCorpusRelation]
    tags: dict[str, Any] = Field(default_factory=dict)


# ── Retrieval corpus ─────────────────────────────────────────────────────────


class RetrievalCorpusNode(BaseModel):
    id: str  # slug
    type: ConceptType
    title: str
    description: str
    quote: str


class RetrievalCorpusExample(BaseModel):
    corpus_id: str
    description: str
    query_node: RetrievalCorpusNode
    candidate_nodes: list[RetrievalCorpusNode]
    expected_relevant_ids: list[str]  # slugs from candidate_nodes


# ── Retrieval result ─────────────────────────────────────────────────────────


class RetrievalResult(BaseModel):
    retrieved_ids: list[str]


# ── Gate result ──────────────────────────────────────────────────────────────


class GateResult(BaseModel):
    run_id: str
    timestamp: str = ""
    # dotted keys: "{ctype}.precision", "{ctype}.recall", "{ctype}.f1"
    identification_metrics: dict[str, float] | None = None
    # dotted keys: "{ctype}.precision", "{ctype}.recall"
    resolution_metrics: dict[str, float] | None = None
    # keys: "recall_at_5", "precision_at_5"
    retrieval_metrics: dict[str, float] | None = None
    # keys: "precision", "recall"
    verification_metrics: dict[str, float] | None = None
    # keys: "group_hit_rate" (gated), "exact_match" (logged, not gated)
    grouping_metrics: dict[str, float] | None = None

    @computed_field  # type: ignore[misc]
    @property
    def passed(self) -> bool:
        if self._all_metrics_are_none():
            return False
        return (
            self._identification_passes()
            and self._resolution_passes()
            and self._retrieval_passes()
            and self._grouping_passes()
            and self._verification_passes()
        )

    def model_post_init(self, __context: object) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(UTC).isoformat()

    def _all_metrics_are_none(self) -> bool:
        return (
            self.identification_metrics is None
            and self.resolution_metrics is None
            and self.retrieval_metrics is None
            and self.verification_metrics is None
            and self.grouping_metrics is None
        )

    def _identification_passes(self) -> bool:
        if self.identification_metrics is None:
            return True  # identification not evaluated, skip to other metrics
        return all(
            self.identification_metrics.get(f"{ctype}.precision", 0.0) >= IDENTIFICATION_PRECISION[ctype]
            and self.identification_metrics.get(f"{ctype}.recall", 0.0) >= IDENTIFICATION_RECALL[ctype]
            and self.identification_metrics.get(f"{ctype}.spurious_rate", 0.0) <= IDENTIFICATION_SPURIOUS_RATE[ctype]
            for ctype in ConceptType
        )

    def _resolution_passes(self) -> bool:
        if self.resolution_metrics is None:
            return True
        return all(
            self.resolution_metrics.get(f"{ctype}.precision", 0.0) >= RESOLUTION_PRECISION[ctype]
            and self.resolution_metrics.get(f"{ctype}.recall", 0.0) >= RESOLUTION_RECALL[ctype]
            for ctype in ConceptType
        )

    def _retrieval_passes(self) -> bool:
        if self.retrieval_metrics is None:
            return True
        return self.retrieval_metrics.get("recall_at_5", 0.0) >= RETRIEVAL_RECALL_AT_5

    def _grouping_passes(self) -> bool:
        if self.grouping_metrics is None:
            return True
        return self.grouping_metrics.get("group_hit_rate", 0.0) >= GROUPING_GROUP_HIT_RATE

    def _verification_passes(self) -> bool:
        if self.verification_metrics is None:
            return True
        return (
            self.verification_metrics.get("precision", 0.0) >= VERIFICATION_PRECISION
            and self.verification_metrics.get("recall", 0.0) >= VERIFICATION_RECALL
        )
