from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field, computed_field

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
    # dotted keys: "{ctype.value}.precision", "{ctype.value}.recall", "{ctype.value}.f1"
    identification_metrics: dict[str, float] | None = None
    # keys: "precision", "recall", "f1"
    resolution_metrics: dict[str, float] | None = None
    # keys: "recall_at_5", "precision_at_5"
    retrieval_metrics: dict[str, float] | None = None

    @computed_field  # type: ignore[misc]
    @property
    def passed(self) -> bool:
        from seshat.eval.thresholds import (
            IDENTIFICATION_PRECISION,
            IDENTIFICATION_RECALL,
            RESOLUTION_PRECISION,
            RESOLUTION_RECALL,
            RETRIEVAL_RECALL_AT_5,
        )

        if self.identification_metrics is None and self.resolution_metrics is None and self.retrieval_metrics is None:
            return False

        if self.identification_metrics is not None:
            for ctype in ConceptType:
                if self.identification_metrics.get(f"{ctype.value}.precision", 0.0) < IDENTIFICATION_PRECISION[ctype]:
                    return False
                if self.identification_metrics.get(f"{ctype.value}.recall", 0.0) < IDENTIFICATION_RECALL[ctype]:
                    return False

        if self.resolution_metrics is not None:
            if self.resolution_metrics.get("precision", 0.0) < RESOLUTION_PRECISION:
                return False
            if self.resolution_metrics.get("recall", 0.0) < RESOLUTION_RECALL:
                return False

        if (  # noqa: SIM103
            self.retrieval_metrics is not None
            and self.retrieval_metrics.get("recall_at_5", 0.0) < RETRIEVAL_RECALL_AT_5
        ):
            return False

        return True

    def model_post_init(self, __context: object) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(UTC).isoformat()
