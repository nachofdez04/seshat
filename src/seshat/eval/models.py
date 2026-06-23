from __future__ import annotations

import hashlib
import json
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
    tags: dict[str, Any] = Field(default_factory=dict)
    query_node: RetrievalCorpusNode
    candidate_nodes: list[RetrievalCorpusNode]
    expected_relevant_ids: list[str]  # slugs from candidate_nodes


# ── Retrieval result ─────────────────────────────────────────────────────────


class RetrievalScoredResult(BaseModel):
    """Slug-keyed search results with scores."""

    results: list[tuple[str, float]]  # (slug, score) pairs, sorted desc by score


# ── Gate result ──────────────────────────────────────────────────────────────


class MetricEntry(BaseModel):
    value: float
    passed: bool


class GateResult(BaseModel):
    run_id: str
    timestamp: str = ""
    # dotted keys: "{ctype}.precision", "{ctype}.recall", "{ctype}.spurious_rate"
    identification_metrics: dict[str, MetricEntry] | None = None
    # dotted keys: "{ctype}.precision", "{ctype}.recall"
    resolution_metrics: dict[str, MetricEntry] | None = None
    # keys: "recall_at_5", "precision_at_5"
    retrieval_metrics: dict[str, MetricEntry] | None = None
    # keys: "precision", "recall"
    grounding_metrics: dict[str, MetricEntry] | None = None
    # keys: "group_hit_rate" (gated), "exact_match" (logged, not gated)
    grouping_metrics: dict[str, MetricEntry] | None = None
    validation_hash: str = ""

    @computed_field  # type: ignore[prop-decorator]
    @property
    def passed(self) -> bool:
        all_metrics = [
            self.identification_metrics,
            self.resolution_metrics,
            self.retrieval_metrics,
            self.grounding_metrics,
            self.grouping_metrics,
        ]
        if all(metric is None for metric in all_metrics):
            return False

        for harness_metrics in all_metrics:
            if harness_metrics is None:
                continue
            if not all(entry.passed for entry in harness_metrics.values()):
                return False

        return True

    def model_post_init(self, __context: object) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(UTC).isoformat()
        self.validation_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        payload = self.model_dump(exclude={"passed", "validation_hash"})
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode()).hexdigest()[:16]
