from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, computed_field, model_validator

from seshat.core.config.settings import PROJECT_ROOT
from seshat.core.models.enums import ConceptType, RelationshipType

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


# ── Transcription corpus ─────────────────────────────────────────────────────


class TranscriptionCorpusExample(BaseModel):
    corpus_id: str
    audio_file: Path  # repo-relative, resolved against PROJECT_ROOT
    reference: str  # spoken text only — no speaker labels, no comment lines
    # sha256 of the audio bytes, filled in by the corpus loader. It rides in the cache
    # fingerprint so regenerating a fixture invalidates its cached hypothesis.
    audio_sha256: str
    tags: dict[str, Any] = Field(default_factory=dict)

    @property
    def resolved_audio_path(self) -> Path:
        return PROJECT_ROOT / self.audio_file


# ── Transcription result ─────────────────────────────────────────────────────


class TranscriptionPrediction(BaseModel):
    """A provider's hypothesis for one corpus example."""

    text: str


# ── Gate result ──────────────────────────────────────────────────────────────


class MetricEntry(BaseModel):
    value: float
    # passed is the threshold verdict for GATED metrics; None for non-gated (informational)
    # metrics, which are logged but never checked against a threshold.
    gated: bool = True
    passed: bool | None = None

    @model_validator(mode="after")
    def _gated_requires_passed(self) -> MetricEntry:
        if self.gated and self.passed is None:
            raise ValueError("a gated MetricEntry must have a non-None `passed`")
        return self


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
    # keys: "wer" (gated, lower-is-better), "wer_macro" (logged, not gated)
    transcription_metrics: dict[str, MetricEntry] | None = None
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
            self.transcription_metrics,
        ]
        if all(metric is None for metric in all_metrics):
            return False

        for harness_metrics in all_metrics:
            if harness_metrics is None:
                continue
            if not all(entry.passed for entry in harness_metrics.values() if entry.gated):
                return False

        return True

    def harness_passed(self, harness: str) -> bool:
        """Whether this harness's own metric block passed, independent of the other blocks.

        Unlike `passed` (the AND of every present block), this isolates a single harness —
        so a green harness reads as green even when another block drags the overall gate down.
        Only gated metrics count; an absent (never-run) block is not a pass.
        """
        block: dict[str, MetricEntry] | None = getattr(self, f"{harness}_metrics")
        return block is not None and all(entry.passed for entry in block.values() if entry.gated)

    def model_post_init(self, __context: object) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(UTC).isoformat()
        self.validation_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        payload = self.model_dump(exclude={"passed", "validation_hash"})
        return self.compute_validation_hash(payload)

    @staticmethod
    def compute_validation_hash(payload: dict[str, Any]) -> str:
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode()).hexdigest()[:16]
