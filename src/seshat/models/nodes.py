from datetime import date, datetime
from uuid import UUID, uuid4

from pydantic import Field

from seshat.models.base import SeshatModel
from seshat.models.enums import (
    ApprovalMethod,
    ConceptType,
    IngestionSource,
    NodeState,
    NodeStatus,
    RelationshipType,
)


class KBRelationship(SeshatModel):
    source_id: UUID
    target_id: UUID
    rel_type: RelationshipType
    job_id: str
    created_at: datetime = Field(description="UTC timestamp when this relationship was written.")


class ConfidenceBreakdown(SeshatModel):
    logprobs: float | None = Field(
        default=None, ge=0, le=1, description="Log-probability signal from the LLM (optional, provider-dependent)."
    )
    verification: float | None = Field(
        default=None, ge=0, le=1, description="Score from the verification LLM call; None means disabled verification."
    )
    # heuristics is always active — never None; guards against divide-by-zero in the scorer.
    heuristics: float = Field(
        ge=0, le=1, description="Heuristic signal (always present); used as fallback when other signals are absent."
    )
    final: float = Field(ge=0, le=1, description="Weighted composite score in [0, 1].")


class NodeMetadata(SeshatModel):
    job_id: str
    meeting_date: date | None = None
    participants: list[str] | None = None
    ingestion_source: IngestionSource = Field(
        default=IngestionSource.JOB, description="Whether this node came from a job or an init run."
    )
    team: str | None = None
    project: str | None = None
    domain: str | None = None
    approved_by: str | None = None
    approved_at: datetime | None = Field(default=None, description="UTC timestamp of approval.")
    approval_method: ApprovalMethod | None = None
    corrected_by: str | None = None
    corrected_at: datetime | None = Field(default=None, description="UTC timestamp of the last correction.")
    confidence_breakdown: ConfidenceBreakdown | None = Field(
        default=None, description="Per-signal confidence breakdown; persisted for UI display."
    )


class KBNode(SeshatModel):
    id: UUID = Field(default_factory=uuid4)
    schema_version: str = Field(default="1.0", pattern=r"^\d+\.\d+$")
    type: ConceptType
    title: str = Field(min_length=1)
    description: str = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)
    source_quote: str = Field(min_length=1)
    status: NodeStatus
    state: NodeState = NodeState.CURRENT
    chunk_index: int | None = Field(
        default=None,
        ge=0,
        description="Source chunk position within the transcript; tiebreaker in within-meeting deduplication.",
    )
    metadata: NodeMetadata


class ResolutionCandidate(SeshatModel):
    node_id: UUID
    rel_type: RelationshipType
    candidate_title: str = Field(description="Title of the candidate node, shown in the review UI.")
    target_node_confidence: float = Field(
        ge=0, le=1, description="Confidence score of the candidate node; candidates are ordered DESC by this field."
    )


class ExtractionResult(SeshatModel):
    job_id: str
    nodes: list[KBNode]
    relationships: list[KBRelationship]
    confidence_breakdowns: dict[UUID, ConfidenceBreakdown]
    resolution_candidates: dict[UUID, list[ResolutionCandidate]]
