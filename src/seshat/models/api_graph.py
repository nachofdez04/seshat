from __future__ import annotations

from datetime import date
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field, model_validator

from seshat.models.enums import ConceptType, IngestionSource, NodeState, NodeStatus, RelationshipType


class NodeFilter(BaseModel):
    node_type: ConceptType | None = Field(default=None, description="Filter by concept type.")
    job_id: str | None = Field(default=None, description="Filter by job ID.")
    team: str | None = Field(default=None, description="Filter by team tag.")
    project: str | None = Field(default=None, description="Filter by project tag.")
    domain: str | None = Field(default=None, description="Filter by domain tag.")
    ingestion_source: IngestionSource | None = Field(default=None, description="Filter by ingestion source.")
    min_confidence: float | None = Field(default=None, ge=0, le=1, description="Minimum confidence score (inclusive).")
    state: NodeState | None = Field(default=None, description="Filter by lifecycle state.")
    status: NodeStatus | None = Field(default=None, description="Filter by approval status.")
    meeting_date_from: date | None = Field(
        default=None, description="Include nodes from meetings on or after this date."
    )
    meeting_date_to: date | None = Field(
        default=None, description="Include nodes from meetings on or before this date."
    )
    limit: int = Field(default=1000, gt=0, le=10_000, description="Maximum number of nodes to return.")
    offset: int = Field(default=0, ge=0, description="Number of nodes to skip (for pagination).")

    @model_validator(mode="after")
    def check_date_range(self) -> NodeFilter:
        if self.meeting_date_to is not None:
            if self.meeting_date_from is not None and self.meeting_date_from > self.meeting_date_to:
                raise ValueError("meeting_date_from must be before or equal to meeting_date_to")
            if self.meeting_date_to > date.today():
                raise ValueError(f"meeting_date_to must not be in the future (got {self.meeting_date_to})")
        return self


class SearchResult(BaseModel):
    node_id: str
    score: float = Field(ge=0)


class RelationshipInput(BaseModel):
    target_id: str
    rel_type: RelationshipType


class ManualNodeCreate(BaseModel):
    type: ConceptType
    title: str
    description: str
    source_quote: str | None = None
    blob_key: str | None = None
    participants: list[str] | None = None
    team: str | None = None
    project: str | None = None
    domain: str | None = None
    meeting_date: date | None = None
    concept_fields: dict[str, Any] | None = None
    relationships: list[RelationshipInput] | None = None

    @model_validator(mode="after")
    def _co_required_quote_fields(self) -> ManualNodeCreate:
        if (self.source_quote is None) != (self.blob_key is None):
            raise ValueError("source_quote and blob_key are co-required: provide both or neither")
        return self


class ManualNodeUpdate(BaseModel):
    title: str
    description: str
    participants: list[str] | None = None
    team: str | None = None
    project: str | None = None
    domain: str | None = None
    meeting_date: date | None = None
    concept_fields: dict[str, Any] | None = None
    relationships: list[RelationshipInput] | None = None
    reason: str | None = None


class NodeOverride(ManualNodeUpdate):
    reason: str = Field(...)  # type: ignore[override]


class BulkNodeCreate(BaseModel):
    nodes: list[ManualNodeCreate]
    on_error: Literal["stop", "continue"] = "stop"


class BulkNodeDelete(BaseModel):
    node_ids: list[str]
    on_error: Literal["stop", "continue"] = "stop"


class BulkFailure(BaseModel):
    node_id: str
    error: str


class BulkResult(BaseModel):
    succeeded: list[str]
    failed: list[BulkFailure]


class ResolveRequest(BaseModel):
    node_ids: list[UUID] = Field(..., min_length=1, max_length=50)


class ResolveResponse(BaseModel):
    relationships_created: int
