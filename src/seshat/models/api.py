from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from seshat.models.enums import ConceptType, IngestionSource, NodeState


class NodeFilter(BaseModel):
    node_type: ConceptType | None = Field(default=None, description="Filter by concept type.")
    job_id: str | None = Field(default=None, description="Filter by job ID.")
    team: str | None = Field(default=None, description="Filter by team tag.")
    project: str | None = Field(default=None, description="Filter by project tag.")
    domain: str | None = Field(default=None, description="Filter by domain tag.")
    ingestion_source: IngestionSource | None = Field(default=None, description="Filter by ingestion source.")
    min_confidence: float | None = Field(default=None, ge=0, le=1, description="Minimum confidence score (inclusive).")
    state: NodeState | None = Field(default=None, description="Filter by lifecycle state.")
    meeting_date_from: date | None = Field(
        default=None, description="Include nodes from meetings on or after this date."
    )
    meeting_date_to: date | None = Field(
        default=None, description="Include nodes from meetings on or before this date."
    )
    limit: int = Field(default=1000, gt=0, le=10_000, description="Maximum number of nodes to return.")
    offset: int = Field(default=0, ge=0, description="Number of nodes to skip (for pagination).")

    @model_validator(mode="after")
    def check_date_range(self) -> "NodeFilter":
        if self.meeting_date_to is not None:
            if self.meeting_date_from is not None and self.meeting_date_from > self.meeting_date_to:
                raise ValueError("meeting_date_from must be before or equal to meeting_date_to")
            if self.meeting_date_to > date.today():
                raise ValueError(f"meeting_date_to must not be in the future (got {self.meeting_date_to})")
        return self


class SearchResult(BaseModel):
    node_id: str
    score: float = Field(ge=0, le=1)


class KBNodeEdit(BaseModel):
    title: str = Field(min_length=1)
    description: str = Field(min_length=1)


class NodeDecision(BaseModel):
    node_id: str
    action: Literal["approve", "reject"]
    edited_content: KBNodeEdit | None = Field(
        default=None, description="Optional content correction applied before approval."
    )
    reason: str | None = Field(default=None, description="Optional reason for rejection.")


class BulkApproveRule(BaseModel):
    threshold: float = Field(
        ge=0.5,
        lt=1,
        description="Minimum confidence to auto-approve. Floor of 0.5 prevents bulk-approving hallucinations.",
    )
    exclude: list[str] | None = Field(default=None, description="Node IDs to exclude from bulk approval.")


class ApproveRequest(BaseModel):
    approve_above_threshold: BulkApproveRule | None = Field(
        default=None, description="Bulk approval rule applied first; individual decisions override it."
    )
    decisions: list[NodeDecision] | None = Field(
        default=None, description="Individual node decisions; always override a bulk-rule outcome for the same node."
    )

    @model_validator(mode="after")
    def _at_least_one_field(self) -> "ApproveRequest":
        if self.approve_above_threshold is None and not self.decisions:
            raise ValueError("ApproveRequest must set approve_above_threshold, decisions, or both")
        return self


class RateLimitError(BaseModel):
    limit_type: Literal["per_user_hourly_cap", "global_concurrency_cap"]
    retry_after_seconds: int | None = None
