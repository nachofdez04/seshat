from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


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
    def _at_least_one_field(self) -> ApproveRequest:
        if self.approve_above_threshold is None and not self.decisions:
            raise ValueError("ApproveRequest must set approve_above_threshold, decisions, or both")
        return self


class RateLimitError(BaseModel):
    limit_type: Literal["per_user_hourly_cap", "global_concurrency_cap"]
    retry_after_seconds: int | None = None
