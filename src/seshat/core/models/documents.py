from datetime import datetime
from enum import StrEnum, auto
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, model_validator

from seshat.core.models.base import SeshatModel
from seshat.core.utils.hashing import sha256_text


class DocumentKind(StrEnum):
    MEETING_SUMMARY = auto()


class DocumentValidationStatus(StrEnum):
    PENDING = auto()
    APPROVED = auto()
    REJECTED = auto()
    EDITED = auto()


class DocumentReviewRequest(BaseModel):
    action: Literal["approve", "reject"]
    expected_revision: str = Field(
        description="content_revision the reviewer saw — optimistic concurrency token, 409 on mismatch."
    )
    expected_validation_revision: int = Field(
        ge=0,
        description="validation_revision the reviewer saw — prevents concurrent decisions from overwriting each other.",
    )
    edited_content: str | None = Field(
        default=None, description="Approve-with-edits content; invalid with action='reject'."
    )
    reason: str | None = Field(default=None, description="Rejection reason; ignored on approve.")

    @model_validator(mode="after")
    def _edited_only_on_approve(self) -> "DocumentReviewRequest":
        if self.action == "reject" and self.edited_content is not None:
            raise ValueError("edited_content is only valid with action='approve'")
        return self


class GeneratedDocument(SeshatModel):
    id: UUID = Field(default_factory=uuid4)
    job_id: str
    kind: DocumentKind
    filename: str = Field(description="Markdown filename, e.g. 'meeting_summary.md'.")
    markdown_content: str
    content_revision: str = Field(
        description="sha256 hex of markdown_content, set at write time — the validation/publishing anchor."
    )
    created_at: datetime = Field(description="UTC timestamp when this document was generated.")
    validation_status: DocumentValidationStatus = DocumentValidationStatus.PENDING
    validation_revision: int = Field(default=0, ge=0)
    edited_content: str | None = Field(default=None, description="Reviewer-edited content, set on approve-with-edits.")
    rejection_reason: str | None = None
    validated_by: str | None = Field(default=None, description="user_id of the reviewer; None for auto-approvals.")
    validated_at: datetime | None = Field(default=None, description="UTC timestamp of the review decision.")
    auto_approved: bool = Field(default=False, description="True when approved by config rule, not a human.")
    approved_revision: str | None = Field(
        default=None, description="sha256 hex of the effective content at approval time."
    )


def effective_content(doc: GeneratedDocument) -> str:
    """Return the content that publishing would emit: the reviewer's edit if present, else the original."""
    return doc.edited_content if doc.edited_content is not None else doc.markdown_content


def document_is_publishable(doc: GeneratedDocument) -> bool:
    """Return True only if the document is approved and the approval still matches its effective content."""
    if doc.validation_status not in (DocumentValidationStatus.APPROVED, DocumentValidationStatus.EDITED):
        return False

    return doc.approved_revision == sha256_text(effective_content(doc))
