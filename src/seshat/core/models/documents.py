from datetime import datetime
from enum import StrEnum, auto
from uuid import UUID, uuid4

from pydantic import Field

from seshat.core.models.base import SeshatModel


class DocumentKind(StrEnum):
    MEETING_SUMMARY = auto()


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
