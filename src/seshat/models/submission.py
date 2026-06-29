from typing import Literal

from pydantic import BaseModel, Field

from seshat.config.settings import SeshatConfigOverride
from seshat.models.api_graph import NodeFilter
from seshat.models.transcript import TranscriptMetadata


class JobSubmissionRequest(BaseModel):
    source_type: Literal["audio", "text"]
    metadata: TranscriptMetadata
    idempotency_key: str | None = Field(
        default=None, description="Client-supplied key; re-submitting with the same key returns the existing job."
    )
    overrides: SeshatConfigOverride | None = Field(
        default=None, description="Per-request config overrides applied on top of the service defaults."
    )
    retrieval_filters: NodeFilter | None = Field(
        default=None, description="Filters applied to RAG hint retrieval for this job."
    )
