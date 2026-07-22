from __future__ import annotations

from datetime import datetime
from typing import Annotated
from uuid import UUID

from pydantic import BaseModel, ConfigDict, StringConstraints, computed_field

from seshat.core.models.documents import DocumentKind
from seshat.core.models.enums import HealthStatus, UserRole
from seshat.core.models.nodes import KBNode, KBRelationship


class HealthResponse(BaseModel):
    status: HealthStatus
    components: dict[str, HealthStatus] | None = None


class NodeListResponse(BaseModel):
    nodes: list[KBNode]


class RelationshipListResponse(BaseModel):
    relationships: list[KBRelationship]


class NodeDetailResponse(BaseModel):
    node: KBNode
    neighbours: list[KBNode]
    relationships: list[KBRelationship] = []


class ImpactNode(BaseModel):
    node: KBNode
    traversal_depth: int


class ImpactResponse(BaseModel):
    nodes: list[ImpactNode]
    relationships: list[KBRelationship] = []


class JobSubmitResponse(BaseModel):
    job_id: str


class JobActionResponse(BaseModel):
    status: str


class ApiKeyResponse(BaseModel):
    id: int
    user_id: str
    role: UserRole
    created_at: datetime
    revoked_at: datetime | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_active(self) -> bool:
        return self.revoked_at is None


class CreateApiKeyRequest(BaseModel):
    user_id: Annotated[str, StringConstraints(min_length=1, strip_whitespace=True)]
    role: UserRole


class CreateApiKeyResponse(BaseModel):
    api_key: str
    user_id: str
    role: UserRole


class TranscriptExcerptResponse(BaseModel):
    text: str
    char_start: int
    char_end: int


class GeneratedDocumentMeta(BaseModel):
    """Document metadata without markdown_content, for list endpoints."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    job_id: str
    kind: DocumentKind
    filename: str
    content_revision: str
    created_at: datetime


class NodeSearchResult(BaseModel):
    detail: NodeDetailResponse
    score: float | None = None


class NodeSearchResponse(BaseModel):
    results: list[NodeSearchResult]
