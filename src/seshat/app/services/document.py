from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from seshat.app.documents.renderer import render_meeting_summary
from seshat.app.services.job import JobNotFoundError, JobStateError, TranscriptNotFoundError
from seshat.core.models.api_graph import NodeFilter
from seshat.core.models.documents import DocumentKind, GeneratedDocument
from seshat.core.models.enums import JobStatus, NodeStatus
from seshat.core.utils.hashing import sha256_text
from seshat.core.utils.log import get_logger

if TYPE_CHECKING:
    from uuid import UUID

    from seshat.app.repositories.blob_repository import BlobRepository
    from seshat.app.repositories.node_repository import NodeRepository
    from seshat.app.repositories.ops_repository import OpsRepository

logger = get_logger(__name__)

_FILENAMES = {DocumentKind.MEETING_SUMMARY: "meeting_summary.md"}


class DocumentService:
    def __init__(self, ops: OpsRepository, blob: BlobRepository, node_repo: NodeRepository) -> None:
        self._ops = ops
        self._blob = blob
        self._node_repo = node_repo

    async def generate_for_job(self, job_id: str) -> GeneratedDocument:
        row = await self._ops.get_job(job_id)
        if not row:
            raise JobNotFoundError(job_id)
        if row["status"] != JobStatus.DONE:
            raise JobStateError("Documents can only be generated for done jobs")

        nodes = await self._node_repo.paginated_query(NodeFilter(job_id=job_id, status=NodeStatus.APPROVED))

        meeting_date = row["meeting_date"]
        blob = await self._blob.get_raw_transcript(meeting_date, job_id) if meeting_date is not None else None
        if blob is None:
            raise TranscriptNotFoundError(job_id)

        transcript = blob.decode("utf-8", errors="replace")
        markdown = render_meeting_summary(job_id, meeting_date, nodes, transcript)

        kind = DocumentKind.MEETING_SUMMARY
        document = GeneratedDocument(
            job_id=job_id,
            kind=kind,
            filename=_FILENAMES[kind],
            markdown_content=markdown,
            content_revision=sha256_text(markdown),
            created_at=datetime.now(UTC),
        )
        stored = await self._ops.upsert_document(document)
        logger.info("Generated %s for job %s from %d approved node(s)", document.filename, job_id, len(nodes))
        return GeneratedDocument.model_validate(stored)

    async def list_for_job(self, job_id: str) -> list[GeneratedDocument]:
        if not await self._ops.get_job(job_id):
            raise JobNotFoundError(job_id)

        rows = await self._ops.get_documents_for_job(job_id)
        return [GeneratedDocument.model_validate(row) for row in rows]

    async def get(self, document_id: UUID) -> GeneratedDocument | None:
        row = await self._ops.get_document(document_id)
        return GeneratedDocument.model_validate(row) if row else None
