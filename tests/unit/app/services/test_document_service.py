from __future__ import annotations

from datetime import UTC, date, datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from seshat.app.services.document import (
    DocumentNotFoundError,
    DocumentRevisionConflictError,
    DocumentService,
)
from seshat.app.services.job import JobNotFoundError, JobStateError, TranscriptNotFoundError
from seshat.core.config.settings import DocumentsConfig
from seshat.core.models.documents import (
    DocumentKind,
    DocumentReviewRequest,
    DocumentValidationStatus,
    GeneratedDocument,
)
from seshat.core.models.enums import NodeStatus
from seshat.core.utils.hashing import sha256_text
from tests.helpers import make_node

_TRANSCRIPT = "We agreed to ship v2 in March."


def _make_service(
    job_row: dict | None = None,
    transcript: bytes | None = _TRANSCRIPT.encode(),
    documents_config: DocumentsConfig | None = None,
) -> tuple[DocumentService, MagicMock, MagicMock, MagicMock]:
    ops = MagicMock()
    ops.get_job = AsyncMock(return_value=job_row)
    ops.upsert_document = AsyncMock(side_effect=lambda doc: doc.model_dump())
    ops.get_documents_for_job = AsyncMock(return_value=[])
    ops.get_document = AsyncMock(return_value=None)

    blob = MagicMock()
    blob.get_raw_transcript = AsyncMock(return_value=transcript)

    node_repo = MagicMock()
    nodes = [make_node(quote=_TRANSCRIPT, transcript=_TRANSCRIPT)]
    node_repo.query = AsyncMock(return_value=nodes)
    node_repo.paginated_query = AsyncMock(return_value=nodes)

    return DocumentService(ops, blob, node_repo, documents_config), ops, blob, node_repo


def _done_job_row() -> dict:
    return {"job_id": "job-1", "status": "done", "meeting_date": date(2026, 4, 21)}


def _document_row(job_id: str = "job-1", **overrides) -> dict:
    markdown = "# Meeting Summary\n"
    row = {
        "id": uuid4(),
        "job_id": job_id,
        "kind": "meeting_summary",
        "filename": "meeting_summary.md",
        "markdown_content": markdown,
        "content_revision": sha256_text(markdown),
        "created_at": datetime.now(UTC),
        "validation_status": "pending",
        "validation_revision": 0,
        "edited_content": None,
        "rejection_reason": None,
        "validated_by": None,
        "validated_at": None,
        "auto_approved": False,
        "approved_revision": None,
    }
    row.update(overrides)
    return row


class TestGenerateForJob:
    async def test_unknown_job_raises_not_found(self):
        svc, _, _, _ = _make_service(job_row=None)
        with pytest.raises(JobNotFoundError):
            await svc.generate_for_job("missing")

    async def test_non_done_job_raises_state_error(self):
        svc, _, _, _ = _make_service(job_row={"job_id": "job-1", "status": "awaiting_review"})
        with pytest.raises(JobStateError):
            await svc.generate_for_job("job-1")

    async def test_missing_transcript_raises(self):
        svc, _, _, _ = _make_service(job_row=_done_job_row(), transcript=None)
        with pytest.raises(TranscriptNotFoundError):
            await svc.generate_for_job("job-1")

    async def test_generates_and_upserts_document(self):
        svc, ops, blob, node_repo = _make_service(job_row=_done_job_row())

        document = await svc.generate_for_job("job-1")

        node_filter = node_repo.paginated_query.call_args.args[0]
        assert node_filter.job_id == "job-1"
        assert node_filter.status == NodeStatus.APPROVED
        blob.get_raw_transcript.assert_awaited_once_with(date(2026, 4, 21), "job-1")
        ops.upsert_document.assert_awaited_once()
        assert document.kind == DocumentKind.MEETING_SUMMARY
        assert document.filename == "meeting_summary.md"
        assert "Use PostgreSQL" in document.markdown_content
        assert document.content_revision == sha256_text(document.markdown_content)

    async def test_fetches_all_approved_nodes_with_paginated_query(self):
        svc, _, _, node_repo = _make_service(job_row=_done_job_row())
        first = make_node("n1", title="First decision", quote=_TRANSCRIPT, transcript=_TRANSCRIPT)
        second = make_node("n2", title="Decision beyond first page", quote=_TRANSCRIPT, transcript=_TRANSCRIPT)
        node_repo.query = AsyncMock(return_value=[first])
        node_repo.paginated_query = AsyncMock(return_value=[first, second])

        document = await svc.generate_for_job("job-1")

        assert "First decision" in document.markdown_content
        assert "Decision beyond first page" in document.markdown_content
        node_repo.paginated_query.assert_awaited_once()

    async def test_generated_document_is_pending_by_default(self):
        svc, _, _, _ = _make_service(job_row=_done_job_row())

        document = await svc.generate_for_job("job-1")

        assert document.validation_status == DocumentValidationStatus.PENDING
        assert document.auto_approved is False
        assert document.approved_revision is None
        assert document.validated_at is None

    async def test_auto_approve_kind_marks_document_approved(self):
        config = DocumentsConfig(auto_approve_kinds=[DocumentKind.MEETING_SUMMARY])
        svc, _, _, _ = _make_service(job_row=_done_job_row(), documents_config=config)

        document = await svc.generate_for_job("job-1")

        assert document.validation_status == DocumentValidationStatus.APPROVED
        assert document.auto_approved is True
        assert document.validated_by is None
        assert document.validated_at is not None
        assert document.approved_revision == document.content_revision


class TestReview:
    async def test_unknown_document_raises_not_found(self):
        svc, _, _, _ = _make_service()
        request = DocumentReviewRequest(action="approve", expected_revision="abc", expected_validation_revision=0)

        with pytest.raises(DocumentNotFoundError):
            await svc.review(uuid4(), request, "reviewer-1")

    async def test_stale_expected_revision_raises_conflict_without_writing(self):
        row = _document_row()
        svc, ops, _, _ = _make_service()
        ops.get_document = AsyncMock(return_value=row)
        ops.review_document = AsyncMock()
        request = DocumentReviewRequest(
            action="approve",
            expected_revision="stale-revision",
            expected_validation_revision=row["validation_revision"],
        )

        with pytest.raises(DocumentRevisionConflictError):
            await svc.review(row["id"], request, "reviewer-1")

        ops.review_document.assert_not_awaited()

    async def test_approve_sets_status_and_approved_revision(self):
        row = _document_row()
        svc, ops, _, _ = _make_service()
        ops.get_document = AsyncMock(return_value=row)
        ops.review_document = AsyncMock(
            return_value=_document_row(
                id=row["id"],
                validation_status="approved",
                validation_revision=1,
                validated_by="reviewer-1",
                validated_at=datetime.now(UTC),
                approved_revision=row["content_revision"],
            )
        )
        request = DocumentReviewRequest(
            action="approve",
            expected_revision=row["content_revision"],
            expected_validation_revision=row["validation_revision"],
        )

        document = await svc.review(row["id"], request, "reviewer-1")

        args = ops.review_document.await_args.args
        assert args[0] == row["id"]
        assert args[1] == row["content_revision"]
        assert args[2] == row["validation_revision"]
        assert args[3] == DocumentValidationStatus.APPROVED
        assert args[4] is None
        assert args[5] is None
        assert args[6] == "reviewer-1"
        assert args[8] == sha256_text(row["markdown_content"])
        assert document.validation_status == DocumentValidationStatus.APPROVED
        assert document.validation_revision == 1

    async def test_approve_with_edits_sets_edited_status_and_hashes_edit(self):
        row = _document_row()
        edited = "# Edited Summary\n"
        svc, ops, _, _ = _make_service()
        ops.get_document = AsyncMock(return_value=row)
        ops.review_document = AsyncMock(
            return_value=_document_row(
                id=row["id"],
                validation_status="edited",
                validation_revision=1,
                edited_content=edited,
                validated_by="reviewer-1",
                validated_at=datetime.now(UTC),
                approved_revision=sha256_text(edited),
            )
        )
        request = DocumentReviewRequest(
            action="approve",
            expected_revision=row["content_revision"],
            expected_validation_revision=row["validation_revision"],
            edited_content=edited,
        )

        document = await svc.review(row["id"], request, "reviewer-1")

        args = ops.review_document.await_args.args
        assert args[3] == DocumentValidationStatus.EDITED
        assert args[4] == edited
        assert args[8] == sha256_text(edited)
        assert document.validation_status == DocumentValidationStatus.EDITED

    async def test_reject_stores_reason_and_clears_approval_fields(self):
        row = _document_row()
        svc, ops, _, _ = _make_service()
        ops.get_document = AsyncMock(return_value=row)
        ops.review_document = AsyncMock(
            return_value=_document_row(
                id=row["id"],
                validation_status="rejected",
                validation_revision=1,
                rejection_reason="wrong decisions",
                validated_by="reviewer-1",
                validated_at=datetime.now(UTC),
            )
        )
        request = DocumentReviewRequest(
            action="reject",
            expected_revision=row["content_revision"],
            expected_validation_revision=row["validation_revision"],
            reason="wrong decisions",
        )

        document = await svc.review(row["id"], request, "reviewer-1")

        args = ops.review_document.await_args.args
        assert args[3] == DocumentValidationStatus.REJECTED
        assert args[4] is None
        assert args[5] == "wrong decisions"
        assert args[8] is None
        assert document.validation_status == DocumentValidationStatus.REJECTED
        assert document.rejection_reason == "wrong decisions"

    async def test_concurrent_regeneration_between_load_and_write_raises_conflict(self):
        row = _document_row()
        svc, ops, _, _ = _make_service()
        ops.get_document = AsyncMock(return_value=row)
        # Conditional UPDATE matched 0 rows: a regeneration landed after the load.
        ops.review_document = AsyncMock(return_value=None)
        request = DocumentReviewRequest(
            action="approve",
            expected_revision=row["content_revision"],
            expected_validation_revision=row["validation_revision"],
        )

        with pytest.raises(DocumentRevisionConflictError):
            await svc.review(row["id"], request, "reviewer-1")


class TestListForJob:
    async def test_unknown_job_raises_not_found(self):
        svc, _, _, _ = _make_service(job_row=None)
        with pytest.raises(JobNotFoundError):
            await svc.list_for_job("missing")

    async def test_maps_rows_to_models(self):
        svc, ops, _, _ = _make_service(job_row=_done_job_row())
        ops.get_documents_for_job = AsyncMock(return_value=[_document_row()])

        documents = await svc.list_for_job("job-1")

        assert len(documents) == 1
        assert isinstance(documents[0], GeneratedDocument)
        assert documents[0].kind == DocumentKind.MEETING_SUMMARY


class TestGet:
    async def test_returns_none_when_missing(self):
        svc, _, _, _ = _make_service()
        assert await svc.get(uuid4()) is None

    async def test_returns_model_when_found(self):
        svc, ops, _, _ = _make_service()
        row = _document_row()
        ops.get_document = AsyncMock(return_value=row)

        document = await svc.get(row["id"])

        assert document is not None
        assert document.id == row["id"]
        assert document.markdown_content == row["markdown_content"]
