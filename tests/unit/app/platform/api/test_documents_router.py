from __future__ import annotations

from datetime import UTC, datetime
from typing import ClassVar
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from seshat.app.services.document import DocumentNotFoundError, DocumentRevisionConflictError
from seshat.app.services.job import JobNotFoundError, JobStateError, TranscriptNotFoundError
from seshat.core.models.documents import DocumentKind, DocumentValidationStatus, GeneratedDocument
from seshat.core.models.enums import UserRole
from seshat.core.utils.hashing import sha256_text
from tests.unit.app.platform.api.conftest import make_app_state, make_current_user


def _make_document(job_id: str = "job-1", **overrides) -> GeneratedDocument:
    markdown = "# Meeting Summary\n"
    fields = {
        "job_id": job_id,
        "kind": DocumentKind.MEETING_SUMMARY,
        "filename": "meeting_summary.md",
        "markdown_content": markdown,
        "content_revision": sha256_text(markdown),
        "created_at": datetime.now(UTC),
    }
    fields.update(overrides)
    return GeneratedDocument(**fields)


def _make_app_state(**overrides):
    document_service = MagicMock()
    document_service.generate_for_job = AsyncMock(return_value=_make_document())
    document_service.list_for_job = AsyncMock(return_value=[_make_document()])
    document_service.get = AsyncMock(return_value=_make_document())
    document_service.review = AsyncMock(
        return_value=_make_document(
            validation_status=DocumentValidationStatus.APPROVED,
            validation_revision=1,
            validated_by="rachel",
            validated_at=datetime.now(UTC),
            approved_revision=sha256_text("# Meeting Summary\n"),
        )
    )
    return make_app_state(document_service=document_service, **overrides)


class TestGenerateDocument:
    async def test_requires_auth(self, api_client):
        async with api_client(_make_app_state()) as ac:
            resp = await ac.post("/jobs/job-1/documents")
        assert resp.status_code == 401

    async def test_viewer_forbidden(self, api_client):
        async with api_client(_make_app_state(), make_current_user(role=UserRole.VIEWER)) as ac:
            resp = await ac.post("/jobs/job-1/documents")
        assert resp.status_code == 403

    async def test_operator_generates_document(self, api_client):
        state = _make_app_state()
        async with api_client(state, make_current_user()) as ac:
            resp = await ac.post("/jobs/job-1/documents")

        assert resp.status_code == 200
        body = resp.json()
        assert body["filename"] == "meeting_summary.md"
        assert body["markdown_content"] == "# Meeting Summary\n"
        state.document_service.generate_for_job.assert_awaited_once_with("job-1")

    async def test_unknown_job_returns_404(self, api_client):
        state = _make_app_state()
        state.document_service.generate_for_job = AsyncMock(side_effect=JobNotFoundError("job-1"))
        async with api_client(state, make_current_user()) as ac:
            resp = await ac.post("/jobs/job-1/documents")
        assert resp.status_code == 404

    async def test_missing_transcript_returns_404(self, api_client):
        state = _make_app_state()
        state.document_service.generate_for_job = AsyncMock(side_effect=TranscriptNotFoundError("job-1"))
        async with api_client(state, make_current_user()) as ac:
            resp = await ac.post("/jobs/job-1/documents")
        assert resp.status_code == 404

    async def test_non_done_job_returns_409(self, api_client):
        state = _make_app_state()
        state.document_service.generate_for_job = AsyncMock(side_effect=JobStateError("not done"))
        async with api_client(state, make_current_user()) as ac:
            resp = await ac.post("/jobs/job-1/documents")
        assert resp.status_code == 409


class TestListDocuments:
    async def test_requires_auth(self, api_client):
        async with api_client(_make_app_state()) as ac:
            resp = await ac.get("/jobs/job-1/documents")
        assert resp.status_code == 401

    async def test_viewer_lists_metadata_without_content(self, api_client):
        async with api_client(_make_app_state(), make_current_user(role=UserRole.VIEWER)) as ac:
            resp = await ac.get("/jobs/job-1/documents")

        assert resp.status_code == 200
        body = resp.json()
        assert len(body) == 1
        assert body[0]["filename"] == "meeting_summary.md"
        assert "content_revision" in body[0]
        assert body[0]["validation_revision"] == 0
        assert "markdown_content" not in body[0]

    async def test_unknown_job_returns_404(self, api_client):
        state = _make_app_state()
        state.document_service.list_for_job = AsyncMock(side_effect=JobNotFoundError("job-1"))
        async with api_client(state, make_current_user(role=UserRole.VIEWER)) as ac:
            resp = await ac.get("/jobs/job-1/documents")
        assert resp.status_code == 404


class TestReviewDocument:
    _BODY: ClassVar[dict] = {
        "action": "approve",
        "expected_revision": sha256_text("# Meeting Summary\n"),
        "expected_validation_revision": 0,
    }

    async def test_requires_auth(self, api_client):
        async with api_client(_make_app_state()) as ac:
            resp = await ac.post(f"/documents/{uuid4()}/review", json=self._BODY)
        assert resp.status_code == 401

    async def test_viewer_forbidden(self, api_client):
        async with api_client(_make_app_state(), make_current_user(role=UserRole.VIEWER)) as ac:
            resp = await ac.post(f"/documents/{uuid4()}/review", json=self._BODY)
        assert resp.status_code == 403

    async def test_reviewer_applies_decision(self, api_client):
        state = _make_app_state()
        document_id = uuid4()
        async with api_client(state, make_current_user(user_id="rachel", role=UserRole.REVIEWER)) as ac:
            resp = await ac.post(f"/documents/{document_id}/review", json=self._BODY)

        assert resp.status_code == 200
        body = resp.json()
        assert body["validation_status"] == "approved"
        assert body["validated_by"] == "rachel"
        call = state.document_service.review.await_args
        assert call.args[0] == document_id
        assert call.args[1].action == "approve"
        assert call.args[2] == "rachel"

    async def test_unknown_document_returns_404(self, api_client):
        state = _make_app_state()
        state.document_service.review = AsyncMock(side_effect=DocumentNotFoundError("missing"))
        async with api_client(state, make_current_user(role=UserRole.REVIEWER)) as ac:
            resp = await ac.post(f"/documents/{uuid4()}/review", json=self._BODY)
        assert resp.status_code == 404

    async def test_revision_conflict_returns_409(self, api_client):
        state = _make_app_state()
        state.document_service.review = AsyncMock(side_effect=DocumentRevisionConflictError("stale"))
        async with api_client(state, make_current_user(role=UserRole.REVIEWER)) as ac:
            resp = await ac.post(f"/documents/{uuid4()}/review", json=self._BODY)
        assert resp.status_code == 409

    async def test_edited_content_on_reject_returns_422(self, api_client):
        body = {
            "action": "reject",
            "expected_revision": "abc",
            "expected_validation_revision": 0,
            "edited_content": "# Edited\n",
        }
        async with api_client(_make_app_state(), make_current_user(role=UserRole.REVIEWER)) as ac:
            resp = await ac.post(f"/documents/{uuid4()}/review", json=body)
        assert resp.status_code == 422


class TestGetDocument:
    async def test_requires_auth(self, api_client):
        async with api_client(_make_app_state()) as ac:
            resp = await ac.get(f"/documents/{uuid4()}")
        assert resp.status_code == 401

    async def test_viewer_gets_full_document(self, api_client):
        document = _make_document()
        state = _make_app_state()
        state.document_service.get = AsyncMock(return_value=document)

        async with api_client(state, make_current_user(role=UserRole.VIEWER)) as ac:
            resp = await ac.get(f"/documents/{document.id}")

        assert resp.status_code == 200
        body = resp.json()
        assert body["id"] == str(document.id)
        assert body["markdown_content"] == document.markdown_content
        assert body["content_revision"] == document.content_revision

    async def test_unknown_document_returns_404(self, api_client):
        state = _make_app_state()
        state.document_service.get = AsyncMock(return_value=None)
        async with api_client(state, make_current_user(role=UserRole.VIEWER)) as ac:
            resp = await ac.get(f"/documents/{uuid4()}")
        assert resp.status_code == 404
