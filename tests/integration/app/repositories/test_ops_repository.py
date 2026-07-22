from __future__ import annotations

import asyncio
import json
from datetime import UTC, date, datetime
from typing import TYPE_CHECKING
from uuid import uuid4

import asyncpg
import pytest

from seshat.app.repositories.ops_repository import ApiKeyAlreadyRevokedError, ApiKeyNotFoundError, OpsRepository
from seshat.core.config.settings import OpsStoreConfig
from seshat.core.models.documents import DocumentKind, DocumentValidationStatus, GeneratedDocument
from seshat.core.models.enums import JobStatus, UserRole
from seshat.core.utils.hashing import sha256_text
from seshat.infra.ops_store.pg_store import PostgresOpsStore
from tests.integration.conftest import SKIP_IF_NO_POSTGRES

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

pytestmark = [pytest.mark.integration, SKIP_IF_NO_POSTGRES]


@pytest.fixture
async def repo(pg_test_url: str) -> AsyncGenerator[OpsRepository]:
    store = PostgresOpsStore(OpsStoreConfig(schema_name="ops"), pg_test_url)
    await store.connect()
    yield OpsRepository(store)
    await store.pool.execute("TRUNCATE ops.api_keys, ops.jobs, ops.generated_documents CASCADE")
    await store.close()


class TestCreateJob:
    async def test_row_contains_all_columns(self, repo: OpsRepository):
        meeting_date = date(2026, 6, 1)
        submission = '{"source_type": "audio", "metadata": {"meeting_date": "2026-06-01"}}'
        raw_key = "jobs/2026-06-01/job-1/raw/input.mp3"

        await repo.create_job("job-1", "user-1", "audio", None, datetime.now(UTC), meeting_date, submission, raw_key)

        row = await repo.get_job("job-1")
        assert row is not None
        assert str(row["meeting_date"]) == "2026-06-01"
        assert json.loads(row["submission"])["source_type"] == "audio"
        assert row["raw_blob_key"] == raw_key
        assert row["status"] == "pending"

    async def test_not_null_constraints_are_satisfied(self, repo: OpsRepository):
        pool = repo._store._pool
        with pytest.raises(asyncpg.NotNullViolationError):
            await pool.execute(
                "INSERT INTO ops.jobs (job_id, user_id, status, source_type, created_at, updated_at)"
                " VALUES ($1, $2, 'pending', $3, $4, $4)",
                "job-bad",
                "user-1",
                "audio",
                datetime.now(UTC),
            )


class TestUpdateJobStatus:
    async def test_status_transitions(self, repo: OpsRepository):
        meeting_date = date(2026, 6, 1)
        submission = '{"source_type": "text", "metadata": {"meeting_date": "2026-06-01"}}'
        raw_key = "jobs/2026-06-01/job-2/raw/input.txt"

        await repo.create_job("job-2", "user-1", "text", None, datetime.now(UTC), meeting_date, submission, raw_key)
        await repo.update_job_status("job-2", JobStatus.IDENTIFYING)

        row = await repo.get_job("job-2")
        assert row is not None
        assert row["status"] == "identifying"


class TestFailJob:
    async def test_fail_job_sets_status_and_error(self, repo: OpsRepository):
        meeting_date = date(2026, 6, 1)
        submission = '{"source_type": "text", "metadata": {"meeting_date": "2026-06-01"}}'
        raw_key = "jobs/2026-06-01/job-3/raw/input.txt"

        await repo.create_job("job-3", "user-1", "text", None, datetime.now(UTC), meeting_date, submission, raw_key)
        await repo.fail_job("job-3", "pipeline", "something broke", recoverable=True)

        row = await repo.get_job("job-3")
        assert row is not None
        assert row["status"] == "failed"
        payload = json.loads(row["error_payload"])
        assert payload["stage"] == "pipeline"
        assert payload["reason"] == "something broke"
        assert payload["recoverable"] is True


class TestContentHashDedup:
    async def test_first_done_job_found_by_content_hash(self, repo: OpsRepository):
        meeting_date = date(2026, 6, 1)
        submission = '{"source_type": "text", "metadata": {"meeting_date": "2026-06-01"}}'
        raw_key = "jobs/2026-06-01/job-hash-1/raw/input.txt"

        await repo.create_job(
            "job-hash-1",
            "user-1",
            "text",
            None,
            datetime.now(UTC),
            meeting_date,
            submission,
            raw_key,
            content_hash="hash-abc",
        )
        await repo.update_job_status("job-hash-1", JobStatus.DONE)

        result = await repo.find_job_by_content_hash("hash-abc")
        assert result == "job-hash-1"

    async def test_failed_job_not_returned_by_content_hash(self, repo: OpsRepository):
        meeting_date = date(2026, 6, 1)
        submission = '{"source_type": "text", "metadata": {"meeting_date": "2026-06-01"}}'
        raw_key = "jobs/2026-06-01/job-hash-2/raw/input.txt"

        await repo.create_job(
            "job-hash-2",
            "user-1",
            "text",
            None,
            datetime.now(UTC),
            meeting_date,
            submission,
            raw_key,
            content_hash="hash-def",
        )
        await repo.fail_job("job-hash-2", "pipeline", "error", recoverable=False)

        result = await repo.find_job_by_content_hash("hash-def")
        assert result is None


class TestRateLimit:
    async def test_count_recent_jobs_per_user(self, repo: OpsRepository):
        meeting_date = date(2026, 6, 1)
        submission = '{"source_type": "text", "metadata": {"meeting_date": "2026-06-01"}}'

        for i in range(10):
            raw_key = f"jobs/2026-06-01/job-rate-{i}/raw/input.txt"
            await repo.create_job(
                f"job-rate-{i}", "user-rate-1", "text", None, datetime.now(UTC), meeting_date, submission, raw_key
            )

        await repo.create_job(
            "job-rate-other",
            "user-rate-2",
            "text",
            None,
            datetime.now(UTC),
            meeting_date,
            submission,
            "jobs/2026-06-01/job-rate-other/raw/input.txt",
        )

        count_user1 = await repo.count_recent_jobs_for_user("user-rate-1")
        count_user2 = await repo.count_recent_jobs_for_user("user-rate-2")

        assert count_user1 == 10
        assert count_user2 == 1


class TestStrandedRecovery:
    async def test_writing_job_returned_as_stranded(self, repo: OpsRepository):
        meeting_date = date(2026, 6, 1)
        submission = '{"source_type": "text", "metadata": {"meeting_date": "2026-06-01"}}'
        raw_key = "jobs/2026-06-01/job-stranded/raw/input.txt"

        await repo.create_job(
            "job-stranded", "user-1", "text", None, datetime.now(UTC), meeting_date, submission, raw_key
        )
        await repo.update_job_status("job-stranded", JobStatus.WRITING)

        stranded = await repo.get_stranded_writing_jobs()
        assert "job-stranded" in stranded

    async def test_done_job_not_returned_as_stranded(self, repo: OpsRepository):
        meeting_date = date(2026, 6, 1)
        submission = '{"source_type": "text", "metadata": {"meeting_date": "2026-06-01"}}'
        raw_key = "jobs/2026-06-01/job-done/raw/input.txt"

        await repo.create_job("job-done", "user-1", "text", None, datetime.now(UTC), meeting_date, submission, raw_key)
        await repo.update_job_status("job-done", JobStatus.DONE)

        stranded = await repo.get_stranded_writing_jobs()
        assert "job-done" not in stranded


class TestApiKeysCRUD:
    async def test_create_and_list(self, repo: OpsRepository):
        now = datetime.now(UTC)
        await repo.create_api_key("hash-abc", "alice", UserRole.REVIEWER, now)

        rows = await repo.list_api_keys()

        assert len(rows) == 1
        assert rows[0]["user_id"] == "alice"
        assert rows[0]["role"] == "reviewer"
        assert rows[0]["revoked_at"] is None

    async def test_get_api_keys_returns_active_only(self, repo: OpsRepository):
        now = datetime.now(UTC)
        await repo.create_api_key("hash-active", "alice", UserRole.REVIEWER, now)
        await repo.create_api_key("hash-revoked", "bob", UserRole.VIEWER, now)

        rows = await repo.list_api_keys()
        revoke_id = next(r["id"] for r in rows if r["user_id"] == "bob")
        await repo.revoke_api_key(revoke_id, datetime.now(UTC))

        active = await repo.get_api_keys()
        user_ids = [t[1] for t in active]
        assert "alice" in user_ids
        assert "bob" not in user_ids

    async def test_revoke_ok_then_already_revoked(self, repo: OpsRepository):
        now = datetime.now(UTC)
        await repo.create_api_key("hash-rev", "charlie", UserRole.OPERATOR, now)
        rows = await repo.list_api_keys()
        key_id = rows[0]["id"]

        await repo.revoke_api_key(key_id, datetime.now(UTC))

        with pytest.raises(ApiKeyAlreadyRevokedError):
            await repo.revoke_api_key(key_id, datetime.now(UTC))

    async def test_revoke_not_found(self, repo: OpsRepository):
        with pytest.raises(ApiKeyNotFoundError):
            await repo.revoke_api_key(99999, datetime.now(UTC))


async def _create_test_job(
    repo: OpsRepository,
    job_id: str,
    user_id: str = "user-1",
    source_type: str = "text",
    idempotency_key: str | None = None,
    content_hash: str | None = None,
) -> None:
    await repo.create_job(
        job_id,
        user_id,
        source_type,
        idempotency_key,
        datetime.now(UTC),
        date(2026, 6, 1),
        '{"source_type": "text", "metadata": {"meeting_date": "2026-06-01"}}',
        f"jobs/2026-06-01/{job_id}/raw/input.txt",
        content_hash,
    )


class TestConcurrentRevoke:
    async def test_concurrent_revoke_exactly_one_ok(self, repo: OpsRepository):
        now = datetime.now(UTC)
        await repo.create_api_key("hash-concurrent", "dave", UserRole.REVIEWER, now)
        rows = await repo.list_api_keys()
        key_id = rows[0]["id"]

        results = await asyncio.gather(
            repo.revoke_api_key(key_id, datetime.now(UTC)),
            repo.revoke_api_key(key_id, datetime.now(UTC)),
            return_exceptions=True,
        )

        errors = [r for r in results if isinstance(r, Exception)]
        successes = [r for r in results if r is None]
        assert len(successes) == 1
        assert len(errors) == 1
        assert isinstance(errors[0], ApiKeyAlreadyRevokedError)


class TestResetFailedJob:
    async def test_reset_restores_pending_status_and_clears_error(self, repo: OpsRepository):
        await _create_test_job(repo, "job-reset-1")
        await repo.fail_job("job-reset-1", "pipeline", "oops", recoverable=True)

        await repo.reset_failed_job("job-reset-1")

        row = await repo.get_job("job-reset-1")
        assert row is not None
        assert row["status"] == "pending"
        assert row["error_payload"] is None
        assert row["finished_at"] is None


class TestFindJobByIdempotencyKey:
    async def test_finds_existing_job_by_key(self, repo: OpsRepository):
        await _create_test_job(repo, "job-idem-1", idempotency_key="key-abc")

        result = await repo.find_job_by_idempotency_key("key-abc")

        assert result is not None
        assert result["job_id"] == "job-idem-1"

    async def test_returns_none_for_unknown_key(self, repo: OpsRepository):
        result = await repo.find_job_by_idempotency_key("nonexistent-key")
        assert result is None


class TestListJobs:
    async def test_filter_by_status(self, repo: OpsRepository):
        await _create_test_job(repo, "job-list-1")
        await _create_test_job(repo, "job-list-2")
        await repo.update_job_status("job-list-1", JobStatus.DONE)

        done = await repo.list_jobs(status=JobStatus.DONE)
        pending = await repo.list_jobs(status=JobStatus.PENDING)

        done_ids = [r["job_id"] for r in done]
        pending_ids = [r["job_id"] for r in pending]
        assert "job-list-1" in done_ids
        assert "job-list-1" not in pending_ids
        assert "job-list-2" in pending_ids

    async def test_filter_by_source_type(self, repo: OpsRepository):
        await _create_test_job(repo, "job-list-audio", source_type="audio")
        await _create_test_job(repo, "job-list-text", source_type="text")

        audio_jobs = await repo.list_jobs(source_type="audio")
        text_jobs = await repo.list_jobs(source_type="text")

        assert any(r["job_id"] == "job-list-audio" for r in audio_jobs)
        assert not any(r["job_id"] == "job-list-audio" for r in text_jobs)


class TestFinishedAt:
    async def test_terminal_status_populates_finished_at(self, repo: OpsRepository):
        await _create_test_job(repo, "job-fin-1")
        await repo.update_job_status("job-fin-1", JobStatus.DONE)

        row = await repo.get_job("job-fin-1")
        assert row is not None
        assert row["finished_at"] is not None

    async def test_non_terminal_status_does_not_set_finished_at(self, repo: OpsRepository):
        await _create_test_job(repo, "job-fin-2")
        await repo.update_job_status("job-fin-2", JobStatus.IDENTIFYING)

        row = await repo.get_job("job-fin-2")
        assert row is not None
        assert row["finished_at"] is None


def _make_document(job_id: str = "job-doc-1", markdown: str = "# Meeting Summary\n") -> GeneratedDocument:
    return GeneratedDocument(
        job_id=job_id,
        kind=DocumentKind.MEETING_SUMMARY,
        filename="meeting_summary.md",
        markdown_content=markdown,
        content_revision=sha256_text(markdown),
        created_at=datetime.now(UTC),
    )


class TestGeneratedDocuments:
    async def test_upsert_and_get_document(self, repo: OpsRepository):
        document = _make_document()
        stored = await repo.upsert_document(document)

        assert stored["id"] == document.id
        row = await repo.get_document(document.id)
        assert row is not None
        assert row["job_id"] == "job-doc-1"
        assert row["kind"] == "meeting_summary"
        assert row["markdown_content"] == document.markdown_content
        assert row["content_revision"] == document.content_revision

    async def test_upsert_overwrites_same_job_and_kind(self, repo: OpsRepository):
        first = _make_document(markdown="# v1\n")
        await repo.upsert_document(first)

        second = _make_document(markdown="# v2\n")
        stored = await repo.upsert_document(second)

        # The original row (and its id) survives; content and revision are replaced.
        assert stored["id"] == first.id
        assert stored["markdown_content"] == "# v2\n"
        assert stored["content_revision"] == sha256_text("# v2\n")
        rows = await repo.get_documents_for_job("job-doc-1")
        assert len(rows) == 1

    async def test_get_documents_for_job_filters_by_job(self, repo: OpsRepository):
        await repo.upsert_document(_make_document(job_id="job-doc-a"))
        await repo.upsert_document(_make_document(job_id="job-doc-b"))

        rows = await repo.get_documents_for_job("job-doc-a")

        assert [r["job_id"] for r in rows] == ["job-doc-a"]

    async def test_get_document_returns_none_for_unknown_id(self, repo: OpsRepository):
        assert await repo.get_document(uuid4()) is None

    async def test_upsert_row_has_pending_validation_defaults(self, repo: OpsRepository):
        document = _make_document(job_id="job-doc-val")
        stored = await repo.upsert_document(document)

        assert stored["validation_status"] == "pending"
        assert stored["validation_revision"] == 0
        assert stored["auto_approved"] is False
        assert stored["approved_revision"] is None


class TestReviewDocument:
    async def _approve(self, repo: OpsRepository, document: GeneratedDocument) -> dict | None:
        return await repo.review_document(
            document.id,
            document.content_revision,
            document.validation_revision,
            DocumentValidationStatus.APPROVED,
            None,
            None,
            "rachel",
            datetime.now(UTC),
            document.content_revision,
        )

    async def test_matching_revision_applies_decision(self, repo: OpsRepository):
        document = _make_document(job_id="job-rev-1")
        await repo.upsert_document(document)

        updated = await self._approve(repo, document)

        assert updated is not None
        assert updated["validation_status"] == "approved"
        assert updated["validation_revision"] == 1
        assert updated["validated_by"] == "rachel"
        assert updated["approved_revision"] == document.content_revision

    async def test_mismatched_revision_updates_nothing(self, repo: OpsRepository):
        document = _make_document(job_id="job-rev-2")
        await repo.upsert_document(document)

        updated = await repo.review_document(
            document.id,
            "stale-revision",
            document.validation_revision,
            DocumentValidationStatus.APPROVED,
            None,
            None,
            "rachel",
            datetime.now(UTC),
            "stale-revision",
        )

        assert updated is None
        row = await repo.get_document(document.id)
        assert row["validation_status"] == "pending"
        assert row["validated_by"] is None

    async def test_upsert_resets_prior_decision(self, repo: OpsRepository):
        document = _make_document(job_id="job-rev-3")
        await repo.upsert_document(document)
        await self._approve(repo, document)

        regenerated = _make_document(job_id="job-rev-3", markdown="# v2\n")
        stored = await repo.upsert_document(regenerated)

        assert stored["validation_status"] == "pending"
        assert stored["validation_revision"] == 2
        assert stored["validated_by"] is None
        assert stored["validated_at"] is None
        assert stored["approved_revision"] is None


class TestCountRunningJobs:
    async def test_counts_running_statuses(self, repo: OpsRepository):
        await _create_test_job(repo, "job-run-1")
        await _create_test_job(repo, "job-run-2")
        await repo.update_job_status("job-run-1", JobStatus.IDENTIFYING)
        await repo.update_job_status("job-run-2", JobStatus.DONE)

        count = await repo.count_running_jobs()

        assert count >= 1
        row_done = await repo.get_job("job-run-2")
        assert row_done["status"] == "done"
