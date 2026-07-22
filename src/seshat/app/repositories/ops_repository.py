from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import date, datetime
    from uuid import UUID

    from seshat.core.models.documents import DocumentValidationStatus, GeneratedDocument
    from seshat.core.models.enums import JobStatus, UserRole
    from seshat.core.models.publishing import PublishResult
    from seshat.infra.ops_store.pg_store import PostgresOpsStore


class ApiKeyNotFoundError(Exception):
    pass


class ApiKeyAlreadyRevokedError(Exception):
    pass


class OpsRepository:
    def __init__(self, store: PostgresOpsStore) -> None:
        self._store = store

    # -- Jobs: Create ----------------------------------------------------------

    async def create_job(
        self,
        job_id: str,
        user_id: str,
        source_type: str,
        idempotency_key: str | None,
        now: datetime,
        meeting_date: date,
        submission_json: str,
        raw_blob_key: str,
        content_hash: str | None = None,
    ) -> None:
        await self._store.create_job(
            job_id,
            user_id,
            source_type,
            idempotency_key,
            now,
            meeting_date,
            submission_json,
            raw_blob_key,
            content_hash,
        )

    # -- Jobs: Read ------------------------------------------------------------

    async def get_job(self, job_id: str) -> dict | None:
        return await self._store.get_job(job_id)

    async def find_job_by_idempotency_key(self, key: str) -> dict | None:
        return await self._store.find_job_by_idempotency_key(key)

    async def find_job_by_content_hash(self, content_hash: str) -> str | None:
        return await self._store.find_job_by_content_hash(content_hash)

    async def list_jobs(
        self,
        status: JobStatus | None = None,
        source_type: str | None = None,
        meeting_date_from: date | None = None,
        meeting_date_to: date | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict]:
        return await self._store.list_jobs(status, source_type, meeting_date_from, meeting_date_to, limit, offset)

    async def count_recent_jobs_for_user(self, user_id: str) -> int:
        return await self._store.count_recent_jobs_for_user(user_id)

    async def count_running_jobs(self) -> int:
        return await self._store.count_running_jobs()

    async def get_stranded_writing_jobs(self) -> list[str]:
        return await self._store.get_stranded_writing_jobs()

    # -- Jobs: Update ----------------------------------------------------------

    async def update_job_status(self, job_id: str, status: JobStatus) -> None:
        await self._store.update_job_status(job_id, status)

    async def fail_job(self, job_id: str, stage: str, reason: str, *, recoverable: bool) -> None:
        await self._store.fail_job(job_id, stage, reason, recoverable=recoverable)

    async def set_job_mlflow_run_id(self, job_id: str, run_id: str) -> None:
        await self._store.set_job_mlflow_run_id(job_id, run_id)

    async def reset_failed_job(self, job_id: str) -> None:
        await self._store.reset_failed_job(job_id)

    async def set_job_submission(
        self,
        job_id: str,
        meeting_date: date,
        submission_json: str,
        raw_blob_key: str,
    ) -> None:
        await self._store.set_job_submission(job_id, meeting_date, submission_json, raw_blob_key)

    # -- Generated documents ---------------------------------------------------

    async def upsert_document(self, document: GeneratedDocument) -> dict:
        return await self._store.upsert_document(
            document.id,
            document.job_id,
            document.kind.value,
            document.filename,
            document.markdown_content,
            document.content_revision,
            document.created_at,
            document.validation_status.value,
            document.edited_content,
            document.rejection_reason,
            document.validated_by,
            document.validated_at,
            document.auto_approved,
            document.approved_revision,
        )

    async def review_document(
        self,
        document_id: UUID,
        expected_revision: str,
        expected_validation_revision: int,
        validation_status: DocumentValidationStatus,
        edited_content: str | None,
        rejection_reason: str | None,
        validated_by: str,
        validated_at: datetime,
        approved_revision: str | None,
    ) -> dict | None:
        return await self._store.review_document(
            document_id,
            expected_revision,
            expected_validation_revision,
            validation_status.value,
            edited_content,
            rejection_reason,
            validated_by,
            validated_at,
            approved_revision,
        )

    async def get_documents_for_job(self, job_id: str) -> list[dict]:
        return await self._store.get_documents_for_job(job_id)

    async def get_document(self, document_id: UUID) -> dict | None:
        return await self._store.get_document(document_id)

    # -- Publish results -------------------------------------------------------

    async def insert_publish_result(self, result: PublishResult) -> None:
        await self._store.insert_publish_result(
            result.job_id,
            result.branch,
            result.commit_sha,
            result.pr_url,
            result.compare_url,
            result.files,
            result.published_at,
        )

    async def get_latest_publish_result(self, job_id: str) -> dict | None:
        return await self._store.get_latest_publish_result(job_id)

    # -- API Keys: Create ------------------------------------------------------

    async def create_api_key(self, key_hash: str, user_id: str, role: UserRole, now: datetime) -> None:
        await self._store.create_api_key(key_hash, user_id, role, now)

    # -- API Keys: Read --------------------------------------------------------

    async def get_api_keys(self) -> list[tuple[str, str, str]]:
        rows = await self._store.get_api_keys()
        return [(row["key_hash"], row["user_id"], row["role"]) for row in rows]

    async def list_api_keys(self) -> list[dict]:
        return await self._store.list_api_keys()

    # -- API Keys: Update ------------------------------------------------------

    async def revoke_api_key(self, key_id: int, now: datetime) -> None:
        result = await self._store.revoke_api_key(key_id, now)
        match result:
            case "not_found":
                raise ApiKeyNotFoundError(key_id)
            case "already_revoked":
                raise ApiKeyAlreadyRevokedError(key_id)

    # -- Lifecycle -------------------------------------------------------------

    async def is_alive(self) -> bool:
        return await self._store.is_alive()
