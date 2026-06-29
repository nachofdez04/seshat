from __future__ import annotations

import json
import logging
from datetime import UTC, date, datetime
from typing import TYPE_CHECKING

from seshat.models.enums import JobStatus

if TYPE_CHECKING:
    import asyncpg


logger = logging.getLogger(__name__)

_RUNNING_STATUSES = (JobStatus.TRANSCRIBING, JobStatus.EXTRACTING, JobStatus.WRITING)


class OpsLedger:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

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
    ) -> None:
        await self._pool.execute(
            "INSERT INTO ops.jobs "
            "(job_id, user_id, status, idempotency_key, source_type, created_at, updated_at, meeting_date, submission, raw_blob_key) "  # noqa: E501
            "VALUES ($1, $2, $3, $4, $5, $6, $6, $7, $8, $9)",
            job_id,
            user_id,
            JobStatus.PENDING,
            idempotency_key,
            source_type,
            now,
            meeting_date,
            submission_json,
            raw_blob_key,
        )

    async def get_job(self, job_id: str) -> asyncpg.Record | None:
        return await self._pool.fetchrow("SELECT * FROM ops.jobs WHERE job_id=$1", job_id)

    async def find_job_by_idempotency_key(self, key: str) -> asyncpg.Record | None:
        return await self._pool.fetchrow("SELECT job_id, status FROM ops.jobs WHERE idempotency_key=$1", key)

    async def update_job_status(self, job_id: str, status: JobStatus) -> None:
        await self._pool.execute(
            "UPDATE ops.jobs SET status=$1, updated_at=$2 WHERE job_id=$3",
            status.value,
            datetime.now(UTC),
            job_id,
        )

    async def fail_job(
        self,
        job_id: str,
        stage: str,
        reason: str,
        *,
        recoverable: bool,
    ) -> None:
        payload = json.dumps(
            {"stage": stage, "status": "failed", "reason": reason, "recoverable": recoverable, "usage": {}}
        )
        await self._pool.execute(
            "UPDATE ops.jobs SET status='failed', error_payload=$1, updated_at=$2 WHERE job_id=$3",
            payload,
            datetime.now(UTC),
            job_id,
        )

    async def count_recent_jobs_for_user(self, user_id: str) -> int:
        return await self._pool.fetchval(
            "SELECT COUNT(*) FROM ops.jobs WHERE user_id=$1 AND created_at > NOW() - INTERVAL '1 hour'",
            user_id,
        )

    async def count_running_jobs(self) -> int:
        return await self._pool.fetchval(
            "SELECT COUNT(*) FROM ops.jobs WHERE status = ANY($1::text[])",
            list(_RUNNING_STATUSES),
        )

    async def get_api_keys(self) -> list[tuple[str, str, str]]:
        rows = await self._pool.fetch("SELECT key_hash, user_id, role FROM ops.api_keys")
        return [(row["key_hash"], row["user_id"], row["role"]) for row in rows]

    async def reset_failed_job(self, job_id: str) -> None:
        await self._pool.execute(
            "UPDATE ops.jobs SET status='pending', error_payload=NULL, updated_at=$1 WHERE job_id=$2",
            datetime.now(UTC),
            job_id,
        )

    async def set_job_submission(
        self,
        job_id: str,
        meeting_date: date,
        submission_json: str,
        raw_blob_key: str,
    ) -> None:
        await self._pool.execute(
            "UPDATE ops.jobs SET meeting_date=$1, submission=$2, raw_blob_key=$3, updated_at=$4 WHERE job_id=$5",
            meeting_date,
            submission_json,
            raw_blob_key,
            datetime.now(UTC),
            job_id,
        )

    async def get_stranded_writing_jobs(self) -> list[str]:
        rows = await self._pool.fetch("SELECT job_id FROM ops.jobs WHERE status='writing'")
        return [row["job_id"] for row in rows]

    async def close(self) -> None:
        await self._pool.close()
