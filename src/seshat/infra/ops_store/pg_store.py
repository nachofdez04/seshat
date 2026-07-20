from __future__ import annotations

import json
import logging
from datetime import UTC, date, datetime
from typing import TYPE_CHECKING, Literal

import asyncpg

from seshat.core.models.enums import JobStatus, UserRole

if TYPE_CHECKING:
    from seshat.core.config.settings import OpsStoreConfig

logger = logging.getLogger(__name__)


class PostgresOpsStore:
    def __init__(self, config: OpsStoreConfig, connection_string: str) -> None:
        self._connection_string = connection_string
        self._schema = config.schema_name
        self._pool_min = config.pool_min_size
        self._pool_max = config.pool_max_size
        self._pool: asyncpg.Pool | None = None

    async def connect(self) -> None:
        self._pool = await asyncpg.create_pool(
            self._connection_string,
            min_size=self._pool_min,
            max_size=self._pool_max,
        )
        logger.info("PostgresOpsStore pool created (min=%d max=%d)", self._pool_min, self._pool_max)

    @property
    def pool(self) -> asyncpg.Pool:
        if self._pool is None:
            raise RuntimeError("PostgresOpsStore.connect() has not been called")
        return self._pool

    async def is_alive(self) -> bool:
        try:
            await self.pool.fetchval("SELECT 1")
            return True
        except Exception:
            return False

    async def close(self) -> None:
        if self._pool:
            await self._pool.close()
            logger.debug("PostgresOpsStore pool closed")

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
        await self.pool.execute(
            f"INSERT INTO {self._schema}.jobs "
            "(job_id, user_id, status, idempotency_key, source_type, created_at, updated_at, meeting_date, submission, raw_blob_key, content_hash) "  # noqa: E501
            "VALUES ($1, $2, $3, $4, $5, $6, $6, $7, $8, $9, $10)",
            job_id,
            user_id,
            JobStatus.PENDING,
            idempotency_key,
            source_type,
            now,
            meeting_date,
            submission_json,
            raw_blob_key,
            content_hash,
        )

    # -- Jobs: Read ------------------------------------------------------------

    async def get_job(self, job_id: str) -> dict | None:
        row = await self.pool.fetchrow(f"SELECT * FROM {self._schema}.jobs WHERE job_id=$1", job_id)
        return self._to_dict(row)

    async def find_job_by_idempotency_key(self, key: str) -> dict | None:
        row = await self.pool.fetchrow(f"SELECT job_id, status FROM {self._schema}.jobs WHERE idempotency_key=$1", key)
        return self._to_dict(row)

    async def find_job_by_content_hash(self, content_hash: str) -> str | None:
        return await self.pool.fetchval(
            f"SELECT job_id FROM {self._schema}.jobs "
            "WHERE content_hash=$1 AND status='done' ORDER BY created_at DESC LIMIT 1",
            content_hash,
        )

    async def list_jobs(
        self,
        status: JobStatus | None = None,
        source_type: str | None = None,
        meeting_date_from: date | None = None,
        meeting_date_to: date | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict]:
        conditions: list[str] = []
        params: list = []

        if status is not None:
            params.append(status.value)
            conditions.append(f"status=${len(params)}")

        if source_type is not None:
            params.append(source_type)
            conditions.append(f"source_type=${len(params)}")

        if meeting_date_from is not None:
            params.append(meeting_date_from)
            conditions.append(f"meeting_date >= ${len(params)}")

        if meeting_date_to is not None:
            params.append(meeting_date_to)
            conditions.append(f"meeting_date <= ${len(params)}")

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        params.extend([limit, offset])
        rows = await self.pool.fetch(
            f"SELECT * FROM {self._schema}.jobs {where} ORDER BY created_at DESC LIMIT ${len(params) - 1} OFFSET ${len(params)}",  # noqa: E501
            *params,
        )
        return self._to_dicts(rows)

    async def count_recent_jobs_for_user(self, user_id: str) -> int:
        return await self.pool.fetchval(
            f"SELECT COUNT(*) FROM {self._schema}.jobs WHERE user_id=$1 AND created_at > NOW() - INTERVAL '1 hour'",
            user_id,
        )

    async def count_running_jobs(self) -> int:
        return await self.pool.fetchval(
            f"SELECT COUNT(*) FROM {self._schema}.jobs WHERE status = ANY($1::text[])",
            list(JobStatus.running_statuses()),
        )

    async def get_stranded_writing_jobs(self) -> list[str]:
        rows = await self.pool.fetch(
            f"SELECT job_id FROM {self._schema}.jobs WHERE status = ANY($1::text[])",
            list(JobStatus.stranded_statuses()),
        )
        return [row["job_id"] for row in rows]

    # -- Jobs: Update ----------------------------------------------------------

    async def update_job_status(self, job_id: str, status: JobStatus) -> None:
        now = datetime.now(UTC)
        await self.pool.execute(
            f"UPDATE {self._schema}.jobs SET status=$1, updated_at=$2, finished_at=$3 WHERE job_id=$4",
            status.value,
            now,
            now if status.is_terminal else None,
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
        now = datetime.now(UTC)
        await self.pool.execute(
            f"UPDATE {self._schema}.jobs"
            " SET status='failed', error_payload=$1, updated_at=$2, finished_at=$2 WHERE job_id=$3",
            payload,
            now,
            job_id,
        )

    async def set_job_mlflow_run_id(self, job_id: str, run_id: str) -> None:
        await self.pool.execute(
            f"UPDATE {self._schema}.jobs SET mlflow_run_id=$1 WHERE job_id=$2",
            run_id,
            job_id,
        )

    async def reset_failed_job(self, job_id: str) -> None:
        await self.pool.execute(
            f"UPDATE {self._schema}.jobs "
            "SET status='pending', error_payload=NULL, finished_at=NULL, updated_at=$1 WHERE job_id=$2",
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
        await self.pool.execute(
            f"UPDATE {self._schema}.jobs "
            "SET meeting_date=$1, submission=$2, raw_blob_key=$3, updated_at=$4 WHERE job_id=$5",
            meeting_date,
            submission_json,
            raw_blob_key,
            datetime.now(UTC),
            job_id,
        )

    # -- API Keys: Create ------------------------------------------------------

    async def create_api_key(self, key_hash: str, user_id: str, role: UserRole, now: datetime) -> None:
        await self.pool.execute(
            f"INSERT INTO {self._schema}.api_keys (key_hash, user_id, role, created_at) VALUES ($1, $2, $3, $4)",
            key_hash,
            user_id,
            role.value,
            now,
        )

    # -- API Keys: Read --------------------------------------------------------

    async def get_api_keys(self) -> list[dict]:
        rows = await self.pool.fetch(
            f"SELECT key_hash, user_id, role FROM {self._schema}.api_keys WHERE revoked_at IS NULL"
        )
        return self._to_dicts(rows)

    async def list_api_keys(self) -> list[dict]:
        rows = await self.pool.fetch(
            f"SELECT id, user_id, role, created_at, revoked_at FROM {self._schema}.api_keys ORDER BY created_at DESC"
        )
        return self._to_dicts(rows)

    # -- API Keys: Update ------------------------------------------------------

    async def revoke_api_key(self, key_id: int, now: datetime) -> Literal["ok", "not_found", "already_revoked"]:
        async with self.pool.acquire() as conn, conn.transaction():
            row = await conn.fetchrow(f"SELECT revoked_at FROM {self._schema}.api_keys WHERE id=$1", key_id)
            if row is None:
                return "not_found"

            if row["revoked_at"] is not None:
                return "already_revoked"

            await conn.execute(f"UPDATE {self._schema}.api_keys SET revoked_at=$1 WHERE id=$2", now, key_id)
            return "ok"

    @staticmethod
    def _to_dict(record: asyncpg.Record | None) -> dict | None:
        return dict(record) if record is not None else None

    @staticmethod
    def _to_dicts(records: list[asyncpg.Record]) -> list[dict]:
        return [dict(r) for r in records]
