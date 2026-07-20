from __future__ import annotations

import json
from datetime import UTC, date, datetime
from typing import TYPE_CHECKING

import asyncpg
import pytest

from seshat.app.repositories.ops_repository import OpsRepository
from seshat.core.config.settings import OpsStoreConfig
from seshat.infra.ops_store.pg_store import PostgresOpsStore
from tests.integration.conftest import SKIP_IF_NO_POSTGRES

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

pytestmark = [pytest.mark.integration, SKIP_IF_NO_POSTGRES]


@pytest.fixture
async def repo(pg_test_url: str) -> AsyncGenerator[OpsRepository]:
    pool = await asyncpg.create_pool(pg_test_url)
    store = PostgresOpsStore(OpsStoreConfig(schema_name="ops"), pg_test_url)
    store._pool = pool
    yield OpsRepository(store)
    await pool.execute("TRUNCATE ops.jobs CASCADE")
    await pool.close()


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

        from seshat.core.models.enums import JobStatus

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
        assert row["error_payload"] is not None
