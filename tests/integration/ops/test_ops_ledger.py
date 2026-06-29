from __future__ import annotations

import json
from datetime import UTC, date, datetime
from typing import TYPE_CHECKING

import asyncpg
import pytest

from seshat.ops.ledger import OpsLedger
from tests.integration.conftest import SKIP_IF_NO_POSTGRES

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

pytestmark = [pytest.mark.integration, SKIP_IF_NO_POSTGRES]


@pytest.fixture
async def ledger(pg_test_url: str) -> AsyncGenerator[OpsLedger]:
    pool = await asyncpg.create_pool(pg_test_url)
    yield OpsLedger(pool)
    await pool.execute("TRUNCATE ops.jobs CASCADE")
    await pool.close()


class TestCreateJob:
    async def test_row_contains_all_columns(self, ledger: OpsLedger):
        meeting_date = date(2026, 6, 1)
        submission = '{"source_type": "audio", "metadata": {"meeting_date": "2026-06-01"}}'
        raw_key = "jobs/2026-06-01/job-1/raw/input.mp3"

        await ledger.create_job("job-1", "user-1", "audio", None, datetime.now(UTC), meeting_date, submission, raw_key)

        row = await ledger.get_job("job-1")
        assert row is not None
        assert str(row["meeting_date"]) == "2026-06-01"
        assert json.loads(row["submission"])["source_type"] == "audio"
        assert row["raw_blob_key"] == raw_key
        assert row["status"] == "pending"

    async def test_not_null_constraints_are_satisfied(self, ledger: OpsLedger):
        # Verify the schema rejects a row that omits meeting_date — confirming the constraint exists.
        pool = ledger._pool
        with pytest.raises(asyncpg.NotNullViolationError):
            await pool.execute(
                "INSERT INTO ops.jobs (job_id, user_id, status, source_type, created_at, updated_at)"
                " VALUES ($1, $2, 'pending', $3, $4, $4)",
                "job-bad",
                "user-1",
                "audio",
                datetime.now(UTC),
            )
