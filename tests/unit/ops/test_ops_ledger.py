from datetime import UTC, date, datetime
from unittest.mock import AsyncMock, MagicMock

from seshat.models.enums import JobStatus
from seshat.ops.ledger import OpsLedger


def _make_pool(**fetch_results):
    pool = MagicMock()
    pool.fetchrow = AsyncMock(side_effect=lambda q, *a: fetch_results.get("fetchrow"))
    pool.fetchval = AsyncMock(side_effect=lambda q, *a: fetch_results.get("fetchval", 0))
    pool.fetch = AsyncMock(side_effect=lambda q, *a: fetch_results.get("fetch", []))
    pool.execute = AsyncMock()
    return pool


def _make_ledger(**fetch_results) -> OpsLedger:
    pool = _make_pool(**fetch_results)
    return OpsLedger(pool)


class TestOpsLedger:
    async def test_create_job(self):
        store = _make_ledger()
        now = datetime.now(UTC)
        await store.create_job("job-1", "user-1", "audio", None, now, date(2026, 6, 1), "{}", "raw/key.mp3")
        store._pool.execute.assert_called_once()
        call_args = store._pool.execute.call_args[0]
        assert "INSERT INTO ops.jobs" in call_args[0]

    async def test_get_job_returns_row(self):
        row = {
            "job_id": "job-1",
            "status": "pending",
            "idempotency_key": None,
            "created_at": datetime.now(UTC),
            "updated_at": datetime.now(UTC),
            "error_payload": None,
            "mlflow_run_id": None,
        }
        store = _make_ledger(fetchrow=row)
        result = await store.get_job("job-1")
        assert result == row

    async def test_get_job_not_found(self):
        store = _make_ledger(fetchrow=None)
        result = await store.get_job("missing")
        assert result is None

    async def test_update_job_status(self):
        store = _make_ledger()
        await store.update_job_status("job-1", JobStatus.EXTRACTING)
        store._pool.execute.assert_called_once()
        call_args = store._pool.execute.call_args[0]
        assert "UPDATE ops.jobs" in call_args[0]
        assert "extracting" in call_args

    async def test_fail_job(self):
        store = _make_ledger()
        await store.fail_job("job-1", "pipeline", "something broke", recoverable=True)
        store._pool.execute.assert_called_once()
        call_args = store._pool.execute.call_args[0]
        assert "error_payload" in call_args[0]

    async def test_count_active_jobs_per_user(self):
        store = _make_ledger(fetchval=3)
        count = await store.count_recent_jobs_for_user("user-1")
        assert count == 3

    async def test_count_running_jobs(self):
        store = _make_ledger(fetchval=1)
        count = await store.count_running_jobs()
        assert count == 1

    async def test_get_api_keys(self):
        rows = [{"key_hash": "h1", "user_id": "alice", "role": "reviewer"}]
        store = _make_ledger(fetch=rows)
        keys = await store.get_api_keys()
        assert keys == [("h1", "alice", "reviewer")]

    async def test_reset_failed_job(self):
        store = _make_ledger()
        await store.reset_failed_job("job-1")
        store._pool.execute.assert_called_once()
        call_args = store._pool.execute.call_args[0]
        assert "pending" in call_args[0]
        assert "error_payload=NULL" in call_args[0]

    async def test_get_stranded_writing_jobs(self):
        rows = [{"job_id": "job-1"}, {"job_id": "job-2"}]
        store = _make_ledger(fetch=rows)
        ids = await store.get_stranded_writing_jobs()
        assert ids == ["job-1", "job-2"]

    async def test_set_job_submission_executes_update(self):
        store = _make_ledger()
        meeting_date = date(2026, 6, 28)
        await store.set_job_submission(
            "job-1", meeting_date, '{"source_type": "audio"}', "jobs/2026-06-28/job-1/raw/input.yaml"
        )
        store._pool.execute.assert_called_once()
        call_args = store._pool.execute.call_args[0]
        assert "UPDATE ops.jobs" in call_args[0]
        assert "meeting_date" in call_args[0]
        assert "submission" in call_args[0]
        assert "raw_blob_key" in call_args[0]
        assert call_args[1] == meeting_date
        assert call_args[2] == '{"source_type": "audio"}'
        assert call_args[3] == "jobs/2026-06-28/job-1/raw/input.yaml"
        assert call_args[5] == "job-1"
