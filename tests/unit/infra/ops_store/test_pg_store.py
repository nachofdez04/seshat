from datetime import UTC, date, datetime
from unittest.mock import AsyncMock, MagicMock

from seshat.core.config.settings import OpsStoreConfig
from seshat.core.models.enums import JobStatus, UserRole
from seshat.infra.ops_store.pg_store import PostgresOpsStore


def _make_store(**fetch_results) -> tuple[PostgresOpsStore, MagicMock]:
    pool = MagicMock()
    pool.fetchrow = AsyncMock(side_effect=lambda q, *a: fetch_results.get("fetchrow"))
    pool.fetchval = AsyncMock(side_effect=lambda q, *a: fetch_results.get("fetchval", 0))
    pool.fetch = AsyncMock(side_effect=lambda q, *a: fetch_results.get("fetch", []))
    pool.execute = AsyncMock()
    store = PostgresOpsStore(OpsStoreConfig(schema_name="ops"), "postgresql://unused")
    store._pool = pool
    return store, pool


class TestPostgresOpsStore:
    async def test_create_job(self):
        store, pool = _make_store()
        now = datetime.now(UTC)
        await store.create_job("job-1", "user-1", "audio", None, now, date(2026, 6, 1), "{}", "raw/key.mp3")
        pool.execute.assert_called_once()
        assert "INSERT INTO ops.jobs" in pool.execute.call_args[0][0]

    async def test_get_job_returns_row(self):
        row = {"job_id": "job-1", "status": "pending"}
        store, _ = _make_store(fetchrow=row)
        assert await store.get_job("job-1") == row

    async def test_get_job_not_found(self):
        store, _ = _make_store(fetchrow=None)
        assert await store.get_job("missing") is None

    async def test_update_job_status_non_terminal(self):
        store, pool = _make_store()
        await store.update_job_status("job-1", JobStatus.IDENTIFYING)
        call_args = pool.execute.call_args[0]
        assert "UPDATE ops.jobs" in call_args[0]
        assert "identifying" in call_args
        assert None in call_args  # finished_at is NULL for non-terminal

    async def test_update_job_status_terminal_sets_finished_at(self):
        store, pool = _make_store()
        await store.update_job_status("job-1", JobStatus.DONE)
        call_args = pool.execute.call_args[0]
        assert "finished_at" in call_args[0]
        # finished_at is a datetime, not None
        assert any(isinstance(a, datetime) and a is not None for a in call_args[1:])

    async def test_fail_job(self):
        store, pool = _make_store()
        await store.fail_job("job-1", "pipeline", "something broke", recoverable=True)
        call_args = pool.execute.call_args[0]
        assert "error_payload" in call_args[0]
        assert "finished_at" in call_args[0]

    async def test_count_recent_jobs_for_user(self):
        store, _ = _make_store(fetchval=3)
        assert await store.count_recent_jobs_for_user("user-1") == 3

    async def test_count_running_jobs(self):
        store, _ = _make_store(fetchval=1)
        assert await store.count_running_jobs() == 1

    async def test_get_api_keys(self):
        rows = [{"key_hash": "h1", "user_id": "alice", "role": "reviewer"}]
        store, _ = _make_store(fetch=rows)
        assert await store.get_api_keys() == rows

    async def test_reset_failed_job(self):
        store, pool = _make_store()
        await store.reset_failed_job("job-1")
        call_args = pool.execute.call_args[0]
        assert "pending" in call_args[0]
        assert "error_payload=NULL" in call_args[0]
        assert "finished_at=NULL" in call_args[0]

    async def test_list_jobs_no_filter(self):
        store, _ = _make_store(fetch=[{"job_id": "job-1"}, {"job_id": "job-2"}])
        assert len(await store.list_jobs()) == 2

    async def test_list_jobs_with_status_filter(self):
        store, pool = _make_store(fetch=[{"job_id": "job-1"}])
        rows = await store.list_jobs(status=JobStatus.DONE)
        assert len(rows) == 1
        assert "WHERE status=" in pool.fetch.call_args[0][0]

    async def test_list_jobs_source_type_filter(self):
        store, pool = _make_store(fetch=[{"job_id": "job-1"}])
        await store.list_jobs(source_type="audio")
        query = pool.fetch.call_args[0][0]
        assert "source_type=" in query

    async def test_list_jobs_date_range_filter(self):
        store, pool = _make_store(fetch=[])
        await store.list_jobs(meeting_date_from=date(2026, 1, 1), meeting_date_to=date(2026, 6, 30))
        query = pool.fetch.call_args[0][0]
        assert "meeting_date >=" in query
        assert "meeting_date <=" in query

    async def test_find_job_by_content_hash_only_returns_done_jobs(self):
        store, pool = _make_store(fetchval="job-1")
        result = await store.find_job_by_content_hash("abc123")
        assert result == "job-1"
        query = pool.fetchval.call_args[0][0]
        assert "status='done'" in query

    async def test_create_api_key(self):
        store, pool = _make_store()
        now = datetime.now(UTC)
        await store.create_api_key("hashed-key", "alice", UserRole.REVIEWER, now)
        call_args = pool.execute.call_args[0]
        assert "INSERT INTO ops.api_keys" in call_args[0]
        assert call_args[1] == "hashed-key"
        assert call_args[2] == "alice"

    async def test_list_api_keys(self):
        rows = [{"id": 1, "user_id": "alice", "role": "reviewer", "created_at": datetime.now(UTC), "revoked_at": None}]
        store, _ = _make_store(fetch=rows)
        assert len(await store.list_api_keys()) == 1

    def _make_revoke_store(self, revoked_at) -> tuple[PostgresOpsStore, AsyncMock]:
        conn = AsyncMock()
        conn.fetchrow = AsyncMock(return_value={"revoked_at": revoked_at} if revoked_at != "missing" else None)
        conn.execute = AsyncMock()
        conn.__aenter__ = AsyncMock(return_value=conn)
        conn.__aexit__ = AsyncMock(return_value=False)
        tx = AsyncMock()
        tx.__aenter__ = AsyncMock(return_value=tx)
        tx.__aexit__ = AsyncMock(return_value=False)
        conn.transaction = MagicMock(return_value=tx)
        pool = MagicMock()
        pool.acquire = MagicMock(return_value=conn)
        store = PostgresOpsStore(OpsStoreConfig(schema_name="ops"), "postgresql://unused")
        store._pool = pool
        return store, conn

    async def test_revoke_api_key_ok(self):
        store, conn = self._make_revoke_store(revoked_at=None)
        assert await store.revoke_api_key(1, datetime.now(UTC)) == "ok"
        conn.execute.assert_called_once()

    async def test_revoke_api_key_not_found(self):
        store, _ = self._make_revoke_store(revoked_at="missing")
        assert await store.revoke_api_key(99, datetime.now(UTC)) == "not_found"

    async def test_revoke_api_key_already_revoked(self):
        store, _ = self._make_revoke_store(revoked_at=datetime.now(UTC))
        assert await store.revoke_api_key(1, datetime.now(UTC)) == "already_revoked"

    async def test_get_stranded_writing_jobs(self):
        rows = [{"job_id": "job-1"}, {"job_id": "job-2"}]
        store, _ = _make_store(fetch=rows)
        assert await store.get_stranded_writing_jobs() == ["job-1", "job-2"]

    async def test_set_job_submission_executes_update(self):
        store, pool = _make_store()
        meeting_date = date(2026, 6, 28)
        await store.set_job_submission(
            "job-1", meeting_date, '{"source_type": "audio"}', "jobs/2026-06-28/job-1/raw/input.yaml"
        )
        call_args = pool.execute.call_args[0]
        assert "UPDATE ops.jobs" in call_args[0]
        assert "meeting_date" in call_args[0]
        assert call_args[1] == meeting_date
        assert call_args[2] == '{"source_type": "audio"}'
        assert call_args[3] == "jobs/2026-06-28/job-1/raw/input.yaml"
        assert call_args[5] == "job-1"
