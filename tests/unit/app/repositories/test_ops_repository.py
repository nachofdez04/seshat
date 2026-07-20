from datetime import UTC, date, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from seshat.app.repositories.ops_repository import ApiKeyAlreadyRevokedError, ApiKeyNotFoundError, OpsRepository
from seshat.core.models.enums import JobStatus, UserRole


def _make_repo(**store_returns) -> OpsRepository:
    store = MagicMock()
    for method, return_value in store_returns.items():
        setattr(store, method, AsyncMock(return_value=return_value))
    for method in [
        "create_job",
        "get_job",
        "find_job_by_idempotency_key",
        "find_job_by_content_hash",
        "list_jobs",
        "count_recent_jobs_for_user",
        "count_running_jobs",
        "get_stranded_writing_jobs",
        "update_job_status",
        "fail_job",
        "reset_failed_job",
        "set_job_submission",
        "create_api_key",
        "get_api_keys",
        "list_api_keys",
        "revoke_api_key",
        "is_alive",
    ]:
        if not hasattr(store, method) or not isinstance(getattr(store, method), AsyncMock):
            setattr(store, method, AsyncMock(return_value=None))
    return OpsRepository(store)


class TestOpsRepository:
    async def test_create_job_delegates(self):
        repo = _make_repo()
        now = datetime.now(UTC)
        await repo.create_job("job-1", "user-1", "audio", None, now, date(2026, 6, 1), "{}", "raw/key.mp3")
        repo._store.create_job.assert_called_once_with(
            "job-1", "user-1", "audio", None, now, date(2026, 6, 1), "{}", "raw/key.mp3", None
        )

    async def test_get_job_delegates(self):
        row = {"job_id": "job-1", "status": "pending"}
        repo = _make_repo(get_job=row)
        assert await repo.get_job("job-1") == row

    async def test_get_job_not_found(self):
        repo = _make_repo(get_job=None)
        assert await repo.get_job("missing") is None

    async def test_update_job_status_delegates(self):
        repo = _make_repo()
        await repo.update_job_status("job-1", JobStatus.IDENTIFYING)
        repo._store.update_job_status.assert_called_once_with("job-1", JobStatus.IDENTIFYING)

    async def test_fail_job_delegates(self):
        repo = _make_repo()
        await repo.fail_job("job-1", "pipeline", "broke", recoverable=True)
        repo._store.fail_job.assert_called_once_with("job-1", "pipeline", "broke", recoverable=True)

    async def test_count_recent_jobs_for_user(self):
        repo = _make_repo(count_recent_jobs_for_user=3)
        assert await repo.count_recent_jobs_for_user("user-1") == 3

    async def test_count_running_jobs(self):
        repo = _make_repo(count_running_jobs=1)
        assert await repo.count_running_jobs() == 1

    async def test_get_api_keys_converts_to_tuples(self):
        rows = [{"key_hash": "h1", "user_id": "alice", "role": "reviewer"}]
        repo = _make_repo(get_api_keys=rows)
        assert await repo.get_api_keys() == [("h1", "alice", "reviewer")]

    async def test_reset_failed_job_delegates(self):
        repo = _make_repo()
        await repo.reset_failed_job("job-1")
        repo._store.reset_failed_job.assert_called_once_with("job-1")

    async def test_list_jobs_no_filter(self):
        repo = _make_repo(list_jobs=[{"job_id": "job-1"}, {"job_id": "job-2"}])
        assert len(await repo.list_jobs()) == 2

    async def test_list_jobs_with_status_filter(self):
        repo = _make_repo(list_jobs=[{"job_id": "job-1"}])
        rows = await repo.list_jobs(status=JobStatus.DONE)
        assert len(rows) == 1
        repo._store.list_jobs.assert_called_once_with(JobStatus.DONE, None, None, None, 50, 0)

    async def test_create_api_key_delegates(self):
        repo = _make_repo()
        now = datetime.now(UTC)
        await repo.create_api_key("hashed-key", "alice", UserRole.REVIEWER, now)
        repo._store.create_api_key.assert_called_once_with("hashed-key", "alice", UserRole.REVIEWER, now)

    async def test_list_api_keys_delegates(self):
        rows = [{"id": 1, "user_id": "alice"}]
        repo = _make_repo(list_api_keys=rows)
        assert await repo.list_api_keys() == rows

    async def test_revoke_api_key_ok(self):
        repo = _make_repo(revoke_api_key="ok")
        await repo.revoke_api_key(1, datetime.now(UTC))  # no exception

    async def test_revoke_api_key_not_found(self):
        repo = _make_repo(revoke_api_key="not_found")
        with pytest.raises(ApiKeyNotFoundError):
            await repo.revoke_api_key(99, datetime.now(UTC))

    async def test_revoke_api_key_already_revoked(self):
        repo = _make_repo(revoke_api_key="already_revoked")
        with pytest.raises(ApiKeyAlreadyRevokedError):
            await repo.revoke_api_key(1, datetime.now(UTC))

    async def test_get_stranded_writing_jobs(self):
        repo = _make_repo(get_stranded_writing_jobs=["job-1", "job-2"])
        assert await repo.get_stranded_writing_jobs() == ["job-1", "job-2"]

    async def test_set_job_submission_delegates(self):
        repo = _make_repo()
        meeting_date = date(2026, 6, 28)
        await repo.set_job_submission("job-1", meeting_date, '{"source_type": "audio"}', "raw/key.yaml")
        repo._store.set_job_submission.assert_called_once_with(
            "job-1", meeting_date, '{"source_type": "audio"}', "raw/key.yaml"
        )

    async def test_find_job_by_content_hash_hit(self):
        repo = _make_repo(find_job_by_content_hash="job-1")
        result = await repo.find_job_by_content_hash("sha256abc")
        assert result == "job-1"

    async def test_find_job_by_content_hash_miss(self):
        repo = _make_repo(find_job_by_content_hash=None)
        result = await repo.find_job_by_content_hash("sha256abc")
        assert result is None

    async def test_find_job_by_idempotency_key_hit(self):
        row = {"job_id": "job-1", "status": "done"}
        repo = _make_repo(find_job_by_idempotency_key=row)
        result = await repo.find_job_by_idempotency_key("key-abc")
        assert result == row

    async def test_find_job_by_idempotency_key_miss(self):
        repo = _make_repo(find_job_by_idempotency_key=None)
        result = await repo.find_job_by_idempotency_key("key-abc")
        assert result is None

    async def test_is_alive_true(self):
        repo = _make_repo(is_alive=True)
        assert await repo.is_alive() is True

    async def test_is_alive_false(self):
        repo = _make_repo(is_alive=False)
        assert await repo.is_alive() is False
