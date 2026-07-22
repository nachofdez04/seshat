from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

from seshat.app.services.job import JobNotFoundError
from seshat.app.services.publishing import NothingToPublishError, PublishError, PublishGitError
from seshat.core.models.enums import UserRole
from seshat.core.models.publishing import PublishResult
from tests.unit.app.platform.api.conftest import make_app_state, make_current_user


def _make_result(job_id: str = "job-1") -> PublishResult:
    return PublishResult(
        job_id=job_id,
        branch="seshat/meeting/job-1-2026-07-22",
        commit_sha="abc1234",
        pr_url="https://github.com/acme/docs/pull/7",
        files=["meetings/job-1/meeting_summary/meeting_summary.md"],
        published_at=datetime.now(UTC),
    )


def _make_app_state(**overrides):
    publishing_service = MagicMock()
    publishing_service.publish_job = AsyncMock(return_value=_make_result())
    publishing_service.get_latest = AsyncMock(return_value=_make_result())
    return make_app_state(publishing_service=publishing_service, **overrides)


class TestPublishJob:
    async def test_requires_auth(self, api_client):
        async with api_client(_make_app_state()) as ac:
            resp = await ac.post("/jobs/job-1/publish")

        assert resp.status_code == 401

    async def test_viewer_forbidden(self, api_client):
        async with api_client(_make_app_state(), make_current_user(role=UserRole.VIEWER)) as ac:
            resp = await ac.post("/jobs/job-1/publish")

        assert resp.status_code == 403

    async def test_operator_publishes(self, api_client):
        state = _make_app_state()
        async with api_client(state, make_current_user()) as ac:
            resp = await ac.post("/jobs/job-1/publish")

        assert resp.status_code == 200
        body = resp.json()
        assert body["nothing_to_publish"] is False
        assert body["result"]["commit_sha"] == "abc1234"
        assert body["result"]["pr_url"] == "https://github.com/acme/docs/pull/7"
        state.publishing_service.publish_job.assert_awaited_once_with("job-1")

    async def test_nothing_to_publish_returns_benign_200(self, api_client):
        state = _make_app_state()
        state.publishing_service.publish_job = AsyncMock(side_effect=NothingToPublishError("already up to date"))
        async with api_client(state, make_current_user()) as ac:
            resp = await ac.post("/jobs/job-1/publish")

        assert resp.status_code == 200
        body = resp.json()
        assert body["nothing_to_publish"] is True
        assert body["result"] is None
        assert "already up to date" in body["detail"]

    async def test_unknown_job_returns_404(self, api_client):
        state = _make_app_state()
        state.publishing_service.publish_job = AsyncMock(side_effect=JobNotFoundError("job-1"))
        async with api_client(state, make_current_user()) as ac:
            resp = await ac.post("/jobs/job-1/publish")

        assert resp.status_code == 404

    async def test_guard_failure_returns_409(self, api_client):
        state = _make_app_state()
        state.publishing_service.publish_job = AsyncMock(side_effect=PublishError("Git publishing is disabled"))
        async with api_client(state, make_current_user()) as ac:
            resp = await ac.post("/jobs/job-1/publish")

        assert resp.status_code == 409
        assert "disabled" in resp.json()["detail"]

    async def test_git_failure_returns_500_with_detail(self, api_client):
        state = _make_app_state()
        state.publishing_service.publish_job = AsyncMock(side_effect=PublishGitError("Git operation failed: push"))
        async with api_client(state, make_current_user()) as ac:
            resp = await ac.post("/jobs/job-1/publish")

        assert resp.status_code == 500
        assert "push" in resp.json()["detail"]


class TestGetPublishResult:
    async def test_requires_auth(self, api_client):
        async with api_client(_make_app_state()) as ac:
            resp = await ac.get("/jobs/job-1/publish")

        assert resp.status_code == 401

    async def test_viewer_reads_latest_result(self, api_client):
        state = _make_app_state()
        async with api_client(state, make_current_user(role=UserRole.VIEWER)) as ac:
            resp = await ac.get("/jobs/job-1/publish")

        assert resp.status_code == 200
        assert resp.json()["branch"] == "seshat/meeting/job-1-2026-07-22"
        state.publishing_service.get_latest.assert_awaited_once_with("job-1")

    async def test_never_published_returns_404(self, api_client):
        state = _make_app_state()
        state.publishing_service.get_latest = AsyncMock(return_value=None)
        async with api_client(state, make_current_user(role=UserRole.VIEWER)) as ac:
            resp = await ac.get("/jobs/job-1/publish")

        assert resp.status_code == 404
