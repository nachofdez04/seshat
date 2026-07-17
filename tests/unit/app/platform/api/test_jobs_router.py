from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from seshat.app.pipeline.ingestion.audio_validator import (
    AudioValidationError,
    FileTooLargeError,
    UnsupportedFormatError,
)
from seshat.app.pipeline.ingestion.text_validator import TextValidationError
from seshat.app.services.job import (
    JobNotFoundError,
    JobStateError,
    RateLimitExceededError,
    TranscriptNotFoundError,
)
from seshat.core.models.api_responses import JobActionResponse, JobSubmitResponse
from seshat.core.models.enums import JobStatus, UserRole
from seshat.core.models.nodes import ExtractionResult
from tests.helpers import make_node
from tests.unit.app.platform.api.conftest import make_app_state, make_current_user


def _make_app_state(**overrides):
    job_service = MagicMock()
    job_service.submit = AsyncMock(return_value=JobSubmitResponse(job_id="job-1"))
    job_service.get = AsyncMock(return_value=None)
    job_service.list_jobs = AsyncMock(return_value=[])
    job_service.get_result = AsyncMock(return_value=None)
    job_service.approve = AsyncMock(return_value=JobActionResponse(status="accepted"))
    job_service.retry = AsyncMock(return_value=JobActionResponse(status="accepted"))
    job_service.get_transcript_excerpt = AsyncMock(return_value="Hello world")

    config = MagicMock()
    config.api.max_jobs_per_user_per_hour = 10
    config.api.max_concurrent_jobs = 5

    return make_app_state(config=config, job_service=job_service, **overrides)


def _make_job_response(status: str = "pending") -> dict[str, Any]:
    from seshat.core.models.jobs import JobResponse

    now = datetime.now(UTC)
    return JobResponse(
        job_id="job-1",
        status=status,
        created_at=now,
        updated_at=now,
    )


class TestSubmitJob:
    async def test_requires_auth(self, api_client):
        async with api_client(_make_app_state()) as ac:
            resp = await ac.post("/jobs", files={"file": b"data"}, data={"body": "{}"})
        assert resp.status_code == 401

    async def test_returns_job_id(self, api_client):
        state = _make_app_state()
        body = json.dumps({"source_type": "text", "metadata": {"meeting_date": "2026-01-15"}})
        async with api_client(state, make_current_user()) as ac:
            resp = await ac.post("/jobs", files={"file": ("input.yaml", b"data", "text/plain")}, data={"body": body})
        assert resp.status_code == 202
        assert "job_id" in resp.json()
        assert state.job_service.submit.call_args.args[3] == "alice"

    async def test_idempotency_returns_existing_job(self, api_client):
        state = _make_app_state()
        state.job_service.submit = AsyncMock(return_value=JobSubmitResponse(job_id="existing-job"))
        body = json.dumps(
            {"source_type": "text", "metadata": {"meeting_date": "2026-01-15"}, "idempotency_key": "key-abc"}
        )
        async with api_client(state, make_current_user()) as ac:
            resp = await ac.post("/jobs", files={"file": b"data"}, data={"body": body})
        assert resp.status_code == 202
        assert resp.json()["job_id"] == "existing-job"

    async def test_rate_limit_per_user(self, api_client):
        state = _make_app_state()
        state.job_service.submit = AsyncMock(side_effect=RateLimitExceededError("per_user_hourly_cap"))
        body = json.dumps({"source_type": "text", "metadata": {"meeting_date": "2026-01-15"}})
        async with api_client(state, make_current_user()) as ac:
            resp = await ac.post("/jobs", files={"file": b"data"}, data={"body": body})
        assert resp.status_code == 429
        assert resp.json()["limit_type"] == "per_user_hourly_cap"

    async def test_rate_limit_global_concurrency(self, api_client):
        state = _make_app_state()
        state.job_service.submit = AsyncMock(side_effect=RateLimitExceededError("global_concurrency_cap"))
        body = json.dumps({"source_type": "text", "metadata": {"meeting_date": "2026-01-15"}})
        async with api_client(state, make_current_user()) as ac:
            resp = await ac.post("/jobs", files={"file": b"data"}, data={"body": body})
        assert resp.status_code == 429
        assert resp.json()["limit_type"] == "global_concurrency_cap"

    async def test_rejects_file_without_extension(self, api_client):
        state = _make_app_state()
        state.job_service.submit = AsyncMock(side_effect=ValueError("Uploaded file must have an extension."))
        body = json.dumps({"source_type": "text", "metadata": {"meeting_date": "2026-01-15"}})
        async with api_client(state, make_current_user()) as ac:
            resp = await ac.post("/jobs", files={"file": ("noextension", b"data", "text/plain")}, data={"body": body})
        assert resp.status_code == 400

    async def test_viewer_cannot_submit(self, api_client):
        body = json.dumps({"source_type": "text", "metadata": {"meeting_date": "2026-01-15"}})
        async with api_client(_make_app_state(), make_current_user(role=UserRole.VIEWER)) as ac:
            resp = await ac.post("/jobs", files={"file": b"data"}, data={"body": body})
        assert resp.status_code == 403

    async def test_overrides_require_operator(self, api_client):
        body = json.dumps(
            {"source_type": "text", "metadata": {"meeting_date": "2026-01-15"}, "overrides": {"extraction": {}}}
        )
        async with api_client(_make_app_state(), make_current_user(role=UserRole.REVIEWER)) as ac:
            resp = await ac.post("/jobs", files={"file": b"data"}, data={"body": body})
        assert resp.status_code == 403

    async def test_malformed_body_json_returns_422(self, api_client):
        async with api_client(_make_app_state(), make_current_user()) as ac:
            resp = await ac.post("/jobs", files={"file": b"data"}, data={"body": "not-json"})
        assert resp.status_code == 422

    async def test_missing_required_fields_in_body_returns_422(self, api_client):
        async with api_client(_make_app_state(), make_current_user()) as ac:
            resp = await ac.post("/jobs", files={"file": b"data"}, data={"body": "{}"})
        assert resp.status_code == 422

    async def test_force_requires_admin(self, api_client):
        body = json.dumps({"source_type": "text", "metadata": {"meeting_date": "2026-01-15"}, "force": True})
        async with api_client(_make_app_state(), make_current_user(role=UserRole.OPERATOR)) as ac:
            resp = await ac.post("/jobs", files={"file": ("input.yaml", b"data", "text/plain")}, data={"body": body})
        assert resp.status_code == 403

    async def test_file_too_large_returns_413(self, api_client):
        state = _make_app_state()
        state.job_service.submit = AsyncMock(side_effect=FileTooLargeError("File size 999 exceeds maximum 10 bytes"))
        body = json.dumps({"source_type": "audio", "metadata": {"meeting_date": "2026-01-15"}})
        async with api_client(state, make_current_user()) as ac:
            resp = await ac.post("/jobs", files={"file": ("rec.mp3", b"data", "audio/mpeg")}, data={"body": body})
        assert resp.status_code == 413

    async def test_unsupported_format_returns_415(self, api_client):
        state = _make_app_state()
        state.job_service.submit = AsyncMock(side_effect=UnsupportedFormatError("Extension mismatch"))
        body = json.dumps({"source_type": "audio", "metadata": {"meeting_date": "2026-01-15"}})
        async with api_client(state, make_current_user()) as ac:
            resp = await ac.post("/jobs", files={"file": ("rec.mp3", b"data", "audio/mpeg")}, data={"body": body})
        assert resp.status_code == 415

    async def test_invalid_audio_returns_422(self, api_client):
        state = _make_app_state()
        state.job_service.submit = AsyncMock(side_effect=AudioValidationError("Unable to determine audio duration"))
        body = json.dumps({"source_type": "audio", "metadata": {"meeting_date": "2026-01-15"}})
        async with api_client(state, make_current_user()) as ac:
            resp = await ac.post("/jobs", files={"file": ("rec.mp3", b"data", "audio/mpeg")}, data={"body": body})
        assert resp.status_code == 422

    async def test_invalid_text_returns_422(self, api_client):
        state = _make_app_state()
        state.job_service.submit = AsyncMock(side_effect=TextValidationError("Invalid YAML"))
        body = json.dumps({"source_type": "text", "metadata": {"meeting_date": "2026-01-15"}})
        async with api_client(state, make_current_user()) as ac:
            resp = await ac.post("/jobs", files={"file": ("input.yaml", b"data", "text/plain")}, data={"body": body})
        assert resp.status_code == 422


class TestListJobs:
    async def test_returns_jobs(self, api_client):
        state = _make_app_state()
        state.job_service.list_jobs = AsyncMock(
            return_value=[_make_job_response("pending"), _make_job_response("done")]
        )
        async with api_client(state, make_current_user()) as ac:
            resp = await ac.get("/jobs")
        assert resp.status_code == 200
        assert len(resp.json()) == 2

    async def test_filters_by_status(self, api_client):
        state = _make_app_state()
        state.job_service.list_jobs = AsyncMock(return_value=[_make_job_response("done")])
        async with api_client(state, make_current_user()) as ac:
            resp = await ac.get("/jobs", params={"job_status": "done"})
        assert resp.status_code == 200
        state.job_service.list_jobs.assert_called_once_with(
            status=JobStatus.DONE,
            source_type=None,
            meeting_date_from=None,
            meeting_date_to=None,
            limit=50,
            offset=0,
        )

    async def test_forwards_source_type_filter(self, api_client):
        state = _make_app_state()
        state.job_service.list_jobs = AsyncMock(return_value=[])
        async with api_client(state, make_current_user()) as ac:
            await ac.get("/jobs", params={"source_type": "audio"})
        call_kwargs = state.job_service.list_jobs.call_args.kwargs
        assert call_kwargs["source_type"] == "audio"

    async def test_forwards_date_range_filters(self, api_client):
        state = _make_app_state()
        state.job_service.list_jobs = AsyncMock(return_value=[])
        async with api_client(state, make_current_user()) as ac:
            await ac.get("/jobs", params={"meeting_date_from": "2026-01-01", "meeting_date_to": "2026-06-30"})
        call_kwargs = state.job_service.list_jobs.call_args.kwargs
        assert str(call_kwargs["meeting_date_from"]) == "2026-01-01"
        assert str(call_kwargs["meeting_date_to"]) == "2026-06-30"

    async def test_negative_limit_returns_422(self, api_client):
        async with api_client(_make_app_state(), make_current_user()) as ac:
            resp = await ac.get("/jobs?limit=-1")
        assert resp.status_code == 422

    async def test_negative_offset_returns_422(self, api_client):
        async with api_client(_make_app_state(), make_current_user()) as ac:
            resp = await ac.get("/jobs?offset=-1")
        assert resp.status_code == 422


class TestGetJob:
    async def test_requires_auth(self, api_client):
        async with api_client(_make_app_state()) as ac:
            resp = await ac.get("/jobs/job-1")
        assert resp.status_code == 401

    async def test_not_found(self, api_client):
        async with api_client(_make_app_state(), make_current_user()) as ac:
            resp = await ac.get("/jobs/job-1")
        assert resp.status_code == 404

    async def test_returns_job_response(self, api_client):
        state = _make_app_state()
        state.job_service.get = AsyncMock(return_value=_make_job_response("pending"))
        async with api_client(state, make_current_user()) as ac:
            resp = await ac.get("/jobs/job-1")
        assert resp.status_code == 200
        assert resp.json()["job_id"] == "job-1"
        assert resp.json()["status"] == "pending"


class TestGetJobResults:
    async def test_requires_auth(self, api_client):
        async with api_client(_make_app_state()) as ac:
            resp = await ac.get("/jobs/job-1/results")
        assert resp.status_code == 401

    async def test_results_not_ready(self, api_client):
        state = _make_app_state()
        state.job_service.get_result = AsyncMock(side_effect=JobStateError("Results not yet available"))
        async with api_client(state, make_current_user()) as ac:
            resp = await ac.get("/jobs/job-1/results")
        assert resp.status_code == 409

    async def test_returns_result_when_awaiting_review(self, api_client):
        node = make_node()
        result = ExtractionResult(job_id="job-1", nodes=[node], relationships=[])
        state = _make_app_state()
        state.job_service.get_result = AsyncMock(return_value=result)
        async with api_client(state, make_current_user()) as ac:
            resp = await ac.get("/jobs/job-1/results")
        assert resp.status_code == 200

    async def test_not_found_when_service_raises(self, api_client):
        state = _make_app_state()
        state.job_service.get_result = AsyncMock(side_effect=JobNotFoundError("job-1"))
        async with api_client(state, make_current_user()) as ac:
            resp = await ac.get("/jobs/job-1/results")
        assert resp.status_code == 404

    async def test_returns_404_when_result_is_none(self, api_client):
        state = _make_app_state()
        state.job_service.get_result = AsyncMock(return_value=None)
        async with api_client(state, make_current_user()) as ac:
            resp = await ac.get("/jobs/job-1/results")
        assert resp.status_code == 404


class TestApproveJob:
    async def test_requires_auth(self, api_client):
        async with api_client(_make_app_state()) as ac:
            resp = await ac.post("/jobs/job-1/approve", json={"decisions": [{"node_id": "n1", "action": "approve"}]})
        assert resp.status_code == 401

    async def test_requires_reviewer_or_operator(self, api_client):
        async with api_client(_make_app_state(), make_current_user(role=UserRole.VIEWER)) as ac:
            resp = await ac.post("/jobs/job-1/approve", json={"decisions": [{"node_id": "n1", "action": "approve"}]})
        assert resp.status_code == 403

    async def test_not_awaiting_review(self, api_client):
        state = _make_app_state()
        state.job_service.approve = AsyncMock(side_effect=JobStateError("Job is not awaiting review"))
        async with api_client(state, make_current_user()) as ac:
            resp = await ac.post("/jobs/job-1/approve", json={"decisions": [{"node_id": "n1", "action": "approve"}]})
        assert resp.status_code == 409

    async def test_not_found(self, api_client):
        state = _make_app_state()
        state.job_service.approve = AsyncMock(side_effect=JobNotFoundError("job-1"))
        async with api_client(state, make_current_user()) as ac:
            resp = await ac.post("/jobs/job-1/approve", json={"decisions": [{"node_id": "n1", "action": "approve"}]})
        assert resp.status_code == 404

    async def test_returns_accepted(self, api_client):
        state = _make_app_state()
        async with api_client(state, make_current_user()) as ac:
            resp = await ac.post(
                "/jobs/job-1/approve",
                json={"decisions": [{"node_id": "n1", "action": "approve"}]},
            )
        assert resp.status_code == 200
        assert resp.json()["status"] == "accepted"

    async def test_passes_user_id_to_service(self, api_client):
        state = _make_app_state()
        async with api_client(state, make_current_user()) as ac:
            await ac.post(
                "/jobs/job-1/approve",
                json={"decisions": [{"node_id": "n1", "action": "approve"}]},
            )
        state.job_service.approve.assert_called_once()
        assert state.job_service.approve.call_args.args[2] == "alice"


class TestRetryJob:
    async def test_requires_auth(self, api_client):
        async with api_client(_make_app_state()) as ac:
            resp = await ac.post("/jobs/job-1/retry")
        assert resp.status_code == 401

    async def test_requires_operator(self, api_client):
        async with api_client(_make_app_state(), make_current_user(role=UserRole.REVIEWER)) as ac:
            resp = await ac.post("/jobs/job-1/retry")
        assert resp.status_code == 403

    async def test_not_failed(self, api_client):
        state = _make_app_state()
        state.job_service.retry = AsyncMock(side_effect=JobStateError("Only failed jobs can be retried"))
        async with api_client(state, make_current_user()) as ac:
            resp = await ac.post("/jobs/job-1/retry")
        assert resp.status_code == 409

    async def test_not_found(self, api_client):
        state = _make_app_state()
        state.job_service.retry = AsyncMock(side_effect=JobNotFoundError("job-1"))
        async with api_client(state, make_current_user()) as ac:
            resp = await ac.post("/jobs/job-1/retry")
        assert resp.status_code == 404

    async def test_no_stored_input_returns_409(self, api_client):
        state = _make_app_state()
        state.job_service.retry = AsyncMock(
            side_effect=JobStateError("Job has no stored input; re-submit via POST /jobs")
        )
        async with api_client(state, make_current_user()) as ac:
            resp = await ac.post("/jobs/job-1/retry")
        assert resp.status_code == 409

    async def test_returns_accepted(self, api_client):
        state = _make_app_state()
        async with api_client(state, make_current_user()) as ac:
            resp = await ac.post("/jobs/job-1/retry")
        assert resp.status_code == 200
        assert resp.json()["status"] == "accepted"


class TestGetTranscriptExcerpt:
    async def test_requires_auth(self, api_client):
        async with api_client(_make_app_state()) as ac:
            resp = await ac.get("/jobs/job-1/transcript/excerpt")
        assert resp.status_code == 401

    async def test_returns_excerpt(self, api_client):
        state = _make_app_state()
        state.job_service.get_transcript_excerpt = AsyncMock(return_value="Hello")
        async with api_client(state, make_current_user()) as ac:
            resp = await ac.get("/jobs/job-1/transcript/excerpt", params={"char_start": 0, "char_end": 5})
        assert resp.status_code == 200
        assert resp.json()["text"] == "Hello"

    async def test_job_not_found_returns_404(self, api_client):
        state = _make_app_state()
        state.job_service.get_transcript_excerpt = AsyncMock(side_effect=JobNotFoundError("job-1"))
        async with api_client(state, make_current_user()) as ac:
            resp = await ac.get("/jobs/job-1/transcript/excerpt")
        assert resp.status_code == 404

    async def test_transcript_not_found_returns_404(self, api_client):
        state = _make_app_state()
        state.job_service.get_transcript_excerpt = AsyncMock(side_effect=TranscriptNotFoundError("job-1"))
        async with api_client(state, make_current_user()) as ac:
            resp = await ac.get("/jobs/job-1/transcript/excerpt")
        assert resp.status_code == 404

    async def test_negative_char_start_returns_422(self, api_client):
        async with api_client(_make_app_state(), make_current_user()) as ac:
            resp = await ac.get("/jobs/job-1/transcript/excerpt", params={"char_start": -1, "char_end": 5})
        assert resp.status_code == 422

    async def test_start_not_less_than_end_returns_422(self, api_client):
        state = _make_app_state()
        state.job_service.get_transcript_excerpt = AsyncMock(
            side_effect=ValueError("char_start (10) must be less than char_end (5)")
        )
        async with api_client(state, make_current_user()) as ac:
            resp = await ac.get("/jobs/job-1/transcript/excerpt", params={"char_start": 10, "char_end": 5})
        assert resp.status_code == 422
