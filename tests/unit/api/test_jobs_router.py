from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from seshat.api.state import AppState
from seshat.models.enums import JobStatus, NodeStatus, UserRole
from seshat.models.nodes import ExtractionResult
from tests.helpers import make_node
from tests.unit.api.conftest import make_current_user


def _make_app_state(**overrides) -> AppState:
    ops = MagicMock()
    ops.find_job_by_idempotency_key = AsyncMock(return_value=None)
    ops.count_recent_jobs_for_user = AsyncMock(return_value=0)
    ops.count_running_jobs = AsyncMock(return_value=0)
    ops.create_job = AsyncMock()
    ops.get_job = AsyncMock(return_value=None)
    ops.update_job_status = AsyncMock()
    ops.reset_failed_job = AsyncMock()
    ops.fail_job = AsyncMock()
    ops.set_job_submission = AsyncMock()

    config = MagicMock()
    config.max_jobs_per_user_per_hour = 10
    config.max_concurrent_jobs = 5

    queue = MagicMock()
    queue.enqueue = AsyncMock()

    blob_store = MagicMock()
    blob_store.put = AsyncMock()
    blob_store.get = AsyncMock()
    blob_store.raw_input_key = MagicMock(return_value="raw/key")
    blob_store.curated_extraction_key = MagicMock(return_value="curated/key")

    state = AppState(
        ops=ops,
        kb_store=MagicMock(),
        config=config,
        queue=queue,
        results={},
        runner=MagicMock(),
        manual_ingestion=MagicMock(),
        blob_store=blob_store,
    )
    for k, v in overrides.items():
        object.__setattr__(state, k, v)
    return state


def _make_job_row(status: str = "pending", *, meeting_date=None, submission=None, raw_blob_key=None) -> dict[str, Any]:
    return {
        "job_id": "job-1",
        "status": status,
        "idempotency_key": None,
        "created_at": datetime.now(UTC),
        "error_payload": None,
        "mlflow_run_id": None,
        "meeting_date": meeting_date,
        "submission": submission,
        "raw_blob_key": raw_blob_key,
    }


class TestSubmitJob:
    async def test_requires_auth(self, api_client):
        async with api_client(_make_app_state()) as ac:
            resp = await ac.post("/jobs", files={"file": b"data"}, data={"body": "{}"})
        assert resp.status_code == 401

    async def test_returns_job_id(self, api_client):
        body = json.dumps({"source_type": "text", "metadata": {"meeting_date": "2026-01-15"}})
        async with api_client(_make_app_state(), make_current_user()) as ac:
            resp = await ac.post("/jobs", files={"file": ("input.yaml", b"data", "text/plain")}, data={"body": body})
        assert resp.status_code == 202
        assert "job_id" in resp.json()

    async def test_idempotency_returns_existing_job(self, api_client):
        state = _make_app_state()
        state.ops.find_job_by_idempotency_key = AsyncMock(return_value={"job_id": "existing-job", "status": "pending"})
        body = json.dumps(
            {"source_type": "text", "metadata": {"meeting_date": "2026-01-15"}, "idempotency_key": "key-abc"}
        )
        async with api_client(state, make_current_user()) as ac:
            resp = await ac.post("/jobs", files={"file": b"data"}, data={"body": body})
        assert resp.status_code == 202
        assert resp.json()["job_id"] == "existing-job"
        state.ops.create_job.assert_not_called()

    async def test_rate_limit_per_user(self, api_client):
        state = _make_app_state()
        state.ops.count_recent_jobs_for_user = AsyncMock(return_value=10)
        body = json.dumps({"source_type": "text", "metadata": {"meeting_date": "2026-01-15"}})
        async with api_client(state, make_current_user()) as ac:
            resp = await ac.post("/jobs", files={"file": b"data"}, data={"body": body})
        assert resp.status_code == 429
        assert resp.json()["limit_type"] == "per_user_hourly_cap"

    async def test_rate_limit_global_concurrency(self, api_client):
        state = _make_app_state()
        state.ops.count_running_jobs = AsyncMock(return_value=5)
        body = json.dumps({"source_type": "text", "metadata": {"meeting_date": "2026-01-15"}})
        async with api_client(state, make_current_user()) as ac:
            resp = await ac.post("/jobs", files={"file": b"data"}, data={"body": body})
        assert resp.status_code == 429
        assert resp.json()["limit_type"] == "global_concurrency_cap"

    async def test_persists_raw_bytes_and_submission(self, api_client):
        state = _make_app_state()
        body = json.dumps({"source_type": "text", "metadata": {"meeting_date": "2026-01-15"}})
        async with api_client(state, make_current_user()) as ac:
            resp = await ac.post("/jobs", files={"file": ("input.yaml", b"data", "text/plain")}, data={"body": body})
        assert resp.status_code == 202
        state.blob_store.put.assert_called_once()
        assert state.blob_store.put.call_args[0][0] == "raw/key"
        state.ops.create_job.assert_called_once()
        assert str(state.ops.create_job.call_args[0][5]) == "2026-01-15"

    async def test_rejects_file_without_extension(self, api_client):
        body = json.dumps({"source_type": "text", "metadata": {"meeting_date": "2026-01-15"}})
        async with api_client(_make_app_state(), make_current_user()) as ac:
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
        state.ops.get_job = AsyncMock(return_value=_make_job_row("pending"))
        async with api_client(state, make_current_user()) as ac:
            resp = await ac.get("/jobs/job-1")
        assert resp.status_code == 200
        assert resp.json()["job_id"] == "job-1"
        assert resp.json()["status"] == "pending"

    async def test_deserialises_error_payload(self, api_client):
        error_payload = json.dumps(
            {"stage": "pre_approval", "status": "failed", "reason": "boom", "recoverable": True, "usage": {}}
        )
        row = _make_job_row("failed")
        row["error_payload"] = error_payload
        state = _make_app_state()
        state.ops.get_job = AsyncMock(return_value=row)
        async with api_client(state, make_current_user()) as ac:
            resp = await ac.get("/jobs/job-1")
        assert resp.status_code == 200
        error = resp.json()["error"]
        assert error["stage"] == "pre_approval"
        assert error["status"] == "failed"
        assert error["recoverable"] is True


class TestGetJobResults:
    async def test_requires_auth(self, api_client):
        async with api_client(_make_app_state()) as ac:
            resp = await ac.get("/jobs/job-1/results")
        assert resp.status_code == 401

    async def test_results_not_ready(self, api_client):
        state = _make_app_state()
        state.ops.get_job = AsyncMock(return_value=_make_job_row("pending"))
        async with api_client(state, make_current_user()) as ac:
            resp = await ac.get("/jobs/job-1/results")
        assert resp.status_code == 409

    async def test_returns_result_when_awaiting_review(self, api_client):
        node = make_node()
        result = ExtractionResult(job_id="job-1", nodes=[node], relationships=[])
        state = _make_app_state(results={"job-1": result})
        state.ops.get_job = AsyncMock(return_value=_make_job_row("awaiting_review"))
        async with api_client(state, make_current_user()) as ac:
            resp = await ac.get("/jobs/job-1/results")
        assert resp.status_code == 200

    async def test_falls_back_to_blob_after_restart(self, api_client):
        from datetime import date

        node = make_node()
        result = ExtractionResult(job_id="job-1", nodes=[node], relationships=[])
        state = _make_app_state()
        state.ops.get_job = AsyncMock(return_value=_make_job_row("done", meeting_date=date(2026, 1, 15)))
        state.blob_store.get = AsyncMock(return_value=result.model_dump_json().encode())
        async with api_client(state, make_current_user()) as ac:
            resp = await ac.get("/jobs/job-1/results")
        assert resp.status_code == 200
        assert resp.json()["job_id"] == "job-1"

    async def test_returns_404_when_blob_also_missing(self, api_client):
        from datetime import date

        state = _make_app_state()
        state.ops.get_job = AsyncMock(return_value=_make_job_row("done", meeting_date=date(2026, 1, 15)))
        state.blob_store.get = AsyncMock(return_value=None)
        async with api_client(state, make_current_user()) as ac:
            resp = await ac.get("/jobs/job-1/results")
        assert resp.status_code == 404


class TestApproveJob:
    async def test_requires_auth(self, api_client):
        async with api_client(_make_app_state()) as ac:
            resp = await ac.post("/jobs/job-1/approve", json={"decisions": [{"node_id": "n1", "action": "approve"}]})
        assert resp.status_code == 401

    async def test_requires_reviewer_or_operator(self, api_client):
        state = _make_app_state()
        state.ops.get_job = AsyncMock(return_value=_make_job_row("awaiting_review"))
        async with api_client(state, make_current_user(role=UserRole.VIEWER)) as ac:
            resp = await ac.post("/jobs/job-1/approve", json={"decisions": [{"node_id": "n1", "action": "approve"}]})
        assert resp.status_code == 403

    async def test_not_awaiting_review(self, api_client):
        state = _make_app_state()
        state.ops.get_job = AsyncMock(return_value=_make_job_row("pending"))
        async with api_client(state, make_current_user()) as ac:
            resp = await ac.post("/jobs/job-1/approve", json={"decisions": [{"node_id": "n1", "action": "approve"}]})
        assert resp.status_code == 409

    def _result_nodes(self, state: AppState) -> dict:
        return {str(n.id): n for n in state.results["job-1"].nodes}

    async def test_bulk_rule_approves_above_threshold(self, api_client):
        node = make_node(confidence=0.9, status=NodeStatus.PENDING_REVIEW)
        result = ExtractionResult(job_id="job-1", nodes=[node], relationships=[])
        state = _make_app_state(results={"job-1": result})
        state.ops.get_job = AsyncMock(return_value=_make_job_row("awaiting_review"))
        async with api_client(state, make_current_user()) as ac:
            resp = await ac.post("/jobs/job-1/approve", json={"approve_above_threshold": {"threshold": 0.8}})
        assert resp.status_code == 200
        assert self._result_nodes(state)[str(node.id)].status == NodeStatus.APPROVED

    async def test_bulk_rule_skips_excluded_nodes(self, api_client):
        node = make_node(confidence=0.9, status=NodeStatus.PENDING_REVIEW)
        result = ExtractionResult(job_id="job-1", nodes=[node], relationships=[])
        state = _make_app_state(results={"job-1": result})
        state.ops.get_job = AsyncMock(return_value=_make_job_row("awaiting_review"))
        async with api_client(state, make_current_user()) as ac:
            resp = await ac.post(
                "/jobs/job-1/approve",
                json={"approve_above_threshold": {"threshold": 0.8, "exclude": [str(node.id)]}},
            )
        assert resp.status_code == 200
        assert self._result_nodes(state)[str(node.id)].status == NodeStatus.PENDING_REVIEW

    async def test_individual_decision_approves(self, api_client):
        node = make_node(status=NodeStatus.PENDING_REVIEW)
        result = ExtractionResult(job_id="job-1", nodes=[node], relationships=[])
        state = _make_app_state(results={"job-1": result})
        state.ops.get_job = AsyncMock(return_value=_make_job_row("awaiting_review"))
        async with api_client(state, make_current_user()) as ac:
            resp = await ac.post(
                "/jobs/job-1/approve",
                json={"decisions": [{"node_id": str(node.id), "action": "approve"}]},
            )
        assert resp.status_code == 200
        assert self._result_nodes(state)[str(node.id)].status == NodeStatus.APPROVED

    async def test_individual_decision_rejects(self, api_client):
        node = make_node(status=NodeStatus.PENDING_REVIEW)
        result = ExtractionResult(job_id="job-1", nodes=[node], relationships=[])
        state = _make_app_state(results={"job-1": result})
        state.ops.get_job = AsyncMock(return_value=_make_job_row("awaiting_review"))
        async with api_client(state, make_current_user()) as ac:
            resp = await ac.post(
                "/jobs/job-1/approve",
                json={"decisions": [{"node_id": str(node.id), "action": "reject"}]},
            )
        assert resp.status_code == 200
        assert self._result_nodes(state)[str(node.id)].status == NodeStatus.REJECTED

    async def test_unknown_node_in_decisions_is_ignored(self, api_client):
        node = make_node(status=NodeStatus.PENDING_REVIEW)
        result = ExtractionResult(job_id="job-1", nodes=[node], relationships=[])
        state = _make_app_state(results={"job-1": result})
        state.ops.get_job = AsyncMock(return_value=_make_job_row("awaiting_review"))
        async with api_client(state, make_current_user()) as ac:
            resp = await ac.post(
                "/jobs/job-1/approve",
                json={"decisions": [{"node_id": "00000000-0000-0000-0000-000000000000", "action": "approve"}]},
            )
        assert resp.status_code == 200
        assert self._result_nodes(state)[str(node.id)].status == NodeStatus.PENDING_REVIEW

    async def test_transitions_job_to_writing_before_enqueue(self, api_client):
        node = make_node(status=NodeStatus.PENDING_REVIEW)
        result = ExtractionResult(job_id="job-1", nodes=[node], relationships=[])
        state = _make_app_state(results={"job-1": result})
        state.ops.get_job = AsyncMock(return_value=_make_job_row("awaiting_review"))
        call_order = []
        state.ops.update_job_status = AsyncMock(side_effect=lambda *_: call_order.append("status"))
        state.queue.enqueue = AsyncMock(side_effect=lambda *_: call_order.append("enqueue"))
        async with api_client(state, make_current_user()) as ac:
            resp = await ac.post(
                "/jobs/job-1/approve",
                json={"decisions": [{"node_id": str(node.id), "action": "approve"}]},
            )
        assert resp.status_code == 200
        assert call_order == ["status", "enqueue"]
        state.ops.update_job_status.assert_called_once_with("job-1", JobStatus.WRITING)


class TestRetryJob:
    async def test_requires_auth(self, api_client):
        async with api_client(_make_app_state()) as ac:
            resp = await ac.post("/jobs/job-1/retry")
        assert resp.status_code == 401

    async def test_requires_operator(self, api_client):
        state = _make_app_state()
        state.ops.get_job = AsyncMock(return_value=_make_job_row("failed"))
        async with api_client(state, make_current_user(role=UserRole.REVIEWER)) as ac:
            resp = await ac.post("/jobs/job-1/retry")
        assert resp.status_code == 403

    async def test_not_failed(self, api_client):
        state = _make_app_state()
        state.ops.get_job = AsyncMock(return_value=_make_job_row("pending"))
        async with api_client(state, make_current_user()) as ac:
            resp = await ac.post("/jobs/job-1/retry")
        assert resp.status_code == 409

    async def test_no_stored_input_returns_409(self, api_client):
        state = _make_app_state()
        state.ops.get_job = AsyncMock(return_value=_make_job_row("failed"))
        async with api_client(state, make_current_user()) as ac:
            resp = await ac.post("/jobs/job-1/retry")
        assert resp.status_code == 409
        state.ops.reset_failed_job.assert_not_called()

    async def test_re_enqueues_from_blob(self, api_client):
        state = _make_app_state()
        state.ops.get_job = AsyncMock(
            return_value=_make_job_row(
                "failed",
                raw_blob_key="jobs/2026-01-15/job-1/raw/input.yaml",
                submission='{"source_type":"text","metadata":{"meeting_date":"2026-01-15"}}',
            )
        )
        state.blob_store.get = AsyncMock(return_value=b"file data")
        async with api_client(state, make_current_user()) as ac:
            resp = await ac.post("/jobs/job-1/retry")
        assert resp.status_code == 200
        state.ops.reset_failed_job.assert_called_once_with("job-1")
        state.queue.enqueue.assert_called_once()
