from __future__ import annotations

import asyncio
from datetime import UTC, date, datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

import pytest

from seshat.app.platform.worker.queue import AsyncioTaskQueue
from seshat.app.services.job import (
    ContentAlreadyIngestedError,
    JobNotFoundError,
    JobService,
    JobStateError,
    RateLimitExceededError,
    TranscriptNotFoundError,
    _apply_auto_mode,
    _apply_bulk_rule,
    _apply_decisions,
)
from seshat.core.models.api_jobs import BulkApproveRule, KBNodeEdit, NodeDecision
from seshat.core.models.enums import ApprovalMethod, JobStatus, NodeStatus
from seshat.core.models.nodes import ExtractionResult, IdentificationResult, ResolutionResult
from tests.helpers import make_node


def _make_service(
    nodes: list | None = None,
) -> tuple[JobService, MagicMock, MagicMock, MagicMock, MagicMock, MagicMock]:
    nodes = nodes or [make_node()]

    ident_result = IdentificationResult(job_id="job-1", nodes=nodes, confidence_breakdowns={})
    resol_result = ResolutionResult(job_id="job-1", relationships=[])

    ingestion = MagicMock()
    ingestion.ingest_text = AsyncMock(return_value=MagicMock())
    ingestion.ingest_audio = AsyncMock(return_value=MagicMock())

    extraction = MagicMock()
    extraction.run_identification = AsyncMock(return_value=ident_result)
    extraction.run_resolution = AsyncMock(return_value=resol_result)
    extraction._config = MagicMock()
    extraction._config.auto_mode = False

    node_repo = MagicMock()
    node_repo.write_batch = AsyncMock(return_value=(len([n for n in nodes if n.status == NodeStatus.APPROVED]), 0))
    node_repo.paginated_query = AsyncMock(return_value=[])
    node_repo.delete_node = AsyncMock()

    ops = MagicMock()
    ops.update_job_status = AsyncMock()
    ops.fail_job = AsyncMock()
    ops.get_job = AsyncMock(return_value=None)
    ops.count_recent_jobs_for_user = AsyncMock(return_value=0)
    ops.count_running_jobs = AsyncMock(return_value=0)
    ops.find_job_by_idempotency_key = AsyncMock(return_value=None)
    ops.find_job_by_content_hash = AsyncMock(return_value=None)
    ops.create_job = AsyncMock()
    ops.reset_failed_job = AsyncMock()
    ops.get_stranded_writing_jobs = AsyncMock(return_value=[])
    ops.set_job_mlflow_run_id = AsyncMock()

    blob = MagicMock()
    blob.put_by_key = AsyncMock()
    blob.get_by_key = AsyncMock(return_value=None)
    blob.put_curated_extraction = AsyncMock()
    blob.get_curated_extraction = AsyncMock(return_value=None)
    blob.raw_input_key = MagicMock(return_value="raw/key")

    config = MagicMock()
    config.api.max_jobs_per_user_per_hour = 10
    config.api.max_concurrent_jobs = 5

    queue = MagicMock()
    queue.enqueue = AsyncMock()

    svc = JobService(config, ops, blob, node_repo, extraction, ingestion, queue)
    return svc, ingestion, extraction, node_repo, ops, blob


def _make_submission(source_type: str = "text") -> MagicMock:
    sub = MagicMock()
    sub.source_type = source_type
    sub.auto_mode = False
    sub.overrides = None
    sub.idempotency_key = None
    sub.force = False
    sub.metadata.meeting_date = date(2026, 1, 15)
    sub.model_dump_json = MagicMock(return_value="{}")
    return sub


class TestPreApproval:
    async def test_text_job_sets_statuses_and_parks_result(self):
        svc, _, _, _, ops, _ = _make_service()
        ident = await svc._run_pre_approval("job-1", b"data", _make_submission())

        assert ident is not None
        assert "job-1" in svc._results
        ops.update_job_status.assert_any_await("job-1", JobStatus.TRANSCRIBING)
        ops.update_job_status.assert_any_await("job-1", JobStatus.IDENTIFYING)
        ops.update_job_status.assert_any_await("job-1", JobStatus.AWAITING_REVIEW)

    async def test_audio_job_calls_ingest_audio(self):
        svc, ingestion, _, _, _, _ = _make_service()
        await svc._run_pre_approval("job-1", b"audio", _make_submission("audio"))
        ingestion.ingest_audio.assert_called_once()
        ingestion.ingest_text.assert_not_called()

    async def test_failure_calls_fail_job_and_returns_none(self):
        svc, ingestion, _, _, ops, _ = _make_service()
        ingestion.ingest_text.side_effect = RuntimeError("network error")

        result = await svc._run_pre_approval("job-1", b"data", _make_submission())

        assert result is None
        ops.fail_job.assert_called_once()
        assert "job-1" not in svc._results


_JOB_ROW = {"meeting_date": date(2026, 1, 1), "status": JobStatus.WRITING}


class TestPostApproval:
    async def test_resolves_writes_and_sets_done(self):
        node = make_node(status=NodeStatus.APPROVED)
        svc, _, extraction, node_repo, ops, _ = _make_service(nodes=[node])
        ops.get_job = AsyncMock(return_value=_JOB_ROW)
        svc._results["job-1"] = ExtractionResult(job_id="job-1", nodes=[node], relationships=[])

        await svc._run_post_approval("job-1")

        extraction.run_resolution.assert_called_once()
        node_repo.write_batch.assert_called_once()
        ops.update_job_status.assert_any_await("job-1", JobStatus.WRITING)
        ops.update_job_status.assert_any_await("job-1", JobStatus.DONE)

    async def test_passes_approved_nodes_to_resolution(self):
        approved = make_node(status=NodeStatus.APPROVED)
        rejected = make_node("rejected", status=NodeStatus.REJECTED)
        svc, _, extraction, _, ops, _ = _make_service(nodes=[approved, rejected])
        ops.get_job = AsyncMock(return_value=_JOB_ROW)
        svc._results["job-1"] = ExtractionResult(job_id="job-1", nodes=[approved, rejected], relationships=[])

        await svc._run_post_approval("job-1")

        _, kwargs = extraction.run_resolution.call_args
        assert kwargs["approved"] == [approved]

    async def test_writes_curated_extraction_blob(self):
        meeting_date = date(2026, 1, 1)
        svc, _, _, _, ops, blob = _make_service()
        ops.get_job = AsyncMock(return_value={"meeting_date": meeting_date, "status": JobStatus.WRITING})
        svc._results["job-1"] = ExtractionResult(job_id="job-1", nodes=[], relationships=[])

        await svc._run_post_approval("job-1")

        blob.put_curated_extraction.assert_called_once()
        call_args = blob.put_curated_extraction.call_args
        assert call_args.args[0] == meeting_date
        assert call_args.args[1] == "job-1"

    async def test_failure_calls_fail_job(self):
        svc, _, extraction, _, ops, _ = _make_service()
        ops.get_job = AsyncMock(return_value=_JOB_ROW)
        svc._results["job-1"] = ExtractionResult(job_id="job-1", nodes=[], relationships=[])
        extraction.run_resolution.side_effect = RuntimeError("llm timeout")

        await svc._run_post_approval("job-1")

        ops.fail_job.assert_called_once()
        call_kwargs = ops.fail_job.call_args
        assert "post_approval" in call_kwargs[0]


def _make_service_real_queue(nodes=None):
    """Like _make_service but with a real AsyncioTaskQueue so enqueued tasks actually run."""
    svc, ingestion, extraction, node_repo, ops, blob = _make_service(nodes=nodes)
    svc._queue = AsyncioTaskQueue()
    return svc, ingestion, extraction, node_repo, ops, blob


class TestAutoMode:
    async def test_auto_mode_field_promotes_pending_and_fires_post_approval(self):
        node = make_node(status=NodeStatus.PENDING_REVIEW)
        svc, _, extraction, node_repo, ops, _ = _make_service_real_queue(nodes=[node])
        ops.get_job = AsyncMock(return_value=_JOB_ROW)

        sub = _make_submission()
        sub.auto_mode = True

        await svc._run_pre_approval("job-1", b"data", sub)
        await asyncio.gather(*svc._queue._tasks.values())

        extraction.run_resolution.assert_called_once()
        node_repo.write_batch.assert_called_once()

    async def test_auto_mode_via_extraction_override_also_works(self):
        node = make_node(status=NodeStatus.PENDING_REVIEW)
        svc, _, extraction, node_repo, ops, _ = _make_service_real_queue(nodes=[node])
        ops.get_job = AsyncMock(return_value=_JOB_ROW)

        sub = _make_submission()
        sub.overrides = MagicMock()
        sub.overrides.extraction = MagicMock()
        sub.overrides.extraction.auto_mode = True

        await svc._run_pre_approval("job-1", b"data", sub)
        await asyncio.gather(*svc._queue._tasks.values())

        extraction.run_resolution.assert_called_once()
        node_repo.write_batch.assert_called_once()

    async def test_pending_nodes_stay_without_auto_mode(self):
        node = make_node(status=NodeStatus.PENDING_REVIEW)
        svc, _, extraction, node_repo, _, _ = _make_service(nodes=[node])

        await svc._run_pre_approval("job-1", b"data", _make_submission())

        extraction.run_resolution.assert_not_called()
        node_repo.write_batch.assert_not_called()


class TestApprove:
    async def test_not_found_raises(self):
        svc, *_ = _make_service()
        svc._ops.get_job = AsyncMock(return_value=None)

        with pytest.raises(JobNotFoundError):
            await svc.approve("job-1", MagicMock(), "alice")

    async def test_wrong_state_raises(self):
        svc, *_ = _make_service()
        svc._ops.get_job = AsyncMock(return_value={"status": "pending"})

        with pytest.raises(JobStateError):
            await svc.approve("job-1", MagicMock(), "alice")

    async def test_missing_result_raises(self):
        svc, *_ = _make_service()
        svc._ops.get_job = AsyncMock(
            return_value={"status": JobStatus.AWAITING_REVIEW, "meeting_date": date(2026, 1, 1)}
        )

        with pytest.raises(JobNotFoundError):
            await svc.approve("job-1", MagicMock(), "alice")


class TestRecoverStranded:
    async def test_marks_stranded_jobs_failed(self):
        svc, *_ = _make_service()
        svc._ops.get_stranded_writing_jobs = AsyncMock(return_value=["job-a", "job-b"])

        await svc.recover_stranded()

        assert svc._ops.fail_job.call_count == 2
        svc._ops.fail_job.assert_any_call("job-a", JobStatus.WRITING, "Server crash during write", recoverable=True)
        svc._ops.fail_job.assert_any_call("job-b", JobStatus.WRITING, "Server crash during write", recoverable=True)


_NOW = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
_NODE_UUID = UUID("00000000-0000-0000-0000-000000000001")


class TestApplyAutoMode:
    def test_pending_nodes_become_approved(self):
        node = make_node(status=NodeStatus.PENDING_REVIEW)
        result = IdentificationResult(job_id="j", nodes=[node], confidence_breakdowns={})

        updated = _apply_auto_mode(result, _NOW)

        assert all(n.status == NodeStatus.APPROVED for n in updated.nodes)

    def test_already_approved_nodes_unchanged(self):
        node = make_node(status=NodeStatus.APPROVED)
        result = IdentificationResult(job_id="j", nodes=[node], confidence_breakdowns={})

        updated = _apply_auto_mode(result, _NOW)

        assert updated.nodes[0].status == NodeStatus.APPROVED
        assert updated.nodes[0].metadata.approval_method == node.metadata.approval_method

    def test_sets_approval_method_to_auto(self):
        node = make_node(status=NodeStatus.PENDING_REVIEW)
        result = IdentificationResult(job_id="j", nodes=[node], confidence_breakdowns={})

        updated = _apply_auto_mode(result, _NOW)

        assert updated.nodes[0].metadata.approval_method == ApprovalMethod.AUTO


class TestApplyBulkRule:
    def test_approves_nodes_above_threshold(self):
        node = make_node(confidence=0.9, status=NodeStatus.PENDING_REVIEW)
        rule = BulkApproveRule(threshold=0.8)

        result = _apply_bulk_rule([node], rule, "alice", _NOW)

        assert result[0].status == NodeStatus.APPROVED

    def test_skips_nodes_below_threshold(self):
        node = make_node(confidence=0.5, status=NodeStatus.PENDING_REVIEW)
        rule = BulkApproveRule(threshold=0.8)

        result = _apply_bulk_rule([node], rule, "alice", _NOW)

        assert result[0].status == NodeStatus.PENDING_REVIEW

    def test_excludes_specified_node_ids(self):
        node = make_node(confidence=0.9, status=NodeStatus.PENDING_REVIEW)
        rule = BulkApproveRule(threshold=0.7, exclude=[str(node.id)])

        result = _apply_bulk_rule([node], rule, "alice", _NOW)

        assert result[0].status == NodeStatus.PENDING_REVIEW

    def test_sets_approval_method_to_bulk(self):
        node = make_node(confidence=0.9, status=NodeStatus.PENDING_REVIEW)
        rule = BulkApproveRule(threshold=0.7)

        result = _apply_bulk_rule([node], rule, "alice", _NOW)

        assert result[0].metadata.approval_method == ApprovalMethod.BULK
        assert result[0].metadata.approved_by == "alice"


class TestApplyDecisions:
    def test_approve_does_not_override_bulk_approval(self):
        node = make_node(confidence=0.9, status=NodeStatus.PENDING_REVIEW)
        rule = BulkApproveRule(threshold=0.7)
        bulk_approved = _apply_bulk_rule([node], rule, "alice", _NOW)

        decisions = [NodeDecision(node_id=str(node.id), action="approve")]
        result = _apply_decisions(bulk_approved, decisions, "alice", _NOW)

        assert result[0].status == NodeStatus.APPROVED
        assert result[0].metadata.approval_method == ApprovalMethod.BULK

    def test_reject_overrides_bulk_approval(self):
        node = make_node(confidence=0.9, status=NodeStatus.PENDING_REVIEW)
        rule = BulkApproveRule(threshold=0.7)
        bulk_approved = _apply_bulk_rule([node], rule, "alice", _NOW)

        decisions = [NodeDecision(node_id=str(node.id), action="reject")]
        result = _apply_decisions(bulk_approved, decisions, "alice", _NOW)

        assert result[0].status == NodeStatus.REJECTED

    def test_edit_on_bulk_approved_node_updates_content(self):
        node = make_node(confidence=0.9, status=NodeStatus.PENDING_REVIEW)
        rule = BulkApproveRule(threshold=0.7)
        bulk_approved = _apply_bulk_rule([node], rule, "alice", _NOW)

        edited = KBNodeEdit(title="Fixed Title", description="Fixed Desc")
        decisions = [NodeDecision(node_id=str(node.id), action="approve", edited_content=edited)]
        result = _apply_decisions(bulk_approved, decisions, "alice", _NOW)

        assert result[0].status == NodeStatus.APPROVED
        assert result[0].title == "Fixed Title"
        assert result[0].metadata.corrected_by == "alice"

    def test_approve_action_sets_status(self):
        node = make_node(status=NodeStatus.PENDING_REVIEW)
        decisions = [NodeDecision(node_id=str(node.id), action="approve")]

        result = _apply_decisions([node], decisions, "alice", _NOW)

        assert result[0].status == NodeStatus.APPROVED

    def test_reject_action_sets_status(self):
        node = make_node(status=NodeStatus.PENDING_REVIEW)
        decisions = [NodeDecision(node_id=str(node.id), action="reject")]

        result = _apply_decisions([node], decisions, "alice", _NOW)

        assert result[0].status == NodeStatus.REJECTED

    def test_unknown_node_id_is_silently_skipped(self):
        node = make_node(status=NodeStatus.PENDING_REVIEW)
        decisions = [NodeDecision(node_id="nonexistent-id", action="approve")]

        result = _apply_decisions([node], decisions, "alice", _NOW)

        assert result[0].status == NodeStatus.PENDING_REVIEW

    def test_edited_content_updates_title_and_description(self):
        node = make_node(status=NodeStatus.PENDING_REVIEW)
        edited = KBNodeEdit(title="New Title", description="New Desc")
        decisions = [NodeDecision(node_id=str(node.id), action="approve", edited_content=edited)]

        result = _apply_decisions([node], decisions, "alice", _NOW)

        assert result[0].title == "New Title"
        assert result[0].description == "New Desc"
        assert result[0].metadata.corrected_by == "alice"

    def test_approve_sets_individual_method(self):
        node = make_node(status=NodeStatus.PENDING_REVIEW)
        decisions = [NodeDecision(node_id=str(node.id), action="approve")]

        result = _apply_decisions([node], decisions, "alice", _NOW)

        assert result[0].metadata.approval_method == ApprovalMethod.INDIVIDUAL
        assert result[0].metadata.approved_by == "alice"


class TestGetResult:
    async def test_raises_not_found_when_job_missing(self):
        svc, *_ = _make_service()
        svc._ops.get_job = AsyncMock(return_value=None)

        with pytest.raises(JobNotFoundError):
            await svc.get_result("job-1")

    async def test_raises_state_error_for_non_review_status(self):
        svc, *_ = _make_service()
        svc._ops.get_job = AsyncMock(return_value={"status": JobStatus.PENDING})

        with pytest.raises(JobStateError):
            await svc.get_result("job-1")

    async def test_returns_in_memory_result_when_present(self):
        svc, *_ = _make_service()
        node = make_node()
        result = ExtractionResult(job_id="job-1", nodes=[node], relationships=[])
        svc._results["job-1"] = result
        svc._ops.get_job = AsyncMock(return_value={"status": JobStatus.AWAITING_REVIEW})

        fetched = await svc.get_result("job-1")

        assert fetched is result

    async def test_returns_none_when_no_meeting_date(self):
        svc, *_ = _make_service()
        svc._ops.get_job = AsyncMock(return_value={"status": JobStatus.DONE, "meeting_date": None})

        result = await svc.get_result("job-1")

        assert result is None


class TestSubmitContentHash:
    async def test_raises_content_already_ingested_when_hash_matches(self):
        svc, *_ = _make_service()
        svc._ops.find_job_by_content_hash = AsyncMock(return_value="existing-job")
        svc._ops.find_job_by_idempotency_key = AsyncMock(return_value=None)
        svc._ops.count_recent_jobs_for_user = AsyncMock(return_value=0)
        svc._ops.count_running_jobs = AsyncMock(return_value=0)

        sub = _make_submission()
        sub.force = False

        with pytest.raises(ContentAlreadyIngestedError) as exc_info:
            await svc.submit(b"data", "file.txt", sub, "alice")

        assert exc_info.value.existing_job_id == "existing-job"

    async def test_idempotency_key_returns_existing_job_without_resubmit(self):
        svc, *_ = _make_service()
        svc._ops.find_job_by_idempotency_key = AsyncMock(
            return_value={"job_id": "existing-job", "status": JobStatus.DONE}
        )

        sub = _make_submission()
        sub.idempotency_key = "key-abc"

        response = await svc.submit(b"data", "file.txt", sub, "alice")

        assert response.job_id == "existing-job"
        svc._ops.create_job.assert_not_called()

    async def test_idempotency_bypasses_failed_job(self):
        svc, *_ = _make_service()
        svc._ops.find_job_by_idempotency_key = AsyncMock(
            return_value={"job_id": "failed-job", "status": JobStatus.FAILED}
        )
        svc._ops.find_job_by_content_hash = AsyncMock(return_value=None)
        svc._ops.count_recent_jobs_for_user = AsyncMock(return_value=0)
        svc._ops.count_running_jobs = AsyncMock(return_value=0)
        svc._blob.put_by_key = AsyncMock()
        svc._blob.raw_input_key = MagicMock(return_value="raw/key.txt")

        sub = _make_submission()
        sub.idempotency_key = "key-abc"

        # A FAILED idempotency match should not short-circuit — a new job is created.
        response = await svc.submit(b"data", "file.txt", sub, "alice")

        svc._ops.create_job.assert_called_once()
        assert response.job_id != "failed-job"


class TestSubmitRateLimit:
    async def test_per_user_cap_raises(self):
        svc, *_ = _make_service()
        svc._ops.count_recent_jobs_for_user = AsyncMock(return_value=10)  # equals max

        with pytest.raises(RateLimitExceededError) as exc_info:
            await svc.submit(b"data", "file.txt", _make_submission(), "alice")

        assert exc_info.value.limit_type == "per_user_hourly_cap"
        svc._ops.create_job.assert_not_called()

    async def test_global_cap_raises(self):
        svc, *_ = _make_service()
        svc._ops.count_running_jobs = AsyncMock(return_value=5)  # equals max

        with pytest.raises(RateLimitExceededError) as exc_info:
            await svc.submit(b"data", "file.txt", _make_submission(), "alice")

        assert exc_info.value.limit_type == "global_concurrency_cap"
        svc._ops.create_job.assert_not_called()

    async def test_missing_extension_raises_value_error(self):
        svc, *_ = _make_service()

        with pytest.raises(ValueError, match="extension"):
            await svc.submit(b"data", "noextension", _make_submission(), "alice")

        svc._ops.create_job.assert_not_called()

    async def test_force_resubmit_deletes_non_approved_nodes(self):
        svc, *_ = _make_service()
        from seshat.core.models.enums import NodeStatus

        pending = make_node(status=NodeStatus.PENDING_REVIEW)
        svc._ops.find_job_by_content_hash = AsyncMock(return_value="old-job")
        svc._node_repo.paginated_query = AsyncMock(return_value=[pending])
        svc._blob.raw_input_key = MagicMock(return_value="raw/key.txt")
        svc._blob.put_by_key = AsyncMock()

        sub = _make_submission()
        sub.force = True

        await svc.submit(b"data", "file.txt", sub, "alice")

        svc._node_repo.delete_node.assert_called_once_with(pending.id, cascade=True)


class TestRetry:
    async def test_not_found_raises(self):
        svc, *_ = _make_service()
        svc._ops.get_job = AsyncMock(return_value=None)

        with pytest.raises(JobNotFoundError):
            await svc.retry("job-1")

    async def test_non_failed_job_raises_state_error(self):
        svc, *_ = _make_service()
        svc._ops.get_job = AsyncMock(return_value={"status": JobStatus.DONE})

        with pytest.raises(JobStateError):
            await svc.retry("job-1")

    async def test_no_stored_input_raises_state_error(self):
        svc, *_ = _make_service()
        svc._ops.get_job = AsyncMock(
            return_value={"status": JobStatus.FAILED, "raw_blob_key": None, "submission": None}
        )

        with pytest.raises(JobStateError, match="no stored input"):
            await svc.retry("job-1")

    async def test_valid_failed_job_re_enqueues(self):
        import json

        svc, *_ = _make_service()
        submission_json = json.dumps({"source_type": "text", "metadata": {"meeting_date": "2026-01-15"}})
        svc._ops.get_job = AsyncMock(
            return_value={
                "status": JobStatus.FAILED,
                "raw_blob_key": "raw/key.txt",
                "submission": submission_json,
            }
        )
        svc._blob.get_by_key = AsyncMock(return_value=b"data")

        await svc.retry("job-1")

        svc._ops.reset_failed_job.assert_called_once_with("job-1")
        svc._queue.enqueue.assert_called_once()


class TestListJobs:
    async def test_returns_empty_list(self):
        svc, *_ = _make_service()
        svc._ops.list_jobs = AsyncMock(return_value=[])

        result = await svc.list_jobs()

        assert result == []

    async def test_caps_limit_at_200(self):
        svc, *_ = _make_service()
        svc._ops.list_jobs = AsyncMock(return_value=[])

        await svc.list_jobs(limit=500)

        call_args = svc._ops.list_jobs.call_args
        assert call_args.kwargs.get("limit", call_args.args[1] if len(call_args.args) > 1 else None) <= 200

    async def test_forwards_source_type_to_ops(self):
        svc, *_ = _make_service()
        svc._ops.list_jobs = AsyncMock(return_value=[])

        await svc.list_jobs(source_type="audio")

        call_kwargs = svc._ops.list_jobs.call_args.kwargs
        assert call_kwargs.get("source_type") == "audio"

    async def test_forwards_date_range_to_ops(self):
        svc, *_ = _make_service()
        svc._ops.list_jobs = AsyncMock(return_value=[])

        from_date = date(2026, 1, 1)
        to_date = date(2026, 6, 30)
        await svc.list_jobs(meeting_date_from=from_date, meeting_date_to=to_date)

        call_kwargs = svc._ops.list_jobs.call_args.kwargs
        assert call_kwargs.get("meeting_date_from") == from_date
        assert call_kwargs.get("meeting_date_to") == to_date


class TestGetTranscriptExcerpt:
    async def test_returns_slice(self):
        svc, _, _, _, ops, blob = _make_service()
        ops.get_job = AsyncMock(return_value={"job_id": "job-1", "meeting_date": date(2025, 1, 1), "status": "done"})
        blob.get_raw_transcript = AsyncMock(return_value=b"Hello world")

        result = await svc.get_transcript_excerpt("job-1", 0, 5)

        assert result == "Hello"

    async def test_raises_job_not_found(self):
        svc, _, _, _, ops, _ = _make_service()
        ops.get_job = AsyncMock(return_value=None)

        with pytest.raises(JobNotFoundError):
            await svc.get_transcript_excerpt("job-1", 0, 5)

    async def test_raises_transcript_not_found_when_meeting_date_missing(self):
        svc, _, _, _, ops, _ = _make_service()
        ops.get_job = AsyncMock(return_value={"job_id": "job-1", "meeting_date": None, "status": "done"})

        with pytest.raises(TranscriptNotFoundError):
            await svc.get_transcript_excerpt("job-1", 0, 5)

    async def test_raises_transcript_not_found_when_blob_missing(self):
        svc, _, _, _, ops, blob = _make_service()
        ops.get_job = AsyncMock(return_value={"job_id": "job-1", "meeting_date": date(2025, 1, 1), "status": "done"})
        blob.get_raw_transcript = AsyncMock(return_value=None)

        with pytest.raises(TranscriptNotFoundError):
            await svc.get_transcript_excerpt("job-1", 0, 5)

    async def test_raises_value_error_when_start_not_less_than_end(self):
        svc, _, _, _, _, _ = _make_service()

        with pytest.raises(ValueError, match="char_start"):
            await svc.get_transcript_excerpt("job-1", 10, 5)
