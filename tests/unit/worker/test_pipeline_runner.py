from __future__ import annotations

from datetime import date
from unittest.mock import AsyncMock, MagicMock

from seshat.models.enums import JobStatus, NodeStatus
from seshat.models.nodes import ExtractionResult, IdentificationResult, ResolutionResult
from seshat.worker.pipeline_runner import PipelineRunner
from tests.helpers import make_node


def _make_runner(
    nodes: list | None = None,
    *,
    source_type: str = "text",
) -> tuple[PipelineRunner, MagicMock, MagicMock, MagicMock, MagicMock, dict]:
    nodes = nodes or [make_node()]

    ident_result = IdentificationResult(
        job_id="job-1",
        nodes=nodes,
        confidence_breakdowns={},
    )
    resol_result = ResolutionResult(job_id="job-1", relationships=[])

    ingestion = MagicMock()
    ingestion.ingest_text = AsyncMock(return_value=MagicMock())
    ingestion.ingest_audio = AsyncMock(return_value=MagicMock())

    extraction = MagicMock()
    extraction.run_identification = AsyncMock(return_value=ident_result)
    extraction.run_resolution = AsyncMock(return_value=resol_result)

    writing = MagicMock()
    writing.write = AsyncMock(return_value=len([n for n in nodes if n.status == NodeStatus.APPROVED]))

    ops = MagicMock()
    ops.update_job_status = AsyncMock()
    ops.fail_job = AsyncMock()
    ops.get_job = AsyncMock(return_value=None)

    blob_store = MagicMock()
    blob_store.put = AsyncMock()
    blob_store.curated_extraction_key = MagicMock(return_value="curated/key")

    result_store: dict = {}
    runner = PipelineRunner(ingestion, extraction, writing, ops, result_store, blob_store)
    return runner, ingestion, extraction, writing, ops, result_store


def _make_submission(source_type: str = "text") -> MagicMock:
    sub = MagicMock()
    sub.source_type = source_type
    return sub


class TestPipelineRunnerPreApproval:
    async def test_text_job_sets_statuses_and_parks_result(self):
        runner, _, _, _, ops, _ = _make_runner()
        ident = await runner._run_pre_approval("job-1", b"data", _make_submission())

        assert ident is not None
        assert "job-1" in runner._pending
        ops.update_job_status.assert_any_await("job-1", JobStatus.TRANSCRIBING)
        ops.update_job_status.assert_any_await("job-1", JobStatus.EXTRACTING)
        ops.update_job_status.assert_any_await("job-1", JobStatus.AWAITING_REVIEW)

    async def test_audio_job_calls_ingest_audio(self):
        runner, ingestion, _, _, _, _ = _make_runner()
        await runner._run_pre_approval("job-1", b"audio", _make_submission("audio"))
        ingestion.ingest_audio.assert_called_once()
        ingestion.ingest_text.assert_not_called()

    async def test_failure_calls_fail_job_and_returns_none(self):
        runner, ingestion, _, _, ops, _ = _make_runner()
        ingestion.ingest_text.side_effect = RuntimeError("network error")

        result = await runner._run_pre_approval("job-1", b"data", _make_submission())

        assert result is None
        ops.fail_job.assert_called_once()
        assert "job-1" not in runner._pending


class TestPipelineRunnerPostApproval:
    async def test_resolves_writes_and_sets_done(self):
        runner, _, extraction, writing, ops, result_store = _make_runner()
        runner._pending["job-1"] = await extraction.run_identification(MagicMock(), "job-1")

        await runner.run_post_approval("job-1")

        extraction.run_resolution.assert_called_once()
        writing.write.assert_called_once()
        ops.update_job_status.assert_any_await("job-1", JobStatus.WRITING)
        ops.update_job_status.assert_any_await("job-1", JobStatus.DONE)
        assert "job-1" in result_store
        assert isinstance(result_store["job-1"], ExtractionResult)

    async def test_passes_approved_nodes_to_resolution(self):
        approved = make_node()
        rejected = make_node("rejected", status=NodeStatus.REJECTED)
        runner, _, extraction, _, _, _ = _make_runner(nodes=[approved, rejected])
        runner._pending["job-1"] = IdentificationResult(
            job_id="job-1", nodes=[approved, rejected], confidence_breakdowns={}
        )

        await runner.run_post_approval("job-1")

        _, kwargs = extraction.run_resolution.call_args
        assert kwargs["approved"] == [approved]

    async def test_writes_extraction_json_before_writing_stage(self):
        meeting_date = date(2026, 1, 1)
        runner, _, extraction, writing, ops, _ = _make_runner()
        ops.get_job = AsyncMock(return_value={"meeting_date": meeting_date})
        runner._blob_store.curated_extraction_key = MagicMock(return_value="curated/2026-01-01/job-1/extraction.json")
        runner._pending["job-1"] = await extraction.run_identification(MagicMock(), "job-1")

        put_order = []
        runner._blob_store.put = AsyncMock(side_effect=lambda *_: put_order.append("put"))
        writing.write = AsyncMock(side_effect=lambda *_: put_order.append("write"))

        await runner.run_post_approval("job-1")

        assert put_order == ["put", "write"]
        runner._blob_store.put.assert_called_once()
        put_key = runner._blob_store.put.call_args[0][0]
        assert put_key == "curated/2026-01-01/job-1/extraction.json"

    async def test_failure_calls_fail_job(self):
        runner, _, extraction, _, ops, _ = _make_runner()
        runner._pending["job-1"] = IdentificationResult(job_id="job-1", nodes=[], confidence_breakdowns={})
        extraction.run_resolution.side_effect = RuntimeError("llm timeout")

        await runner.run_post_approval("job-1")

        ops.fail_job.assert_called_once()
        call_kwargs = ops.fail_job.call_args
        assert "post_approval" in call_kwargs[0]


class TestPipelineRunnerRun:
    async def test_run_skips_post_approval_when_pending_review(self):
        node = make_node(status=NodeStatus.PENDING_REVIEW)
        runner, _, extraction, writing, _, _ = _make_runner(nodes=[node])

        await runner.run("job-1", b"data", _make_submission())

        extraction.run_resolution.assert_not_called()
        writing.write.assert_not_called()

    async def test_run_calls_post_approval_when_all_approved(self):
        runner, _, extraction, writing, _, _ = _make_runner()

        await runner.run("job-1", b"data", _make_submission())

        extraction.run_resolution.assert_called_once()
        writing.write.assert_called_once()
