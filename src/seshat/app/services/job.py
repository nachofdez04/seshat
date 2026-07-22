from __future__ import annotations

import hashlib
import json
import uuid
from datetime import UTC, date, datetime
from typing import TYPE_CHECKING, Any, Literal

import asyncpg
import mlflow

from seshat.app.platform.observability.mlflow_run_logging import (
    log_identification_failures,
    log_resolution_failures,
    set_error_tag,
    set_phase_tag,
)
from seshat.core.models.api_graph import NodeFilter
from seshat.core.models.api_responses import JobActionResponse, JobSubmitResponse
from seshat.core.models.enums import ApprovalMethod, JobStatus, NodeStatus
from seshat.core.models.jobs import JobResponse
from seshat.core.models.nodes import ExtractionResult, IdentificationResult
from seshat.core.models.submission import JobSubmissionRequest
from seshat.core.utils.log import get_logger, set_job_id

if TYPE_CHECKING:
    from seshat.app.pipeline.extraction.orchestrator import ExtractionOrchestrator
    from seshat.app.pipeline.ingestion.orchestrator import IngestionOrchestrator
    from seshat.app.platform.worker.queue import AsyncioTaskQueue
    from seshat.app.repositories.blob_repository import BlobRepository
    from seshat.app.repositories.node_repository import NodeRepository
    from seshat.app.repositories.ops_repository import OpsRepository
    from seshat.core.config.settings import SeshatConfig
    from seshat.core.models.api_jobs import ApproveRequest, BulkApproveRule, NodeDecision
    from seshat.core.models.nodes import KBNode, KBRelationship

logger = get_logger(__name__)


class JobNotFoundError(Exception):
    pass


class JobStateError(Exception):
    pass


class RateLimitExceededError(Exception):
    def __init__(self, limit_type: Literal["per_user_hourly_cap", "global_concurrency_cap"]) -> None:
        self.limit_type = limit_type
        super().__init__(limit_type)


class ContentAlreadyIngestedError(Exception):
    def __init__(self, existing_job_id: str) -> None:
        self.existing_job_id = existing_job_id
        super().__init__(existing_job_id)


class TranscriptNotFoundError(Exception):
    pass


class JobService:
    def __init__(
        self,
        config: SeshatConfig,
        ops: OpsRepository,
        blob: BlobRepository,
        node_repo: NodeRepository,
        extraction: ExtractionOrchestrator,
        ingestion: IngestionOrchestrator,
        queue: AsyncioTaskQueue,
    ) -> None:
        self._config = config
        self._ops = ops
        self._blob = blob
        self._node_repo = node_repo
        self._extraction = extraction
        self._ingestion = ingestion
        self._queue = queue
        self._results: dict[str, ExtractionResult] = {}

    # -- Public lifecycle methods ----------------------------------------------

    async def submit(
        self,
        file_bytes: bytes,
        filename: str | None,
        submission: JobSubmissionRequest,
        user_id: str,
    ) -> JobSubmitResponse:
        idempotency_key = submission.idempotency_key
        if idempotency_key:
            existing = await self._ops.find_job_by_idempotency_key(idempotency_key)
            if existing and existing["status"] != JobStatus.FAILED:
                return JobSubmitResponse(job_id=existing["job_id"])

        if await self._ops.count_recent_jobs_for_user(user_id) >= self._config.api.max_jobs_per_user_per_hour:
            raise RateLimitExceededError("per_user_hourly_cap")

        if await self._ops.count_running_jobs() >= self._config.api.max_concurrent_jobs:
            raise RateLimitExceededError("global_concurrency_cap")

        if not filename or "." not in filename:
            raise ValueError("Uploaded file must have an extension.")

        source_type = submission.source_type
        meeting_date = submission.metadata.meeting_date
        self._ingestion.validate(file_bytes, source_type, meeting_date, filename)

        content_hash = hashlib.sha256(file_bytes).hexdigest()
        existing_job_id = await self._ops.find_job_by_content_hash(content_hash)
        if existing_job_id:
            if not submission.force:
                raise ContentAlreadyIngestedError(existing_job_id)

            logger.warning("Force re-ingest: permanently deleting all nodes for job %s", existing_job_id)
            await self._delete_job_nodes(existing_job_id)

        job_id = str(uuid.uuid4())
        now = datetime.now(UTC)
        ext = filename.rsplit(".", 1)[-1]
        raw_key = self._blob.raw_input_key(meeting_date, job_id, ext)
        submission_json = submission.model_dump_json()
        # Two concurrent requests with the same idempotency key can both pass the
        # check above and race to insert. The loser catches the constraint violation
        # and re-reads the winner's row. Re-raise for any other violation (e.g. a
        # duplicate content_hash inserted between our check and our insert).
        try:
            await self._ops.create_job(
                job_id,
                user_id,
                source_type,
                idempotency_key,
                now,
                meeting_date,
                submission_json,
                raw_key,
                content_hash,
            )
        except asyncpg.UniqueViolationError:
            if idempotency_key:
                existing = await self._ops.find_job_by_idempotency_key(idempotency_key)
                if existing:
                    return JobSubmitResponse(job_id=existing["job_id"])
            raise

        await self._blob.put_by_key(raw_key, file_bytes)
        await self._enqueue(job_id, self._run_pre_approval, file_bytes, submission, user_id)

        return JobSubmitResponse(job_id=job_id)

    async def approve(self, job_id: str, approve_request: ApproveRequest, user_id: str) -> JobActionResponse:
        row = await self._ops.get_job(job_id)
        if not row:
            raise JobNotFoundError(job_id)
        if row["status"] != JobStatus.AWAITING_REVIEW:
            raise JobStateError("Job is not awaiting review")

        result = await self._load_extraction_result(job_id, row)
        if result is None:
            raise JobNotFoundError("Extraction result not found")

        now = datetime.now(UTC)
        nodes = list(result.nodes)

        if approve_request.approve_above_threshold:
            nodes = _apply_bulk_rule(nodes, approve_request.approve_above_threshold, user_id, now)

        if approve_request.decisions:
            nodes = _apply_decisions(nodes, approve_request.decisions, user_id, now)

        # Only update the cache if the pipeline is still holding the pre-approval result.
        # If the entry was already popped by _run_post_approval (auto-mode race), writing
        # it back would leave stale relationships=[] that get_result would serve instead
        # of reading the final blob written after resolution completes.
        if job_id in self._results:
            self._results[job_id] = result._with(nodes=nodes)
        else:
            logger.warning("approve called after _run_post_approval already consumed results for job %s", job_id)

        await self._ops.update_job_status(job_id, JobStatus.RESOLVING)
        await self._enqueue(job_id, self._run_post_approval)

        return JobActionResponse(status="accepted")

    async def retry(self, job_id: str) -> JobActionResponse:
        row = await self._ops.get_job(job_id)
        if not row:
            raise JobNotFoundError(job_id)
        if row["status"] != JobStatus.FAILED:
            raise JobStateError("Only failed jobs can be retried")

        raw_blob_key = row["raw_blob_key"]
        submission_json = row["submission"]
        if not raw_blob_key or not submission_json:
            raise JobStateError("Job has no stored input; re-submit via POST /jobs")

        file_bytes = await self._blob.get_by_key(raw_blob_key)
        submission = JobSubmissionRequest.model_validate_json(submission_json)

        await self._ops.reset_failed_job(job_id)
        await self._enqueue(job_id, self._run_pre_approval, file_bytes, submission)

        return JobActionResponse(status="accepted")

    async def get(self, job_id: str) -> JobResponse | None:
        row = await self._ops.get_job(job_id)
        if not row:
            return None
        return _job_response_from_row(row)

    async def list_jobs(
        self,
        status: JobStatus | None = None,
        source_type: str | None = None,
        meeting_date_from: date | None = None,
        meeting_date_to: date | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[JobResponse]:
        rows = await self._ops.list_jobs(
            status=status,
            source_type=source_type,
            meeting_date_from=meeting_date_from,
            meeting_date_to=meeting_date_to,
            limit=min(limit, 200),
            offset=offset,
        )
        return [_job_response_from_row(row) for row in rows]

    async def get_result(self, job_id: str) -> ExtractionResult | None:
        row = await self._ops.get_job(job_id)
        if not row:
            raise JobNotFoundError(job_id)
        if row["status"] not in (JobStatus.AWAITING_REVIEW, JobStatus.DONE):
            raise JobStateError("Results not yet available")

        return await self._load_extraction_result(job_id, row)

    async def get_transcript_excerpt(self, job_id: str, char_start: int, char_end: int) -> str:
        if char_start >= char_end:
            raise ValueError(f"char_start ({char_start}) must be less than char_end ({char_end})")

        row = await self._ops.get_job(job_id)
        if not row:
            raise JobNotFoundError(job_id)

        meeting_date = row["meeting_date"]
        if meeting_date is None:
            raise TranscriptNotFoundError(job_id)

        blob = await self._blob.get_raw_transcript(meeting_date, job_id)
        if blob is None:
            raise TranscriptNotFoundError(job_id)

        text = blob.decode("utf-8", errors="replace")
        return text[char_start:char_end]

    async def recover_stranded(self) -> None:
        stranded = await self._ops.get_stranded_writing_jobs()
        for job_id in stranded:
            await self._ops.fail_job(job_id, JobStatus.WRITING, "Server crash during write", recoverable=True)
            logger.warning("Startup recovery: marked stranded job %s as FAILED", job_id)

    # -- Private execution methods (enqueued as callbacks) --------------------

    async def _run_pre_approval(
        self,
        job_id: str,
        file_bytes: bytes,
        submission: JobSubmissionRequest,
        user_id: str | None = None,
    ) -> IdentificationResult | None:
        with mlflow.start_run(
            run_name=job_id,
            tags={"job_id": job_id, "phase": "starting", "source": "pipeline"},
        ) as run:
            await self._ops.set_job_mlflow_run_id(job_id, run.info.run_id)
            set_phase_tag("ingestion")

            try:
                set_job_id(job_id)
                await self._ops.update_job_status(job_id, JobStatus.TRANSCRIBING)

                if submission.source_type == "audio":
                    doc = await self._ingestion.ingest_audio(
                        file_bytes,
                        submission.metadata.meeting_date,
                        job_id,
                        submission.metadata,
                    )
                else:
                    doc = await self._ingestion.ingest_text(
                        file_bytes,
                        submission.metadata.meeting_date,
                        job_id,
                        "input.yaml",
                    )

                set_phase_tag("identification")
                await self._ops.update_job_status(job_id, JobStatus.IDENTIFYING)
                config_override = submission.overrides.extraction if submission.overrides else None
                identification_result = await self._extraction.run_identification(
                    doc, job_id, user_id=user_id, config_override=config_override
                )

                if self._effective_auto_mode(submission):
                    identification_result = _apply_auto_mode(identification_result, datetime.now(UTC))

                await self._store_result(job_id, identification_result, [])

                node_counts = _count_by_status(identification_result.nodes)
                mlflow.log_params({"job_id": job_id, "source_type": submission.source_type})
                mlflow.log_metrics({f"nodes.{k}": v for k, v in node_counts.items()})
                log_identification_failures(identification_result.failed_concept_types)

                await self._ops.update_job_status(job_id, JobStatus.AWAITING_REVIEW)

                has_pending = any(n.status == NodeStatus.PENDING_REVIEW for n in identification_result.nodes)
                if not has_pending:
                    await self._enqueue(job_id, self._run_post_approval)

                return identification_result

            except Exception as exc:
                set_error_tag(exc)
                logger.exception("Job failed (pre-approval): %s", exc)
                await self._ops.fail_job(job_id, "pre_approval", str(exc), recoverable=True)
                return None

    async def _run_post_approval(self, job_id: str) -> None:
        row = await self._ops.get_job(job_id)
        if not row:
            await self._ops.fail_job(job_id, "post_approval", f"Job {job_id} not found", recoverable=False)
            return

        existing_run_id = row.get("mlflow_run_id")
        if existing_run_id:
            active_run = mlflow.start_run(run_id=existing_run_id)
        else:
            active_run = mlflow.start_run(run_name=job_id, tags={"job_id": job_id, "source": "pipeline"})

        with active_run:
            set_phase_tag("resolution")

            try:
                set_job_id(job_id)
                result = await self._load_extraction_result(job_id, row)
                if result is None:
                    raise RuntimeError(f"Extraction result for job {job_id} not found in memory or blob store")
                self._results.pop(job_id, None)

                approved = [n for n in result.nodes if n.status == NodeStatus.APPROVED]
                await self._ops.update_job_status(job_id, JobStatus.RESOLVING)
                resolution_result = await self._extraction.run_resolution(job_id, approved=approved)

                # update the resolved relationships in the result before writing to the database
                result = result._with(relationships=resolution_result.relationships)
                await self._ops.update_job_status(job_id, JobStatus.WRITING)
                written_nodes, written_rels = await self._node_repo.write_batch(result)

                await self._blob.put_curated_extraction(row["meeting_date"], job_id, result.model_dump_json().encode())
                mlflow.log_metrics({"nodes.written": written_nodes, "relationships.written": written_rels})
                log_resolution_failures(resolution_result.failed_sources)

                await self._ops.update_job_status(job_id, JobStatus.DONE)
                set_phase_tag("finalized")

                logger.info("Job done: %d node(s) and %d relationship(s) written", written_nodes, written_rels)

            except Exception as exc:
                set_error_tag(exc)
                logger.exception("Job failed (post-approval): %s", exc)
                await self._ops.fail_job(job_id, "post_approval", str(exc), recoverable=True)

    # -- Private helpers -------------------------------------------------------

    async def _enqueue(self, job_id: str, fn: Any, *args: Any, **kwargs: Any) -> None:
        # Convenience wrapper so callers don't repeat job_id: every pipeline method
        # takes job_id as its first positional argument after self.
        await self._queue.enqueue(job_id, fn, job_id, *args, **kwargs)

    async def _load_extraction_result(self, job_id: str, row: dict) -> ExtractionResult | None:
        result = self._results.get(job_id)
        if result:
            return result

        raw = await self._blob.get_curated_extraction(row["meeting_date"], job_id)
        return ExtractionResult.model_validate_json(raw) if raw else None

    def _effective_auto_mode(self, submission: JobSubmissionRequest) -> bool:
        if submission.auto_mode:
            return True

        overrides = submission.overrides
        if overrides and overrides.extraction and overrides.extraction.auto_mode:
            return True

        return self._extraction._config.auto_mode

    async def _delete_job_nodes(self, job_id: str) -> None:
        nodes = await self._node_repo.paginated_query(NodeFilter(job_id=job_id))
        for node in nodes:
            await self._node_repo.delete_node(node.id, cascade=True)

    async def _store_result(
        self,
        job_id: str,
        identification_result: IdentificationResult,
        relationships: list[KBRelationship],
    ) -> ExtractionResult:
        result = ExtractionResult(
            job_id=job_id,
            nodes=identification_result.nodes,
            relationships=relationships,
            confidence_breakdowns={str(k): v for k, v in identification_result.confidence_breakdowns.items()},
        )
        self._results[job_id] = result

        job_row = await self._ops.get_job(job_id)
        meeting_date = job_row["meeting_date"] if job_row else None
        if meeting_date is not None:
            await self._blob.put_curated_extraction(meeting_date, job_id, result.model_dump_json().encode())
        else:
            logger.warning("Job has no meeting_date; skipping `extraction.json` blob write")

        return result


# -- Pure transforms -----------------------------------------------------------


def _apply_auto_mode(identification_result: IdentificationResult, now: datetime) -> IdentificationResult:
    updated_nodes = []
    for node in identification_result.nodes:
        if node.status == NodeStatus.PENDING_REVIEW:
            metadata = node.metadata._with(
                approval_method=ApprovalMethod.AUTO,
                approved_at=now,
            )
            node = node._with(status=NodeStatus.APPROVED, metadata=metadata)
        updated_nodes.append(node)
    return identification_result._with(nodes=updated_nodes)


def _apply_bulk_rule(nodes: list[KBNode], rule: BulkApproveRule, user_id: str, now: datetime) -> list[KBNode]:
    exclude = set(rule.exclude or [])
    result = []
    for node in nodes:
        node_id = str(node.id)
        if node.status == NodeStatus.PENDING_REVIEW and node.confidence >= rule.threshold and node_id not in exclude:
            metadata = node.metadata._with(approval_method=ApprovalMethod.BULK, approved_by=user_id, approved_at=now)
            node = node._with(status=NodeStatus.APPROVED, metadata=metadata)
        result.append(node)
    return result


def _apply_decisions(nodes: list[KBNode], decisions: list[NodeDecision], user_id: str, now: datetime) -> list[KBNode]:
    node_map = {str(n.id): n for n in nodes}
    for decision in decisions:
        node = node_map.get(decision.node_id)
        if not node:
            continue

        if decision.action == "approve":
            if node.status == NodeStatus.APPROVED and not decision.edited_content:
                continue

            node_kwargs: dict[str, Any] = {}
            meta_kwargs: dict[str, Any] = {
                "approval_method": ApprovalMethod.INDIVIDUAL,
                "approved_by": user_id,
                "approved_at": now,
            }

            if edited_content := decision.edited_content:
                node_kwargs |= {"title": edited_content.title, "description": edited_content.description}
                meta_kwargs |= {"corrected_by": user_id, "corrected_at": now}

            node_kwargs |= {"status": NodeStatus.APPROVED, "metadata": node.metadata._with(**meta_kwargs)}
            node_map[decision.node_id] = node._with(**node_kwargs)

        elif decision.action == "reject":
            node_map[decision.node_id] = node._with(status=NodeStatus.REJECTED)

    return list(node_map.values())


def _count_by_status(nodes: list) -> dict[str, int]:
    counts: dict[str, int] = {}
    for n in nodes:
        counts[n.status.value] = counts.get(n.status.value, 0) + 1
    return counts


def _job_response_from_row(row: Any) -> JobResponse:
    error = json.loads(row["error_payload"]) if row["error_payload"] else None

    submission = json.loads(row.get("submission", "{}"))
    extraction_overrides = submission.get("overrides", {}).get("extraction", {})
    threshold = extraction_overrides.get("confidence_threshold")

    model_kwargs = dict(row) | {"error": error, "confidence_threshold": threshold}
    return JobResponse.model_validate(model_kwargs)
