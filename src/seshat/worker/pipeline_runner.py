from __future__ import annotations

from typing import TYPE_CHECKING

from seshat.models.enums import JobStatus, NodeStatus
from seshat.models.nodes import ExtractionResult
from seshat.utils.log import get_logger, set_job_id

if TYPE_CHECKING:
    from seshat.blob_store.s3_store import S3BlobStore
    from seshat.models.nodes import IdentificationResult, KBRelationship
    from seshat.models.submission import JobSubmissionRequest
    from seshat.ops.ledger import OpsLedger
    from seshat.pipeline.extraction.orchestrator import ExtractionOrchestrator
    from seshat.pipeline.ingestion.orchestrator import IngestionOrchestrator
    from seshat.worker.writing_stage import WritingStage

logger = get_logger(__name__)


class PipelineRunner:
    def __init__(
        self,
        ingestion_orchestrator: IngestionOrchestrator,
        extraction_orchestrator: ExtractionOrchestrator,
        writing_stage: WritingStage,
        ops_ledger: OpsLedger,
        result_store: dict[str, ExtractionResult],
        blob_store: S3BlobStore,
    ) -> None:
        self._ingestion = ingestion_orchestrator
        self._extraction = extraction_orchestrator
        self._writing = writing_stage
        self._ops = ops_ledger
        self._results = result_store
        self._blob_store = blob_store
        self._pending: dict[str, IdentificationResult] = {}

    async def run(self, job_id: str, file_bytes: bytes, submission: JobSubmissionRequest) -> None:
        """Convenience method: run pre-approval and, if no review needed, post-approval."""
        identification_result = await self._run_pre_approval(job_id, file_bytes, submission)
        if identification_result is None:
            return

        # In auto_mode all nodes are APPROVED or REJECTED by assign_status, so this is always
        # False and post-approval fires immediately without a human review step.
        has_pending = any(n.status == NodeStatus.PENDING_REVIEW for n in identification_result.nodes)
        if not has_pending:
            await self.run_post_approval(job_id)

    async def _run_pre_approval(
        self,
        job_id: str,
        file_bytes: bytes,
        submission: JobSubmissionRequest,
    ) -> IdentificationResult | None:
        """Ingest and identify. Stores result; sets AWAITING_REVIEW. Returns None on failure."""
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
                doc = await self._ingestion.ingest_text(file_bytes, "input.yaml", job_id)

            await self._ops.update_job_status(job_id, JobStatus.EXTRACTING)
            identification_result = await self._extraction.run_identification(doc, job_id)

            self._pending[job_id] = identification_result
            # Pre-populate results so GET /jobs/{id}/results works during AWAITING_REVIEW.
            # run_post_approval overwrites this with the complete result (including relationships).
            await self._store_result(job_id, identification_result, [])

            await self._ops.update_job_status(job_id, JobStatus.AWAITING_REVIEW)
            return identification_result

        except Exception as exc:
            logger.exception("Job %s failed (pre-approval): %s", job_id, exc)
            # MVP: all failures marked recoverable; operator reads error_payload to triage.
            # Revisit when a retry queue is added — some failures (e.g. bad input) are permanent.
            await self._ops.fail_job(job_id, "pre_approval", str(exc), recoverable=True)
            return None

    async def run_post_approval(self, job_id: str) -> None:
        """Resolve, write, and mark DONE. Expects _run_pre_approval to have completed."""
        try:
            identification_result = self._pending.pop(job_id)
            set_job_id(job_id)

            approved = [n for n in identification_result.nodes if n.status == NodeStatus.APPROVED]
            resol = await self._extraction.run_resolution(job_id, approved=approved)

            result = await self._store_result(job_id, identification_result, resol.relationships)

            await self._ops.update_job_status(job_id, JobStatus.WRITING)
            written = await self._writing.write(result)

            await self._ops.update_job_status(job_id, JobStatus.DONE)
            logger.info("Job %s done: %d nodes written", job_id, written)

        except Exception as exc:
            logger.exception("Job %s failed (post-approval): %s", job_id, exc)
            # MVP: all failures marked recoverable; see pre_approval comment above.
            await self._ops.fail_job(job_id, "post_approval", str(exc), recoverable=True)

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
            await self._blob_store.put(
                self._blob_store.curated_extraction_key(meeting_date, job_id),
                result.model_dump_json().encode(),
            )
        else:
            logger.warning("Job %s has no meeting_date; skipping extraction.json blob write", job_id)

        return result
