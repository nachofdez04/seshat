from collections import defaultdict

from pydantic import BaseModel, Field

from seshat.models.enums import CallType, JobStatus


class UsageRecord(BaseModel):
    call_type: CallType
    units: float = Field(description="Number of tokens, seconds, or other units consumed.")


class ErrorPayload(BaseModel):
    stage: JobStatus
    reason: str = Field(description="Human-readable error message.")
    recoverable: bool = Field(description="Whether the job can be retried without changes.")
    usage: dict[JobStatus, list[UsageRecord]] = Field(
        default_factory=lambda: defaultdict[JobStatus, list[UsageRecord]](list)
    )


class JobResponse(BaseModel):
    job_id: str
    status: JobStatus
    idempotency_key: str | None = Field(
        default=None, description="Client-supplied idempotency key, echoed from the submission request."
    )
    current_stage: JobStatus | None = None
    stage_progress: str | None = Field(
        default=None, description="Human-readable progress string for the current stage."
    )
    elapsed_seconds: float
    error: ErrorPayload | None = None
    mlflow_run_id: str | None = None
