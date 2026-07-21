from __future__ import annotations

from datetime import date
from typing import Annotated

from fastapi import APIRouter, Depends, Form, HTTPException, Query, UploadFile, status
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from seshat.app.pipeline.ingestion.audio_validator import (
    AudioValidationError,
    FileTooLargeError,
    UnsupportedFormatError,
)
from seshat.app.pipeline.ingestion.text_validator import TextValidationError
from seshat.app.platform.api.dependencies import CurrentUser, get_app_state, require_role
from seshat.app.platform.api.state import AppState
from seshat.app.services.job import (
    ContentAlreadyIngestedError,
    JobNotFoundError,
    JobStateError,
    RateLimitExceededError,
    TranscriptNotFoundError,
)
from seshat.core.models.api_jobs import ApproveRequest, RateLimitError
from seshat.core.models.api_responses import JobActionResponse, JobSubmitResponse, TranscriptExcerptResponse
from seshat.core.models.enums import JobStatus, UserRole
from seshat.core.models.jobs import JobResponse
from seshat.core.models.nodes import ExtractionResult
from seshat.core.models.submission import JobSubmissionRequest

router = APIRouter(prefix="/jobs", tags=["jobs"], dependencies=[Depends(require_role(UserRole.VIEWER))])


@router.get(
    "",
    response_model=list[JobResponse],
    summary="List jobs",
    responses={
        200: {"description": "Jobs returned"},
        401: {"description": "Missing or invalid API key"},
    },
)
async def list_jobs(
    state: Annotated[AppState, Depends(get_app_state)],
    job_status: JobStatus | None = None,
    source_type: str | None = None,
    meeting_date_from: date | None = None,
    meeting_date_to: date | None = None,
    limit: Annotated[int, Query(ge=0)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[JobResponse]:
    return await state.job_service.list_jobs(
        status=job_status,
        source_type=source_type,
        meeting_date_from=meeting_date_from,
        meeting_date_to=meeting_date_to,
        limit=limit,
        offset=offset,
    )


@router.post(
    "",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=None,
    summary="Submit a new ingestion job",
    responses={
        202: {"description": "Job accepted and queued"},
        400: {"description": "Invalid file (missing extension)"},
        401: {"description": "Missing or invalid API key"},
        403: {"description": "Insufficient role or overrides require operator"},
        409: {"description": "Content already ingested (use force=true to re-ingest)"},
        413: {"description": "Uploaded file exceeds the maximum allowed size"},
        415: {"description": "Unsupported or mismatched audio format"},
        422: {"description": "Malformed request body or invalid input file"},
        429: {"description": "Rate limit exceeded (per-user or global concurrency cap)"},
    },
)
async def submit_job(
    state: Annotated[AppState, Depends(get_app_state)],
    user: Annotated[CurrentUser, Depends(require_role(UserRole.REVIEWER))],
    file: Annotated[UploadFile, ...],
    body: Annotated[str, Form()],
) -> JobSubmitResponse | JSONResponse:
    # body is a raw Form string (multipart upload), so FastAPI never sees a typed
    # model and the global RequestValidationError handler won't fire — validate manually.
    try:
        submission = JobSubmissionRequest.model_validate_json(body)
    except ValidationError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=exc.errors()) from exc

    if submission.overrides is not None and not user.role.is_at_least(UserRole.OPERATOR):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Config overrides require operator role")

    if submission.force and not user.role.is_at_least(UserRole.ADMIN):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Force re-ingest deletes existing nodes — admin role required"
        )

    file_bytes = await file.read()

    try:
        return await state.job_service.submit(file_bytes, file.filename, submission, user.user_id)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except FileTooLargeError as exc:
        raise HTTPException(status_code=status.HTTP_413_CONTENT_TOO_LARGE, detail=str(exc)) from exc
    except UnsupportedFormatError as exc:
        raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail=str(exc)) from exc
    except (AudioValidationError, TextValidationError) as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)) from exc
    except RateLimitExceededError as exc:
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content=RateLimitError(limit_type=exc.limit_type).model_dump(),
        )
    except ContentAlreadyIngestedError as exc:
        return JSONResponse(
            status_code=status.HTTP_409_CONFLICT,
            content={"detail": "Content already ingested", "existing_job_id": exc.existing_job_id},
        )


@router.get(
    "/{job_id}",
    response_model=JobResponse,
    summary="Get job status",
    responses={
        200: {"description": "Job found"},
        401: {"description": "Missing or invalid API key"},
        404: {"description": "Job not found"},
    },
)
async def get_job(
    job_id: str,
    state: Annotated[AppState, Depends(get_app_state)],
) -> JobResponse:
    result = await state.job_service.get(job_id)
    if not result:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
    return result


@router.get(
    "/{job_id}/results",
    summary="Get extraction results for a job",
    responses={
        200: {"description": "Extraction result returned"},
        401: {"description": "Missing or invalid API key"},
        404: {"description": "Job not found or result not in storage"},
        409: {"description": "Results not yet available (job not in awaiting_review or done state)"},
    },
)
async def get_job_results(
    job_id: str,
    state: Annotated[AppState, Depends(get_app_state)],
) -> ExtractionResult:
    try:
        result = await state.job_service.get_result(job_id)
    except JobNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except JobStateError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc

    if result is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Extraction result not found in storage")

    return result


@router.get(
    "/{job_id}/transcript/excerpt",
    response_model=TranscriptExcerptResponse,
    summary="Retrieve a text slice from the job's transcript blob",
    responses={
        200: {"description": "Transcript excerpt returned"},
        401: {"description": "Missing or invalid API key"},
        404: {"description": "Job or transcript blob not found"},
        422: {"description": "char_start or char_end out of range"},
    },
)
async def get_transcript_excerpt(
    job_id: str,
    state: Annotated[AppState, Depends(get_app_state)],
    char_start: Annotated[int, Query(ge=0)] = 0,
    char_end: Annotated[int, Query(ge=1)] = 500,
) -> TranscriptExcerptResponse:
    try:
        text = await state.job_service.get_transcript_excerpt(job_id, char_start, char_end)
    except JobNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
    except TranscriptNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Transcript not found — job may not have completed ingestion yet",
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc))

    return TranscriptExcerptResponse(text=text, char_start=char_start, char_end=char_end)


@router.post(
    "/{job_id}/approve",
    summary="Submit review decisions and trigger writing",
    responses={
        200: {"description": "Decisions accepted; writing stage enqueued"},
        401: {"description": "Missing or invalid API key"},
        403: {"description": "Insufficient role (reviewer or operator required)"},
        404: {"description": "Job or extraction result not found"},
        409: {"description": "Job is not in awaiting_review state"},
    },
)
async def approve_job(
    job_id: str,
    approve_request: ApproveRequest,
    state: Annotated[AppState, Depends(get_app_state)],
    user: Annotated[CurrentUser, Depends(require_role(UserRole.REVIEWER))],
) -> JobActionResponse:
    try:
        return await state.job_service.approve(job_id, approve_request, user.user_id)
    except JobNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except JobStateError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc


@router.post(
    "/{job_id}/retry",
    summary="Retry a failed job",
    responses={
        200: {"description": "Job reset and re-queued"},
        401: {"description": "Missing or invalid API key"},
        403: {"description": "Operator role required"},
        404: {"description": "Job not found"},
        409: {"description": "Job is not failed, or has no stored input to retry from"},
    },
)
async def retry_job(
    job_id: str,
    state: Annotated[AppState, Depends(get_app_state)],
    _user: Annotated[CurrentUser, Depends(require_role(UserRole.OPERATOR))],
) -> JobActionResponse:
    try:
        return await state.job_service.retry(job_id)
    except JobNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except JobStateError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
