from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from seshat.app.platform.api.dependencies import CurrentUser, get_app_state, require_role
from seshat.app.platform.api.state import AppState
from seshat.app.services.job import JobNotFoundError
from seshat.app.services.publishing import NothingToPublishError, PublishError, PublishGitError
from seshat.core.models.api_responses import PublishResponse
from seshat.core.models.enums import UserRole
from seshat.core.models.publishing import PublishResult

router = APIRouter(tags=["publishing"], dependencies=[Depends(require_role(UserRole.VIEWER))])


@router.post(
    "/jobs/{job_id}/publish",
    response_model=PublishResponse,
    summary="Publish the approved documents of a job to the target git repository",
    responses={
        200: {"description": "Publish completed, or nothing_to_publish=true when already up to date"},
        401: {"description": "Missing or invalid API key"},
        403: {"description": "Operator role required"},
        404: {"description": "Job not found"},
        409: {"description": "Publishing disabled/unconfigured, job not done, or no approved documents"},
        500: {"description": "A git operation failed; see detail"},
    },
)
async def publish_job(
    job_id: str,
    state: Annotated[AppState, Depends(get_app_state)],
    _user: Annotated[CurrentUser, Depends(require_role(UserRole.OPERATOR))],
) -> PublishResponse:
    try:
        result = await state.publishing_service.publish_job(job_id)
    except JobNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found") from exc
    except NothingToPublishError as exc:
        return PublishResponse(nothing_to_publish=True, detail=str(exc))
    except PublishGitError as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc
    except PublishError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc

    return PublishResponse(result=result)


@router.get(
    "/jobs/{job_id}/publish",
    response_model=PublishResult,
    summary="Get the latest publish result for a job",
    responses={
        200: {"description": "Latest publish result returned"},
        401: {"description": "Missing or invalid API key"},
        404: {"description": "The job has never been published"},
    },
)
async def get_publish_result(
    job_id: str,
    state: Annotated[AppState, Depends(get_app_state)],
) -> PublishResult:
    result = await state.publishing_service.get_latest(job_id)
    if result is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No publish result for this job")

    return result
