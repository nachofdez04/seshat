from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status

from seshat.app.platform.api.dependencies import CurrentUser, get_app_state, require_role
from seshat.app.platform.api.state import AppState
from seshat.app.services.document import DocumentNotFoundError, DocumentRevisionConflictError
from seshat.app.services.job import JobNotFoundError, JobStateError, TranscriptNotFoundError
from seshat.core.models.api_responses import GeneratedDocumentMeta
from seshat.core.models.documents import DocumentReviewRequest, GeneratedDocument
from seshat.core.models.enums import UserRole

router = APIRouter(tags=["documents"], dependencies=[Depends(require_role(UserRole.VIEWER))])


@router.post(
    "/jobs/{job_id}/documents",
    response_model=GeneratedDocument,
    summary="Generate (or regenerate) the meeting summary for a done job",
    responses={
        200: {"description": "Document generated and stored"},
        401: {"description": "Missing or invalid API key"},
        403: {"description": "Operator role required"},
        404: {"description": "Job or transcript blob not found"},
        409: {"description": "Job is not done"},
    },
)
async def generate_document(
    job_id: str,
    state: Annotated[AppState, Depends(get_app_state)],
    _user: Annotated[CurrentUser, Depends(require_role(UserRole.OPERATOR))],
) -> GeneratedDocument:
    try:
        return await state.document_service.generate_for_job(job_id)
    except JobNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found") from exc
    except TranscriptNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Transcript not found") from exc
    except JobStateError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc


@router.get(
    "/jobs/{job_id}/documents",
    response_model=list[GeneratedDocumentMeta],
    summary="List generated documents for a job (metadata only)",
    responses={
        200: {"description": "Document metadata returned"},
        401: {"description": "Missing or invalid API key"},
        404: {"description": "Job not found"},
    },
)
async def list_documents(
    job_id: str,
    state: Annotated[AppState, Depends(get_app_state)],
) -> list[GeneratedDocumentMeta]:
    try:
        documents = await state.document_service.list_for_job(job_id)
    except JobNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found") from exc

    return [GeneratedDocumentMeta.model_validate(document) for document in documents]


@router.post(
    "/documents/{document_id}/review",
    response_model=GeneratedDocument,
    summary="Approve, approve-with-edits, or reject a generated document",
    responses={
        200: {"description": "Review decision applied; updated document returned"},
        401: {"description": "Missing or invalid API key"},
        403: {"description": "Reviewer role required"},
        404: {"description": "Document not found"},
        409: {"description": "The document content or validation state changed after it was fetched"},
    },
)
async def review_document(
    document_id: UUID,
    request: DocumentReviewRequest,
    state: Annotated[AppState, Depends(get_app_state)],
    user: Annotated[CurrentUser, Depends(require_role(UserRole.REVIEWER))],
) -> GeneratedDocument:
    try:
        return await state.document_service.review(document_id, request, user.user_id)
    except DocumentNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found") from exc
    except DocumentRevisionConflictError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc


@router.get(
    "/documents/{document_id}",
    response_model=GeneratedDocument,
    summary="Get a generated document including its content",
    responses={
        200: {"description": "Document returned"},
        401: {"description": "Missing or invalid API key"},
        404: {"description": "Document not found"},
    },
)
async def get_document(
    document_id: UUID,
    state: Annotated[AppState, Depends(get_app_state)],
) -> GeneratedDocument:
    document = await state.document_service.get(document_id)
    if document is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    return document
