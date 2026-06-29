from __future__ import annotations

from enum import StrEnum, auto
from typing import Annotated

import httpx
from fastapi import APIRouter, Depends, Response
from pydantic import BaseModel

from seshat.api.dependencies import get_app_state
from seshat.api.state import AppState


class HealthStatus(StrEnum):
    OK = auto()
    DEGRADED = auto()
    ERROR = auto()


class HealthResponse(BaseModel):
    status: HealthStatus
    components: dict[str, HealthStatus]


router = APIRouter(prefix="/health", tags=["health"])


@router.get("", response_model=HealthResponse)
async def health(state: Annotated[AppState, Depends(get_app_state)], response: Response) -> HealthResponse:
    config = state.config

    postgres = await _check_postgres(state)
    mlflow = await _check_http(f"{config.observability.mlflow_tracking_uri}/health")

    localstack_url = config.blob_store.endpoint_url or "http://localstack:4566"
    localstack = await _check_http(f"{localstack_url}/_localstack/health")

    components = {"postgres": postgres, "mlflow": mlflow, "localstack": localstack}
    overall = HealthStatus.OK if all(v == HealthStatus.OK for v in components.values()) else HealthStatus.DEGRADED

    if overall != HealthStatus.OK:
        response.status_code = 503

    return HealthResponse(status=overall, components=components)


async def _check_postgres(state: AppState) -> HealthStatus:
    try:
        await state.ops._pool.fetchval("SELECT 1")
        return HealthStatus.OK
    except Exception:
        return HealthStatus.ERROR


async def _check_http(url: str) -> HealthStatus:
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            await client.get(url)
        return HealthStatus.OK
    except httpx.HTTPError:
        return HealthStatus.ERROR
