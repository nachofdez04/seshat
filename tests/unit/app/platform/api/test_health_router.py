from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from seshat.app.platform.api.state import AppState
from seshat.core.models.api_responses import HealthStatus
from seshat.core.models.enums import UserRole
from tests.unit.app.platform.api.conftest import make_current_user


def _make_app_state(*, pg_ok: bool = True, mlflow_ok: bool = True, blob_ok: bool = True) -> AppState:
    health_service = MagicMock()
    health_service.check_postgres = AsyncMock(return_value=HealthStatus.OK if pg_ok else HealthStatus.ERROR)
    health_service.check_mlflow = AsyncMock(return_value=HealthStatus.OK if mlflow_ok else HealthStatus.ERROR)
    health_service.check_blob = AsyncMock(return_value=HealthStatus.OK if blob_ok else HealthStatus.ERROR)

    return AppState(
        config=MagicMock(),
        admin_service=MagicMock(),
        health_service=health_service,
        graph_service=MagicMock(),
        job_service=MagicMock(),
        document_service=MagicMock(),
        publishing_service=MagicMock(),
    )


class TestHealthEndpoint:
    async def test_all_ok_returns_200(self, api_client):
        async with api_client(_make_app_state(), make_current_user(role=UserRole.VIEWER)) as ac:
            resp = await ac.get("/health/components")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    async def test_postgres_down_returns_503(self, api_client):
        async with api_client(_make_app_state(pg_ok=False), make_current_user(role=UserRole.VIEWER)) as ac:
            resp = await ac.get("/health/components")
        assert resp.status_code == 503
        assert resp.json()["components"]["postgres"] == "error"

    async def test_external_service_down_returns_503(self, api_client):
        async with api_client(_make_app_state(mlflow_ok=False), make_current_user(role=UserRole.VIEWER)) as ac:
            resp = await ac.get("/health/components")
        assert resp.status_code == 503
        assert resp.json()["status"] == "degraded"
