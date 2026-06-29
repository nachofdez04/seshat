from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from seshat.api.state import AppState
from seshat.models.enums import UserRole
from tests.unit.api.conftest import make_current_user


def _make_app_state(*, pg_ok: bool = True) -> AppState:
    pool = MagicMock()
    pool.fetchval = AsyncMock(return_value=1 if pg_ok else None)
    if not pg_ok:
        pool.fetchval.side_effect = RuntimeError("connection refused")

    ops = MagicMock()
    ops._pool = pool

    config = MagicMock()
    config.observability.mlflow_tracking_uri = "http://mlflow:5000"
    config.blob_store.endpoint_url = "http://localstack:4566"

    return AppState(
        ops=ops,
        kb_store=MagicMock(),
        config=config,
        queue=MagicMock(),
        results={},
        runner=MagicMock(),
        manual_ingestion=MagicMock(),
        blob_store=MagicMock(),
    )


class TestHealthEndpoint:
    async def test_all_ok_returns_200(self, api_client):
        with patch("seshat.api.routers.health._check_http", new=AsyncMock(return_value="ok")):
            async with api_client(_make_app_state(), make_current_user(role=UserRole.VIEWER)) as ac:
                resp = await ac.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    async def test_postgres_down_returns_503(self, api_client):
        with patch("seshat.api.routers.health._check_http", new=AsyncMock(return_value="ok")):
            async with api_client(_make_app_state(pg_ok=False), make_current_user(role=UserRole.VIEWER)) as ac:
                resp = await ac.get("/health")
        assert resp.status_code == 503
        assert resp.json()["components"]["postgres"] == "error"

    async def test_external_service_down_returns_503(self, api_client):
        async def _failing(url: str):
            from seshat.api.routers.health import HealthStatus

            return HealthStatus.ERROR

        with patch("seshat.api.routers.health._check_http", new=_failing):
            async with api_client(_make_app_state(), make_current_user(role=UserRole.VIEWER)) as ac:
                resp = await ac.get("/health")
        assert resp.status_code == 503
        assert resp.json()["status"] == "degraded"
