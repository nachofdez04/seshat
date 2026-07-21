from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from seshat.app.services.health import HealthService, _check_http
from seshat.core.config.settings import BlobStoreConfig, ObservabilityConfig
from seshat.core.models.enums import HealthStatus


def _make_service(*, blob_config: BlobStoreConfig | None = None) -> HealthService:
    ops = MagicMock()
    ops.is_alive = AsyncMock(return_value=True)
    blob = MagicMock()
    blob._store = MagicMock()
    blob._store.client = MagicMock()
    blob._store.client.head_bucket = AsyncMock()
    return HealthService(
        ops_repo=ops,
        blob_repo=blob,
        blob_config=blob_config or BlobStoreConfig(),
        observability_config=ObservabilityConfig(),
    )


class TestHealthServiceCheckPostgres:
    async def test_returns_ok_when_alive(self):
        svc = _make_service()
        svc._ops.is_alive = AsyncMock(return_value=True)
        assert await svc.check_postgres() == HealthStatus.OK

    async def test_returns_error_when_not_alive(self):
        svc = _make_service()
        svc._ops.is_alive = AsyncMock(return_value=False)
        assert await svc.check_postgres() == HealthStatus.ERROR


class TestHealthServiceCheckMlflow:
    async def test_returns_ok_on_successful_http(self):
        svc = _make_service()
        with patch("seshat.app.services.health._check_http", new=AsyncMock(return_value=HealthStatus.OK)):
            result = await svc.check_mlflow()
        assert result == HealthStatus.OK

    async def test_passes_correct_url(self):
        svc = _make_service()
        expected_url = f"{ObservabilityConfig().mlflow_tracking_uri}/health"
        with patch("seshat.app.services.health._check_http", new=AsyncMock(return_value=HealthStatus.OK)) as mock_http:
            await svc.check_mlflow()
        mock_http.assert_called_once_with(expected_url)


class TestHealthServiceCheckBlob:
    async def test_localstack_endpoint_uses_http_check(self):
        config = BlobStoreConfig(endpoint_url="http://localhost:4566")
        svc = _make_service(blob_config=config)
        with patch("seshat.app.services.health._check_http", new=AsyncMock(return_value=HealthStatus.OK)) as mock_http:
            result = await svc.check_blob()
        assert result == HealthStatus.OK
        mock_http.assert_called_once_with("http://localhost:4566/_localstack/health")

    async def test_aws_s3_head_bucket_ok(self):
        svc = _make_service(blob_config=BlobStoreConfig())
        svc._blob._store.client.head_bucket = AsyncMock()
        result = await svc.check_blob()
        assert result == HealthStatus.OK

    async def test_aws_s3_head_bucket_raises_returns_error(self):
        svc = _make_service(blob_config=BlobStoreConfig())
        svc._blob._store.client.head_bucket = AsyncMock(side_effect=Exception("no bucket"))
        result = await svc.check_blob()
        assert result == HealthStatus.ERROR


class TestCheckHttp:
    async def test_returns_ok_on_success(self):
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(return_value=mock_response)
        with patch("seshat.app.services.health.httpx.AsyncClient", return_value=mock_client):
            result = await _check_http("http://example.com/health")
        assert result == HealthStatus.OK

    async def test_returns_error_on_http_error(self):
        import httpx

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(side_effect=httpx.ConnectError("refused"))
        with patch("seshat.app.services.health.httpx.AsyncClient", return_value=mock_client):
            result = await _check_http("http://example.com/health")
        assert result == HealthStatus.ERROR

    async def test_returns_error_on_5xx_response(self):
        import httpx

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock(
            side_effect=httpx.HTTPStatusError("500", request=MagicMock(), response=MagicMock())
        )
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(return_value=mock_response)
        with patch("seshat.app.services.health.httpx.AsyncClient", return_value=mock_client):
            result = await _check_http("http://example.com/health")
        assert result == HealthStatus.ERROR
