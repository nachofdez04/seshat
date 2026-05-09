from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import pytest

from seshat.blob_store.s3_store import S3BlobStore

if TYPE_CHECKING:
    from seshat.config.settings import SeshatConfig


@pytest.fixture
def store(minimal_config: SeshatConfig) -> S3BlobStore:
    return S3BlobStore(minimal_config.blob_store)


def _make_mock_session():
    """Return a mock aioboto3 session whose .client() is an async context manager."""
    fake_client = MagicMock()
    ctx = AsyncMock()
    ctx.__aenter__.return_value = fake_client
    ctx.__aexit__.return_value = None
    session = MagicMock()
    session.client.return_value = ctx
    return session, ctx, fake_client


class TestS3BlobStoreLifecycle:
    async def test_connect_sets_client(self, store: S3BlobStore):
        session, ctx, fake_client = _make_mock_session()
        store._session = session

        await store.connect()

        assert store._client is fake_client
        assert store._client_ctx is ctx
        ctx.__aenter__.assert_awaited_once()

    async def test_close_clears_client(self, store: S3BlobStore):
        session, ctx, _ = _make_mock_session()
        store._session = session
        await store.connect()

        await store.close()

        assert store._client is None
        assert store._client_ctx is None
        ctx.__aexit__.assert_awaited_once_with(None, None, None)

    async def test_close_is_idempotent(self, store: S3BlobStore):
        """close() on a not-yet-connected store must not raise."""
        await store.close()

    async def test_client_property_raises_before_connect(self, store: S3BlobStore):
        with pytest.raises(RuntimeError, match="connect"):
            _ = store.client
