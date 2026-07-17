import pytest

from tests.integration.conftest import SKIP_IF_NO_LOCALSTACK

# loop_scope="module" required: blob_store fixture uses a module-scoped aioboto3 session
pytestmark = [pytest.mark.asyncio(loop_scope="module"), pytest.mark.integration, SKIP_IF_NO_LOCALSTACK]


class TestS3BlobStoreGet:
    async def test_put_then_get(self, blob_store):
        key = "test/blob.txt"
        data = b"hello world"
        await blob_store.put(key, data)
        result = await blob_store.get(key)
        assert result == data


class TestS3BlobStoreExists:
    async def test_true(self, blob_store):
        key = "test/exists.txt"
        await blob_store.put(key, b"data")
        assert await blob_store.exists(key) is True

    async def test_false(self, blob_store):
        assert await blob_store.exists("test/does-not-exist.txt") is False
