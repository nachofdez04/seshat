import pytest

from seshat.blob_store.s3_store import S3BlobStore
from seshat.config.settings import BlobStoreConfig
from tests.integration.conftest import LOCALSTACK_REGION, LOCALSTACK_TEST_BUCKET, SKIP_IF_NO_LOCALSTACK

pytestmark = [pytest.mark.integration, SKIP_IF_NO_LOCALSTACK]


@pytest.fixture
async def store(localstack_s3_url):
    config = BlobStoreConfig(
        bucket=LOCALSTACK_TEST_BUCKET,
        region=LOCALSTACK_REGION,
        endpoint_url=localstack_s3_url,
    )
    s = S3BlobStore(config)
    await s.connect()
    yield s
    await s.close()


class TestS3BlobStoreGet:
    async def test_put_then_get(self, store):
        key = "test/blob.txt"
        data = b"hello world"
        await store.put(key, data)
        result = await store.get(key)
        assert result == data


class TestS3BlobStoreExists:
    async def test_true(self, store):
        key = "test/exists.txt"
        await store.put(key, b"data")
        assert await store.exists(key) is True

    async def test_false(self, store):
        assert await store.exists("test/does-not-exist.txt") is False
