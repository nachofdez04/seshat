from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import aioboto3
from botocore.exceptions import ClientError

from seshat.utils.retry import async_retry

if TYPE_CHECKING:
    from aiobotocore.session import ClientCreatorContext
    from types_aiobotocore_s3.client import S3Client

    from seshat.config.settings import BlobStoreConfig


# ClientError.Code is either a numeric HTTP status or an S3 semantic code depending on the error path.
_NON_RETRYABLE_S3 = frozenset(
    {
        "400",
        "403",
        "404",
        "NoSuchKey",
        "NoSuchBucket",
    }
)


def _s3_should_retry(exc: Exception) -> bool:
    if isinstance(exc, ClientError):
        code = exc.response.get("Error", {}).get("Code", "")
        return code not in _NON_RETRYABLE_S3
    return True


_S3_ASYNC_RETRY = async_retry(retryable_exceptions=(ClientError,), should_retry=_s3_should_retry)

logger = logging.getLogger(__name__)


class S3BlobStore:
    def __init__(self, config: BlobStoreConfig) -> None:
        self._bucket = config.bucket
        self._region = config.region
        self._endpoint_url = config.endpoint_url
        self._session = aioboto3.Session()

        self._client_ctx: ClientCreatorContext[S3Client] | None = None
        self._client: S3Client | None = None

        logger.debug(
            "S3BlobStore initialised (bucket=%s region=%s endpoint=%s)",
            self._bucket,
            self._region,
            self._endpoint_url,
        )

    async def connect(self) -> None:
        # aioboto3.Session.client() returns an async context manager at runtime, not a live client
        self._client_ctx = self._session.client("s3", region_name=self._region, endpoint_url=self._endpoint_url)
        # Enter it manually here so the client lifetime spans the store's connect/close lifecycle.
        self._client = await self._client_ctx.__aenter__()
        logger.info("S3BlobStore client connected (bucket=%s)", self._bucket)

    async def close(self) -> None:
        if self._client_ctx is not None:
            await self._client_ctx.__aexit__(None, None, None)
            self._client_ctx = None
            self._client = None
            logger.debug("S3BlobStore client closed")

    @property
    def client(self) -> S3Client:
        if self._client is None:
            raise RuntimeError("S3BlobStore.connect() has not been called")
        return self._client

    @_S3_ASYNC_RETRY
    async def put(self, key: str, data: bytes) -> None:
        await self.client.put_object(Bucket=self._bucket, Key=key, Body=data)

    @_S3_ASYNC_RETRY
    async def get(self, key: str) -> bytes:
        response = await self.client.get_object(Bucket=self._bucket, Key=key)
        return await response["Body"].read()

    @_S3_ASYNC_RETRY
    async def exists(self, key: str) -> bool:
        try:
            await self.client.head_object(Bucket=self._bucket, Key=key)
            return True
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") == "404":
                return False
            raise
