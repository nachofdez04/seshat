from __future__ import annotations

from typing import TYPE_CHECKING

from seshat.blob_store.factory import get_blob_store
from seshat.blob_store.s3_store import S3BlobStore

if TYPE_CHECKING:
    from seshat.config.settings import SeshatConfig


class TestGetBlobStore:
    def test_returns_s3_blob_store(self, minimal_config: SeshatConfig):
        store = get_blob_store(minimal_config)
        assert isinstance(store, S3BlobStore)

    def test_propagates_bucket(self, minimal_config: SeshatConfig):
        store = get_blob_store(minimal_config)
        assert store._bucket == minimal_config.blob_store.bucket
