from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from seshat.blob_store.s3_store import S3BlobStore

if TYPE_CHECKING:
    from seshat.config.settings import SeshatConfig

logger = logging.getLogger(__name__)


def get_blob_store(config: SeshatConfig) -> S3BlobStore:
    logger.debug("Initialising blob store (bucket=%s)", config.blob_store.bucket)
    return S3BlobStore(config.blob_store)
