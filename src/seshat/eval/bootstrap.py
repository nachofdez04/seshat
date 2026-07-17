from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from seshat.app.pipeline.bootstrap import build_extraction_orchestrator as _build_extraction_orchestrator
from seshat.app.repositories.blob_repository import BlobRepository
from seshat.app.repositories.node_repository import NodeRepository
from seshat.infra.blob_store.factory import get_blob_store
from seshat.infra.knowledge_store.factory import get_kb_store
from seshat.infra.vector_store.factory import get_vector_store

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from seshat.app.pipeline.extraction.orchestrator import ExtractionOrchestrator
    from seshat.core.config.settings import SeshatConfig


@asynccontextmanager
async def build_extraction_orchestrator(
    seshat_config: SeshatConfig,
) -> AsyncIterator[ExtractionOrchestrator]:
    vector_store = get_vector_store(seshat_config)

    kb_store = get_kb_store(seshat_config)
    await kb_store.connect()

    blob_store = get_blob_store(seshat_config)
    await blob_store.connect()

    node_repo = NodeRepository(kb_store, vector_store)
    blob_repo = BlobRepository(blob_store)

    try:
        yield _build_extraction_orchestrator(seshat_config, node_repo, blob_repo)
    finally:
        await kb_store.close()
        await blob_store.close()
