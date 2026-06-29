from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from seshat.blob_store.factory import get_blob_store
from seshat.knowledge_store.factory import get_kb_store
from seshat.pipeline.bootstrap import build_extraction_orchestrator as _build_extraction_orchestrator
from seshat.pipeline.bootstrap import build_vector_store

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from seshat.config.settings import SeshatConfig
    from seshat.pipeline.extraction.orchestrator import ExtractionOrchestrator


@asynccontextmanager
async def build_extraction_orchestrator(
    seshat_config: SeshatConfig,
) -> AsyncIterator[ExtractionOrchestrator]:
    vector_store = build_vector_store(seshat_config)

    kb_store = get_kb_store(seshat_config)
    await kb_store.connect()

    blob_store = get_blob_store(seshat_config)
    await blob_store.connect()

    try:
        yield _build_extraction_orchestrator(seshat_config, kb_store, vector_store, blob_store)
    finally:
        await kb_store.close()
        await blob_store.close()
