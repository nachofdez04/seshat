from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

from seshat.blob_store.factory import get_blob_store
from seshat.knowledge_store.factory import get_kb_store
from seshat.ops.factory import get_ops_ledger
from seshat.pipeline.bootstrap import build_extraction_orchestrator, build_ingestion_orchestrator, build_vector_store
from seshat.worker.manual_ingestion import ManualIngestionService
from seshat.worker.writing_stage import WritingStage

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from seshat.blob_store.s3_store import S3BlobStore
    from seshat.config.settings import SeshatConfig
    from seshat.knowledge_store.pg_store import PostgresKBStore
    from seshat.ops.ledger import OpsLedger
    from seshat.pipeline.extraction.orchestrator import ExtractionOrchestrator
    from seshat.pipeline.ingestion.orchestrator import IngestionOrchestrator
    from seshat.vector_store.base_store import AbstractVectorStore


@dataclass
class WorkerContext:
    extraction_orch: ExtractionOrchestrator
    ingestion_orch: IngestionOrchestrator
    writing_stage: WritingStage
    ops: OpsLedger
    kb_store: PostgresKBStore
    vector_store: AbstractVectorStore
    manual_ingestion: ManualIngestionService
    blob_store: S3BlobStore


@asynccontextmanager
async def build_worker_context(seshat_config: SeshatConfig) -> AsyncIterator[WorkerContext]:
    async with get_ops_ledger(seshat_config) as ops:
        vector_store = build_vector_store(seshat_config)

        kb_store = get_kb_store(seshat_config)
        await kb_store.connect()

        blob_store = get_blob_store(seshat_config)
        await blob_store.connect()

        try:
            extraction_orch = build_extraction_orchestrator(seshat_config, kb_store, vector_store, blob_store)
            ingestion_orch = build_ingestion_orchestrator(seshat_config, blob_store)
            writing_stage = WritingStage(kb_store, vector_store)
            manual_ingestion = ManualIngestionService(kb_store, vector_store, extraction_orch)
            yield WorkerContext(
                extraction_orch=extraction_orch,
                ingestion_orch=ingestion_orch,
                writing_stage=writing_stage,
                ops=ops,
                kb_store=kb_store,
                vector_store=vector_store,
                manual_ingestion=manual_ingestion,
                blob_store=blob_store,
            )
        finally:
            await kb_store.close()
            await blob_store.close()
