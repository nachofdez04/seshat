from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from seshat.blob_store.s3_store import S3BlobStore
    from seshat.config.settings import SeshatConfig
    from seshat.knowledge_store.pg_store import PostgresKBStore
    from seshat.models.nodes import ExtractionResult
    from seshat.ops.ledger import OpsLedger
    from seshat.worker.manual_ingestion import ManualIngestionService
    from seshat.worker.pipeline_runner import PipelineRunner
    from seshat.worker.queue import AsyncioTaskQueue


@dataclass
class AppState:
    config: SeshatConfig
    kb_store: PostgresKBStore
    manual_ingestion: ManualIngestionService
    ops: OpsLedger
    queue: AsyncioTaskQueue
    results: dict[str, ExtractionResult]
    runner: PipelineRunner
    blob_store: S3BlobStore
