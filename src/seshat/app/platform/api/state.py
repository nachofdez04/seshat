from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

from seshat.app.pipeline.bootstrap import (
    build_extraction_orchestrator,
    build_ingestion_orchestrator,
)
from seshat.app.platform.worker.queue import AsyncioTaskQueue
from seshat.app.repositories.blob_repository import BlobRepository
from seshat.app.repositories.node_repository import NodeRepository
from seshat.app.repositories.ops_repository import OpsRepository
from seshat.app.services.admin import AdminService
from seshat.app.services.graph import GraphService
from seshat.app.services.health import HealthService
from seshat.app.services.job import JobService
from seshat.infra.blob_store.factory import get_blob_store
from seshat.infra.knowledge_store.factory import get_kb_store
from seshat.infra.ops_store.factory import get_ops_store
from seshat.infra.vector_store.factory import get_vector_store

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from seshat.core.config.settings import SeshatConfig


@dataclass
class AppState:
    config: SeshatConfig
    admin_service: AdminService
    health_service: HealthService
    graph_service: GraphService
    job_service: JobService


@asynccontextmanager
async def build_app_state(config: SeshatConfig) -> AsyncIterator[AppState]:
    ops_store = get_ops_store(config)
    await ops_store.connect()

    kb_store = get_kb_store(config)
    await kb_store.connect()

    blob_store = get_blob_store(config)
    await blob_store.connect()

    try:
        vector_store = get_vector_store(config)
        node_repo = NodeRepository(kb_store, vector_store)
        blob_repo = BlobRepository(blob_store)
        extraction_orchestrator = build_extraction_orchestrator(config, node_repo, blob_repo)
        ingestion_orchestrator = build_ingestion_orchestrator(config, blob_repo)
        ops_repo = OpsRepository(ops_store)
        admin_service = AdminService(ops_repo=ops_repo)
        health_service = HealthService(
            ops_repo=ops_repo,
            blob_repo=blob_repo,
            blob_config=config.blob_store,
            observability_config=config.observability,
        )
        graph_service = GraphService(node_repo, extraction_orchestrator)
        queue = AsyncioTaskQueue()
        job_service = JobService(
            config,
            ops_repo,
            blob_repo,
            node_repo,
            extraction_orchestrator,
            ingestion_orchestrator,
            queue,
        )
        yield AppState(
            config=config,
            admin_service=admin_service,
            health_service=health_service,
            graph_service=graph_service,
            job_service=job_service,
        )
    finally:
        await kb_store.close()
        await blob_store.close()
        await ops_store.close()
