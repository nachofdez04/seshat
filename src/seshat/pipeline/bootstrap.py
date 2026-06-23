from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from seshat.agents.grounding import GroundingAgent
from seshat.agents.identification.registry import IdentificationAgentRegistry
from seshat.agents.resolution.registry import ResolutionRegistry
from seshat.blob_store.factory import get_blob_store
from seshat.knowledge_store.factory import get_kb_store
from seshat.pipeline.extraction.node_retriever import NodeRetriever
from seshat.pipeline.extraction.orchestrator import ExtractionOrchestrator
from seshat.pipeline.llm_factory import _build_llm, get_grounding_llm, get_identification_llm, get_resolution_llm
from seshat.vector_store.factory import get_vector_store

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from seshat.blob_store.s3_store import S3BlobStore
    from seshat.config.settings import SeshatConfig
    from seshat.knowledge_store.pg_store import PostgresKBStore
    from seshat.vector_store.base_store import AbstractVectorStore


@asynccontextmanager
async def build_orchestrator(seshat_config: SeshatConfig) -> AsyncIterator[ExtractionOrchestrator]:
    kb_store = get_kb_store(seshat_config)
    await kb_store.connect()

    vector_store = get_vector_store(seshat_config)
    blob_store = get_blob_store(seshat_config)
    await blob_store.connect()

    try:
        yield _build_orchestrator(seshat_config, kb_store, vector_store, blob_store)
    finally:
        await kb_store.close()
        await blob_store.close()


def _build_orchestrator(
    config: SeshatConfig,
    kb_store: PostgresKBStore,
    vector_store: AbstractVectorStore,
    blob_store: S3BlobStore,
) -> ExtractionOrchestrator:
    identification_llm = get_identification_llm(config)
    resolution_llm = get_resolution_llm(config)

    review_llm = None
    identification_self_review_cfg = config.extraction.identification_self_review
    if identification_self_review_cfg.enabled and identification_self_review_cfg.llm is not None:
        review_llm = _build_llm(identification_self_review_cfg.llm, config)

    resolution_review_llm = None
    resolution_self_review_cfg = config.extraction.resolution_self_review
    if resolution_self_review_cfg.enabled and resolution_self_review_cfg.llm is not None:
        resolution_review_llm = _build_llm(resolution_self_review_cfg.llm, config)

    identification_registry = IdentificationAgentRegistry(
        llm=identification_llm, config=config.extraction, review_llm=review_llm
    )
    resolution_registry = ResolutionRegistry(
        llm=resolution_llm, config=config.extraction, review_llm=resolution_review_llm
    )
    node_retriever = NodeRetriever(rag_config=config.rag, kb_store=kb_store, vector_store=vector_store)

    grounding_agent = None
    if config.extraction.grounding is not None:
        grounding_llm = get_grounding_llm(config)
        grounding_agent = GroundingAgent(llm=grounding_llm, config=config.extraction.grounding)

    return ExtractionOrchestrator(
        config=config.extraction,
        identification_registry=identification_registry,
        resolution_registry=resolution_registry,
        node_retriever=node_retriever,
        kb_store=kb_store,
        blob_store=blob_store,
        grounding_agent=grounding_agent,
    )
