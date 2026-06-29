from __future__ import annotations

from typing import TYPE_CHECKING

from seshat.agents.grounding import GroundingAgent
from seshat.agents.identification.registry import IdentificationAgentRegistry
from seshat.agents.resolution.registry import ResolutionRegistry
from seshat.pipeline.extraction.keyword_extractor import build_keyword_extractor
from seshat.pipeline.extraction.node_retriever import NodeRetriever
from seshat.pipeline.extraction.orchestrator import ExtractionOrchestrator
from seshat.pipeline.ingestion.orchestrator import IngestionOrchestrator
from seshat.pipeline.llm_factory import _build_llm, get_grounding_llm, get_identification_llm, get_resolution_llm
from seshat.transcription.factory import get_transcriber
from seshat.vector_store.factory import get_vector_store

if TYPE_CHECKING:
    from seshat.blob_store.s3_store import S3BlobStore
    from seshat.config.settings import SeshatConfig
    from seshat.knowledge_store.pg_store import PostgresKBStore
    from seshat.vector_store.base_store import AbstractVectorStore


def build_vector_store(seshat_config: SeshatConfig) -> AbstractVectorStore:
    keyword_extractor = None
    if seshat_config.rag.keyword_extraction_llm is not None:
        llm = _build_llm(seshat_config.rag.keyword_extraction_llm, seshat_config)
        keyword_extractor = build_keyword_extractor(llm)
    return get_vector_store(seshat_config, keyword_extractor=keyword_extractor)


def build_extraction_orchestrator(
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


def build_ingestion_orchestrator(config: SeshatConfig, blob_store: S3BlobStore) -> IngestionOrchestrator:
    transcriber = get_transcriber(config)
    ingestion_orch = IngestionOrchestrator(transcriber, blob_store, config.transcription)
    return ingestion_orch
