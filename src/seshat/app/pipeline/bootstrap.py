from __future__ import annotations

from typing import TYPE_CHECKING

from seshat.app.agents.grounding import GroundingAgent
from seshat.app.agents.identification.registry import IdentificationAgentRegistry
from seshat.app.agents.resolution.registry import ResolutionRegistry
from seshat.app.pipeline.extraction.node_retriever import NodeRetriever
from seshat.app.pipeline.extraction.orchestrator import ExtractionOrchestrator
from seshat.app.pipeline.extraction.reranker import reranker_factory
from seshat.app.pipeline.extraction.search_engine import SearchEngine
from seshat.app.pipeline.ingestion.orchestrator import IngestionOrchestrator
from seshat.app.pipeline.llm_factory import _build_llm, get_grounding_llm, get_identification_llm, get_resolution_llm
from seshat.app.transcription.factory import get_transcriber
from seshat.infra.secrets.factory import get_secrets_resolver
from seshat.infra.vector_store.factory import get_vector_store

if TYPE_CHECKING:
    from seshat.app.pipeline.extraction.reranker import AbstractReranker
    from seshat.app.repositories.blob_repository import BlobRepository
    from seshat.app.repositories.node_repository import NodeRepository
    from seshat.core.config.settings import SeshatConfig
    from seshat.infra.vector_store.base_store import AbstractVectorStore


def build_extraction_orchestrator(
    config: SeshatConfig,
    node_repo: NodeRepository,
    blob_repo: BlobRepository,
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

    vector_store = get_vector_store(config)
    search_engine = get_search_engine(config, vector_store)
    reranker = _get_reranker(config)
    node_retriever = NodeRetriever(
        rag_config=config.rag,
        node_repo=node_repo,
        search_engine=search_engine,
        reranker=reranker,
    )

    grounding_agent = None
    if config.extraction.grounding is not None:
        grounding_llm = get_grounding_llm(config)
        grounding_agent = GroundingAgent(llm=grounding_llm, config=config.extraction.grounding)

    return ExtractionOrchestrator(
        config=config.extraction,
        identification_registry=identification_registry,
        resolution_registry=resolution_registry,
        node_retriever=node_retriever,
        node_repo=node_repo,
        blob_repo=blob_repo,
        grounding_agent=grounding_agent,
    )


def build_ingestion_orchestrator(config: SeshatConfig, blob_repo: BlobRepository) -> IngestionOrchestrator:
    transcriber = get_transcriber(config)
    return IngestionOrchestrator(transcriber, blob_repo, config.transcription)


def get_search_engine(config: SeshatConfig, vector_store: AbstractVectorStore) -> SearchEngine:
    keyword_llm = _build_llm(config.rag.keyword_extraction_llm, config) if config.rag.keyword_extraction_llm else None
    multi_query_llm = _build_llm(config.rag.multi_query.llm, config) if config.rag.multi_query else None
    return SearchEngine(
        rag_config=config.rag,
        vector_store=vector_store,
        keyword_llm=keyword_llm,
        multi_query_llm=multi_query_llm,
    )


def _get_reranker(config: SeshatConfig) -> AbstractReranker | None:
    if config.rag.reranker is None:
        return None

    reranker_cfg = config.rag.reranker
    assert reranker_cfg.api_key_secret_key is not None

    api_key = get_secrets_resolver(config).get_secret(reranker_cfg.api_key_secret_key)
    return reranker_factory(reranker_cfg, api_key)
