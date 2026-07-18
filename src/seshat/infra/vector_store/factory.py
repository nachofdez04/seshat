from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import SecretStr

from seshat.app.platform.observability.usage_tracker import TrackingEmbeddings
from seshat.core.models.enums import EmbeddingProvider, VectorStoreProvider
from seshat.core.utils.log import get_logger
from seshat.infra.secrets.factory import get_secrets_resolver

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings

    from seshat.core.config.settings import SeshatConfig, VectorIndexConfig
    from seshat.infra.vector_store.base_store import AbstractVectorStore


logger = get_logger(__name__)


def _build_embeddings(index: VectorIndexConfig, config: SeshatConfig) -> Embeddings:
    secrets = get_secrets_resolver(config)
    api_key = SecretStr(secrets.get_secret(index.api_key_secret_key))  # type: ignore[arg-type]

    raw: Embeddings
    match index.embedding_provider:
        case EmbeddingProvider.OPENAI:
            from langchain_openai import OpenAIEmbeddings

            raw = OpenAIEmbeddings(model=index.embedding_model, api_key=api_key)
        case EmbeddingProvider.AZURE_OPENAI:
            from langchain_openai import AzureOpenAIEmbeddings

            raw = AzureOpenAIEmbeddings(azure_deployment=index.embedding_model, api_key=api_key)
        case EmbeddingProvider.COHERE:
            from seshat.infra.vector_store.cohere_embeddings import CohereEmbeddings

            raw = CohereEmbeddings(model=index.embedding_model, api_key=api_key)
        case _:
            raise ValueError(f"Unsupported embedding provider: {index.embedding_provider!r}")

    return TrackingEmbeddings(raw)


def get_vector_store(config: SeshatConfig) -> AbstractVectorStore:
    secrets = get_secrets_resolver(config)
    connection_string = secrets.get_secret(config.vector_store.connection_secret_key)

    logger.debug("Initialising embedding model integration: %s", config.vector_index.embedding_provider)
    embeddings = _build_embeddings(config.vector_index, config)

    logger.debug("Initialising vector store: %s", config.vector_store.provider)
    match config.vector_store.provider:
        case VectorStoreProvider.PGVECTOR:
            from seshat.infra.vector_store.pgvector_store import PGVectorStore

            return PGVectorStore(config.vector_store, config.vector_index, embeddings, connection_string)
        case _:
            raise ValueError(f"Unknown vector store provider: {config.vector_store.provider!r}")
