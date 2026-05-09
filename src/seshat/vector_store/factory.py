from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from seshat.models.enums import EmbeddingProvider, VectorStoreProvider
from seshat.secrets.factory import get_secrets_resolver

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings

    from seshat.config.settings import SeshatConfig, VectorIndexConfig
    from seshat.vector_store.base_store import AbstractVectorStore

logger = logging.getLogger(__name__)


def _build_embeddings(index: VectorIndexConfig) -> Embeddings:
    match index.embedding_provider:
        case EmbeddingProvider.OPENAI:
            from langchain_openai import OpenAIEmbeddings

            return OpenAIEmbeddings(model=index.embedding_model)
        case _:
            raise ValueError(f"Unsupported embedding provider: {index.embedding_provider!r}")


def get_vector_store(config: SeshatConfig) -> AbstractVectorStore:
    secrets = get_secrets_resolver(config)
    connection_string = secrets.get_secret(config.vector_store.connection_secret_key)

    logger.debug("Initialising embedding model integration: %s", config.vector_index.embedding_provider)
    embeddings = _build_embeddings(config.vector_index)

    logger.debug("Initialising vector store: %s", config.vector_store.provider)
    match config.vector_store.provider:
        case VectorStoreProvider.PGVECTOR:
            from seshat.vector_store.pgvector_store import PGVectorStore

            return PGVectorStore(config.vector_store, config.vector_index, embeddings, connection_string)
        case _:
            raise ValueError(f"Unknown vector store provider: {config.vector_store.provider!r}")
