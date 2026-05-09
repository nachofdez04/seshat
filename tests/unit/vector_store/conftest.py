import pytest

from seshat.config.settings import VectorIndexConfig, VectorStoreConfig
from seshat.models.enums import EmbeddingProvider, VectorStoreProvider


@pytest.fixture
def pg_vector_store_config() -> VectorStoreConfig:
    return VectorStoreConfig(provider=VectorStoreProvider.PGVECTOR)


@pytest.fixture
def vector_index_config() -> VectorIndexConfig:
    return VectorIndexConfig(embedding_provider=EmbeddingProvider.OPENAI, embedding_model="text-embedding-3-small")
