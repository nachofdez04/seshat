from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from seshat.config.settings import VectorIndexConfig, VectorStoreConfig
from seshat.vector_store.factory import get_vector_store
from seshat.vector_store.pgvector_store import PGVectorStore
from tests.unit.conftest import _FAKE_DB_URL

if TYPE_CHECKING:
    from seshat.config.settings import SeshatConfig


@pytest.mark.usefixtures("mocked_secrets_resolver")
class TestGetVectorStore:
    def test_pgvector_provider_returns_pgvector_store(self, minimal_config: SeshatConfig):
        with patch("seshat.vector_store.pgvector_store.PGVector"), patch("langchain_openai.OpenAIEmbeddings"):
            store = get_vector_store(minimal_config)
        assert isinstance(store, PGVectorStore)

    def test_resolves_connection_string_from_secret_key(self, minimal_config: SeshatConfig, mocked_secrets_resolver):
        with patch("seshat.vector_store.pgvector_store.PGVector"), patch("langchain_openai.OpenAIEmbeddings"):
            get_vector_store(minimal_config)
        mocked_secrets_resolver.get_secret.assert_called_once_with(minimal_config.vector_store.connection_secret_key)

    def test_propagates_connection_string(self, minimal_config: SeshatConfig):
        with patch("seshat.vector_store.pgvector_store.PGVector"), patch("langchain_openai.OpenAIEmbeddings"):
            store = get_vector_store(minimal_config)
        assert store._connection_string == store._validate_connection_string(_FAKE_DB_URL)

    def test_unknown_provider_raises(self, minimal_config: SeshatConfig):
        bad_config = minimal_config.model_copy(
            update={"vector_store": VectorStoreConfig.model_construct(provider="unknown")}
        )
        with patch("langchain_openai.OpenAIEmbeddings"), pytest.raises(ValueError, match="unknown"):
            get_vector_store(bad_config)

    def test_unsupported_embedding_provider_raises(self, minimal_config: SeshatConfig):
        bad_config = minimal_config.model_copy(
            update={"vector_index": VectorIndexConfig.model_construct(embedding_provider="unknown")}
        )
        with pytest.raises(ValueError, match="Unsupported embedding provider"):
            get_vector_store(bad_config)
