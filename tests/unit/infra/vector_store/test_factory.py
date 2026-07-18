from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from seshat.core.config.settings import VectorIndexConfig, VectorStoreConfig
from seshat.core.models.enums import EmbeddingProvider
from seshat.infra.vector_store.cohere_embeddings import CohereEmbeddings
from seshat.infra.vector_store.factory import get_vector_store

if TYPE_CHECKING:
    from seshat.core.config.settings import SeshatConfig


@pytest.mark.usefixtures("mocked_secrets_resolver")
class TestGetVectorStore:
    def test_resolves_connection_string_from_secret_key(self, minimal_config: SeshatConfig, mocked_secrets_resolver):
        with patch("seshat.infra.vector_store.pgvector_store.PGVector"), patch("langchain_openai.OpenAIEmbeddings"):
            get_vector_store(minimal_config)
        mocked_secrets_resolver.get_secret.assert_any_call(minimal_config.vector_store.connection_secret_key)

    def test_resolves_embedding_api_key(self, minimal_config: SeshatConfig, mocked_secrets_resolver):
        with patch("seshat.infra.vector_store.pgvector_store.PGVector"), patch("langchain_openai.OpenAIEmbeddings"):
            get_vector_store(minimal_config)
        mocked_secrets_resolver.get_secret.assert_any_call(minimal_config.vector_index.api_key_secret_key)

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

    def test_cohere_embedding_provider_builds_cohere_embeddings(self, minimal_config: SeshatConfig):
        cohere_config = minimal_config.model_copy(
            update={"vector_index": VectorIndexConfig(embedding_provider=EmbeddingProvider.COHERE)}
        )
        with (
            patch("seshat.infra.vector_store.pgvector_store.PGVector") as mock_pgvector,
            patch("cohere.AsyncClientV2"),
        ):
            get_vector_store(cohere_config)
        embeddings = mock_pgvector.call_args.kwargs["embeddings"]
        assert isinstance(embeddings._embeddings, CohereEmbeddings)
