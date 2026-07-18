from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr

from seshat.infra.vector_store.cohere_embeddings import CohereEmbeddings


def _embed_response(vectors: list[list[float]]):
    response = MagicMock()
    response.embeddings.float_ = vectors
    return response


class TestCohereEmbeddingsSyncMethods:
    def test_embed_documents_raises(self):
        with patch("cohere.AsyncClientV2"):
            embeddings = CohereEmbeddings("embed-v4.0", SecretStr("key"))
        with pytest.raises(NotImplementedError):
            embeddings.embed_documents(["a"])

    def test_embed_query_raises(self):
        with patch("cohere.AsyncClientV2"):
            embeddings = CohereEmbeddings("embed-v4.0", SecretStr("key"))
        with pytest.raises(NotImplementedError):
            embeddings.embed_query("a")


class TestCohereEmbeddingsAembedDocuments:
    async def test_returns_one_vector_per_text(self):
        mock_client = AsyncMock()
        mock_client.embed = AsyncMock(return_value=_embed_response([[0.1, 0.2], [0.3, 0.4]]))
        with patch("cohere.AsyncClientV2", return_value=mock_client):
            embeddings = CohereEmbeddings("embed-v4.0", SecretStr("key"))
            result = await embeddings.aembed_documents(["doc one", "doc two"])

        assert result == [[0.1, 0.2], [0.3, 0.4]]
        mock_client.embed.assert_awaited_once_with(
            model="embed-v4.0",
            input_type="search_document",
            texts=["doc one", "doc two"],
            embedding_types=["float"],
        )


class TestCohereEmbeddingsAembedQuery:
    async def test_returns_single_vector(self):
        mock_client = AsyncMock()
        mock_client.embed = AsyncMock(return_value=_embed_response([[0.5, 0.6]]))
        with patch("cohere.AsyncClientV2", return_value=mock_client):
            embeddings = CohereEmbeddings("embed-v4.0", SecretStr("key"))
            result = await embeddings.aembed_query("a query")

        assert result == [0.5, 0.6]
        mock_client.embed.assert_awaited_once_with(
            model="embed-v4.0",
            input_type="search_query",
            texts=["a query"],
            embedding_types=["float"],
        )
