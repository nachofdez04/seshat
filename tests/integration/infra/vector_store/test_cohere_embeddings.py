from __future__ import annotations

import os

import pytest
from pydantic import SecretStr

from seshat.infra.vector_store.cohere_embeddings import CohereEmbeddings
from tests.integration.conftest import SKIP_IF_NO_COHERE_API

pytestmark = [pytest.mark.integration, pytest.mark.llm, pytest.mark.embedding, SKIP_IF_NO_COHERE_API]


@pytest.fixture
def embeddings() -> CohereEmbeddings:
    return CohereEmbeddings("embed-v4.0", SecretStr(os.environ["COHERE_API_KEY"]))


class TestCohereEmbeddingsAembedDocuments:
    async def test_returns_one_nonempty_vector_per_text(self, embeddings: CohereEmbeddings):
        result = await embeddings.aembed_documents(["Redis cache eviction policy", "PostgreSQL schema migration"])

        assert len(result) == 2
        assert all(len(vector) > 0 for vector in result)


class TestCohereEmbeddingsAembedQuery:
    async def test_returns_single_nonempty_vector(self, embeddings: CohereEmbeddings):
        result = await embeddings.aembed_query("Redis memory management")

        assert len(result) > 0
