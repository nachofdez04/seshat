import pytest
from langchain_openai import OpenAIEmbeddings

from seshat.config.settings import VectorIndexConfig, VectorStoreConfig
from seshat.models.api import NodeFilter
from seshat.models.enums import ConceptType
from seshat.vector_store.pgvector_store import PGVectorStore
from tests.integration.conftest import SKIP_IF_NO_OPENAI, SKIP_IF_NO_POSTGRES

pytestmark = [pytest.mark.integration, SKIP_IF_NO_POSTGRES, SKIP_IF_NO_OPENAI]

_TEST_NODE_ID = "test-node-1"


@pytest.fixture
async def store(pg_test_url):
    config = VectorStoreConfig()
    index = VectorIndexConfig(collection="test_collection")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    s = PGVectorStore(config, index, embeddings, pg_test_url)
    yield s
    await s.delete(_TEST_NODE_ID)


class TestPGVectorStoreSearch:
    async def test_upsert_then_search(self, store: PGVectorStore):
        await store.upsert(
            _TEST_NODE_ID,
            "Use PostgreSQL for session storage",
            {"node_type": "adr", "confidence": 0.9},
        )
        results = await store.search("PostgreSQL session storage", top_k=5)
        assert any(r.node_id == _TEST_NODE_ID for r in results)

    async def test_with_node_type_filter_matching(self, store: PGVectorStore):
        await store.upsert(
            _TEST_NODE_ID,
            "Use PostgreSQL for session storage",
            {"node_type": "adr", "confidence": 0.9},
        )
        results = await store.search(
            "PostgreSQL session storage",
            top_k=5,
            node_filter=NodeFilter(node_type=ConceptType.ADR),
        )
        assert any(r.node_id == _TEST_NODE_ID for r in results)

    async def test_with_node_type_filter_nonmatching(self, store: PGVectorStore):
        await store.upsert(
            _TEST_NODE_ID,
            "Use PostgreSQL for session storage",
            {"node_type": "adr", "confidence": 0.9},
        )
        results = await store.search(
            "PostgreSQL session storage",
            top_k=5,
            node_filter=NodeFilter(node_type=ConceptType.RISK),
        )
        assert not any(r.node_id == _TEST_NODE_ID for r in results)


class TestPGVectorStoreDelete:
    async def test_delete(self, store: PGVectorStore):
        await store.upsert(
            _TEST_NODE_ID,
            "Use Redis for caching",
            {"node_type": "adr", "confidence": 0.8},
        )
        await store.delete(_TEST_NODE_ID)
        results = await store.search("Redis caching", top_k=5)
        assert not any(r.node_id == _TEST_NODE_ID for r in results)
