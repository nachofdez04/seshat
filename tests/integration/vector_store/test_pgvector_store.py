from __future__ import annotations

import pytest
from langchain_core.embeddings import Embeddings

from seshat.models.api_graph import NodeFilter
from seshat.models.enums import ConceptType, SearchMode
from seshat.vector_store.pgvector_store import PGVectorStore
from tests.integration.conftest import SKIP_IF_NO_EMBEDDINGS_API, SKIP_IF_NO_POSTGRES

pytestmark = [pytest.mark.integration, SKIP_IF_NO_POSTGRES]

_EMBEDDING_MARKS = [pytest.mark.llm, pytest.mark.embedding, SKIP_IF_NO_EMBEDDINGS_API]

_TEST_NODE_ID = "test-node-1"
_DISTINCTIVE_TERM = "zymurgy"  # rare term unlikely to match unrelated documents


class _DummyEmbeddings(Embeddings):
    """Zero-vector embeddings for keyword-only tests — no embedding API call made."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * 1536 for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.0] * 1536


async def _make_store_with_extractor(pg_test_url, collection: str) -> PGVectorStore:
    from seshat.config.settings import VectorIndexConfig, VectorStoreConfig

    async def _passthrough_extractor(query: str) -> str:
        return query

    index = VectorIndexConfig().model_copy(update={"collection": collection})
    return PGVectorStore(VectorStoreConfig(), index, _DummyEmbeddings(), pg_test_url, _passthrough_extractor)


@pytest.fixture
async def keyword_store(pg_test_url):
    store = await _make_store_with_extractor(pg_test_url, "test_keyword_search")
    yield store
    await store._store.adelete_collection()


@pytest.fixture
async def hybrid_store(pg_test_url):
    store = await _make_store_with_extractor(pg_test_url, "test_hybrid_search")
    yield store
    await store._store.adelete_collection()


class TestPGVectorStoreSearch:
    pytestmark = _EMBEDDING_MARKS

    async def test_upsert_then_search(self, vector_store: PGVectorStore):
        await vector_store.upsert(
            _TEST_NODE_ID,
            "Use PostgreSQL for session storage",
            {"node_type": "decision", "confidence": 0.9},
        )
        results = await vector_store.search("PostgreSQL session storage", top_k=5)
        assert any(r.node_id == _TEST_NODE_ID for r in results)

    async def test_with_node_type_filter_matching(self, vector_store: PGVectorStore):
        await vector_store.upsert(
            _TEST_NODE_ID,
            "Use PostgreSQL for session storage",
            {"node_type": "decision", "confidence": 0.9},
        )
        results = await vector_store.search(
            "PostgreSQL session storage",
            top_k=5,
            node_filter=NodeFilter(node_type=ConceptType.DECISION),
        )
        assert any(r.node_id == _TEST_NODE_ID for r in results)

    async def test_with_node_type_filter_nonmatching(self, vector_store: PGVectorStore):
        await vector_store.upsert(
            _TEST_NODE_ID,
            "Use PostgreSQL for session storage",
            {"node_type": "decision", "confidence": 0.9},
        )
        results = await vector_store.search(
            "PostgreSQL session storage",
            top_k=5,
            node_filter=NodeFilter(node_type=ConceptType.RISK),
        )
        assert not any(r.node_id == _TEST_NODE_ID for r in results)


class TestPGVectorStoreDelete:
    pytestmark = _EMBEDDING_MARKS

    async def test_delete(self, vector_store: PGVectorStore):
        await vector_store.upsert(
            _TEST_NODE_ID,
            "Use Redis for caching",
            {"node_type": "decision", "confidence": 0.8},
        )
        await vector_store.delete(_TEST_NODE_ID)
        results = await vector_store.search("Redis caching", top_k=5)
        assert not any(r.node_id == _TEST_NODE_ID for r in results)


class TestKeywordSearch:
    async def test_finds_node_by_distinctive_term(self, keyword_store: PGVectorStore):
        await keyword_store.upsert(
            _TEST_NODE_ID,
            f"Decision about {_DISTINCTIVE_TERM} process",
            {"node_type": "decision", "confidence": 0.9},
        )
        results = await keyword_store.search(_DISTINCTIVE_TERM, top_k=5, mode=SearchMode.KEYWORD)
        assert any(r.node_id == _TEST_NODE_ID for r in results)

    async def test_absent_term_returns_empty(self, keyword_store: PGVectorStore):
        await keyword_store.upsert(
            _TEST_NODE_ID,
            "Decision about PostgreSQL storage",
            {"node_type": "decision", "confidence": 0.9},
        )
        results = await keyword_store.search("xylophone quasar", top_k=5, mode=SearchMode.KEYWORD)
        assert results == []

    async def test_does_not_return_node_missing_the_term(self, keyword_store: PGVectorStore):
        node_b = "test-node-2"
        await keyword_store.upsert(
            _TEST_NODE_ID,
            f"Decision about {_DISTINCTIVE_TERM}",
            {"node_type": "decision", "confidence": 0.9},
        )
        await keyword_store.upsert(
            node_b,
            "Decision about caching strategy",
            {"node_type": "decision", "confidence": 0.9},
        )
        results = await keyword_store.search(_DISTINCTIVE_TERM, top_k=5, mode=SearchMode.KEYWORD)
        assert any(r.node_id == _TEST_NODE_ID for r in results)
        assert not any(r.node_id == node_b for r in results)

    async def test_results_have_positive_score(self, keyword_store: PGVectorStore):
        await keyword_store.upsert(
            _TEST_NODE_ID,
            f"Decision about {_DISTINCTIVE_TERM}",
            {"node_type": "decision", "confidence": 0.9},
        )
        results = await keyword_store.search(_DISTINCTIVE_TERM, top_k=5, mode=SearchMode.KEYWORD)
        assert all(r.score > 0 for r in results)


class TestHybridSearch:
    async def test_finds_node_present_in_both_legs(self, hybrid_store: PGVectorStore):
        text = f"Use Redis for caching sessions {_DISTINCTIVE_TERM}"
        await hybrid_store.upsert(_TEST_NODE_ID, text, {"node_type": "decision", "confidence": 0.9})
        results = await hybrid_store.search(text, top_k=5, mode=SearchMode.HYBRID)
        assert any(r.node_id == _TEST_NODE_ID for r in results)

    async def test_node_in_both_legs_ranks_first(self, hybrid_store: PGVectorStore):
        node_b = "test-node-2"
        shared_text = f"Use Redis for caching {_DISTINCTIVE_TERM}"
        await hybrid_store.upsert(_TEST_NODE_ID, shared_text, {"node_type": "decision", "confidence": 0.9})
        # node_b text is semantically similar but lacks the distinctive term
        await hybrid_store.upsert(node_b, "Use Redis for caching", {"node_type": "decision", "confidence": 0.9})

        results = await hybrid_store.search(shared_text, top_k=5, mode=SearchMode.HYBRID)

        assert results[0].node_id == _TEST_NODE_ID
