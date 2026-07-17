from __future__ import annotations

from uuid import UUID

import pytest
from langchain_core.embeddings import Embeddings

from seshat.core.config.settings import VectorIndexConfig, VectorStoreConfig
from seshat.core.models.api_graph import NodeFilter
from seshat.core.models.enums import ConceptType, SearchMode
from seshat.infra.vector_store.pgvector_store import PGVectorStore
from tests.integration.conftest import SKIP_IF_NO_EMBEDDINGS_API, SKIP_IF_NO_POSTGRES

pytestmark = [pytest.mark.integration, SKIP_IF_NO_POSTGRES]

_EMBEDDING_MARKS = [pytest.mark.llm, pytest.mark.embedding, SKIP_IF_NO_EMBEDDINGS_API]


_TEST_NODE_ID = "00000000-0000-0000-0000-000000000001"
_TEST_NODE_UUID = UUID(_TEST_NODE_ID)
_NODE_B_ID = "00000000-0000-0000-0000-000000000002"
_NODE_B_UUID = UUID(_NODE_B_ID)
_DISTINCTIVE_TERM = "zymurgy"  # rare term unlikely to match unrelated documents


class _DummyEmbeddings(Embeddings):
    """Zero-vector embeddings for keyword-only tests — no embedding API call made."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * 1536 for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.0] * 1536


def _make_store(pg_test_url, collection: str) -> PGVectorStore:
    index = VectorIndexConfig().model_copy(update={"collection": collection})
    return PGVectorStore(VectorStoreConfig(), index, _DummyEmbeddings(), pg_test_url)


@pytest.fixture
async def keyword_store(pg_test_url):
    store = _make_store(pg_test_url, "test_keyword_search")
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
        assert any(r.node_id == _TEST_NODE_UUID for r in results)

    async def test_with_node_type_filter_matching(self, vector_store: PGVectorStore):
        await vector_store.upsert(
            _TEST_NODE_ID,
            "Use PostgreSQL for session storage",
            {"node_type": "decision", "confidence": 0.9},
        )
        await vector_store.upsert(
            _NODE_B_ID,
            "Risk of data loss during migration",
            {"node_type": "risk", "confidence": 0.8},
        )
        results = await vector_store.search(
            "PostgreSQL session storage",
            top_k=5,
            node_filter=NodeFilter(node_type=ConceptType.DECISION),
        )
        result_ids = {r.node_id for r in results}
        assert _TEST_NODE_UUID in result_ids
        assert _NODE_B_UUID not in result_ids

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
        assert not any(r.node_id == _TEST_NODE_UUID for r in results)


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
        assert not any(r.node_id == _TEST_NODE_UUID for r in results)


class TestKeywordSearch:
    async def test_finds_node_by_distinctive_term(self, keyword_store: PGVectorStore):
        await keyword_store.upsert(
            _TEST_NODE_ID,
            f"Decision about {_DISTINCTIVE_TERM} process",
            {"node_type": "decision", "confidence": 0.9},
        )
        results = await keyword_store.search(_DISTINCTIVE_TERM, top_k=5, mode=SearchMode.KEYWORD)
        assert any(r.node_id == _TEST_NODE_UUID for r in results)

    async def test_absent_term_returns_empty(self, keyword_store: PGVectorStore):
        await keyword_store.upsert(
            _TEST_NODE_ID,
            "Decision about PostgreSQL storage",
            {"node_type": "decision", "confidence": 0.9},
        )
        results = await keyword_store.search("xylophone quasar", top_k=5, mode=SearchMode.KEYWORD)
        assert results == []

    async def test_does_not_return_node_missing_the_term(self, keyword_store: PGVectorStore):
        await keyword_store.upsert(
            _TEST_NODE_ID,
            f"Decision about {_DISTINCTIVE_TERM}",
            {"node_type": "decision", "confidence": 0.9},
        )
        await keyword_store.upsert(
            _NODE_B_ID,
            "Decision about caching strategy",
            {"node_type": "decision", "confidence": 0.9},
        )
        results = await keyword_store.search(_DISTINCTIVE_TERM, top_k=5, mode=SearchMode.KEYWORD)
        assert any(r.node_id == _TEST_NODE_UUID for r in results)
        assert not any(r.node_id == _NODE_B_UUID for r in results)

    async def test_results_have_positive_score(self, keyword_store: PGVectorStore):
        await keyword_store.upsert(
            _TEST_NODE_ID,
            f"Decision about {_DISTINCTIVE_TERM}",
            {"node_type": "decision", "confidence": 0.9},
        )
        results = await keyword_store.search(_DISTINCTIVE_TERM, top_k=5, mode=SearchMode.KEYWORD)
        assert all(r.score > 0 for r in results)


class TestScoreThreshold:
    pytestmark = _EMBEDDING_MARKS

    async def test_low_threshold_includes_relevant_result(self, vector_store: PGVectorStore):
        await vector_store.upsert(
            _TEST_NODE_ID,
            "Use PostgreSQL for transactional data",
            {"node_type": "decision", "confidence": 0.9},
        )
        results = await vector_store.search("PostgreSQL transactional", top_k=5, score_threshold=0.0)
        assert any(r.node_id == _TEST_NODE_UUID for r in results)

    async def test_high_threshold_excludes_irrelevant_result(self, vector_store: PGVectorStore):
        await vector_store.upsert(
            _TEST_NODE_ID,
            "Use PostgreSQL for transactional data",
            {"node_type": "decision", "confidence": 0.9},
        )
        results = await vector_store.search("PostgreSQL transactional", top_k=5, score_threshold=0.999)
        # A threshold of 0.999 is near-impossible to satisfy for real embeddings
        assert not any(r.node_id == _TEST_NODE_UUID for r in results)


@pytest.fixture
async def fresh_keyword_store(pg_test_url):
    store = _make_store(pg_test_url, "test_pagination")
    yield store
    await store._store.adelete_collection()


class TestPagination:
    async def test_top_k_limits_results(self, fresh_keyword_store: PGVectorStore):
        ids = [f"00000000-0000-0000-0000-{i:012d}" for i in range(1, 6)]
        for node_id in ids:
            await fresh_keyword_store.upsert(
                node_id,
                f"Decision about {_DISTINCTIVE_TERM} strategy {node_id}",
                {"node_type": "decision", "confidence": 0.9},
            )

        results = await fresh_keyword_store.search(_DISTINCTIVE_TERM, top_k=3, mode=SearchMode.KEYWORD)

        assert len(results) == 3


@pytest.fixture
async def empty_keyword_store(pg_test_url):
    store = _make_store(pg_test_url, "test_empty_collection")
    yield store
    await store._store.adelete_collection()


class TestEmptyCollection:
    async def test_search_on_fresh_store_returns_empty_not_error(self, empty_keyword_store: PGVectorStore):
        results = await empty_keyword_store.search("anything", top_k=5, mode=SearchMode.KEYWORD)
        assert results == []


class TestKeywordSearchFilters:
    async def test_node_type_filter_applied_in_keyword_mode(self, keyword_store: PGVectorStore):
        await keyword_store.upsert(
            _TEST_NODE_ID,
            f"Decision about {_DISTINCTIVE_TERM}",
            {"node_type": "decision", "confidence": 0.9, "job_id": "job-1"},
        )
        await keyword_store.upsert(
            _NODE_B_ID,
            f"Risk about {_DISTINCTIVE_TERM}",
            {"node_type": "risk", "confidence": 0.8, "job_id": "job-1"},
        )

        results = await keyword_store.search(
            _DISTINCTIVE_TERM,
            top_k=5,
            mode=SearchMode.KEYWORD,
            node_filter=NodeFilter(node_type=ConceptType.DECISION),
        )

        result_ids = {r.node_id for r in results}
        assert _TEST_NODE_UUID in result_ids
        assert _NODE_B_UUID not in result_ids

    async def test_exclude_job_id_removes_matching_nodes(self, keyword_store: PGVectorStore):
        await keyword_store.upsert(
            _TEST_NODE_ID,
            f"Decision about {_DISTINCTIVE_TERM}",
            {"node_type": "decision", "confidence": 0.9, "job_id": "job-exclude"},
        )
        await keyword_store.upsert(
            _NODE_B_ID,
            f"Another {_DISTINCTIVE_TERM} decision",
            {"node_type": "decision", "confidence": 0.9, "job_id": "job-keep"},
        )

        results = await keyword_store.search(
            _DISTINCTIVE_TERM,
            top_k=5,
            mode=SearchMode.KEYWORD,
            exclude_job_id="job-exclude",
        )

        result_ids = {r.node_id for r in results}
        assert _TEST_NODE_UUID not in result_ids
        assert _NODE_B_UUID in result_ids


class TestUpsertOverwrite:
    async def test_second_upsert_overwrites_first(self, keyword_store: PGVectorStore):
        await keyword_store.upsert(
            _TEST_NODE_ID,
            f"Original text about {_DISTINCTIVE_TERM}",
            {"node_type": "decision", "confidence": 0.9},
        )
        await keyword_store.upsert(
            _TEST_NODE_ID,
            "Completely different text about caching",
            {"node_type": "decision", "confidence": 0.9},
        )

        results = await keyword_store.search(_DISTINCTIVE_TERM, top_k=5, mode=SearchMode.KEYWORD)
        assert not any(r.node_id == _TEST_NODE_UUID for r in results)

        results = await keyword_store.search("caching", top_k=5, mode=SearchMode.KEYWORD)
        assert any(r.node_id == _TEST_NODE_UUID for r in results)
