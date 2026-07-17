from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

import pytest

from seshat.app.pipeline.extraction.search_engine import (
    SearchEngine,
    _Keywords,
    _QueryVariants,
    _rrf,
)
from seshat.core.config.settings import MultiQueryConfig, RAGConfig, _LLMConfig
from seshat.core.models.api_graph import NodeFilter, SearchResult
from seshat.core.models.enums import LLMProvider, SearchMode
from tests.helpers import make_structured_llm

_N1 = "00000000-0000-0000-0000-000000000001"
_N2 = "00000000-0000-0000-0000-000000000002"
_N3 = "00000000-0000-0000-0000-000000000003"

_LLM_CFG = _LLMConfig(provider=LLMProvider.OPENAI, model="gpt-4o-mini")


def _search_result(node_id: str, score: float = 0.9) -> SearchResult:
    return SearchResult(node_id=UUID(node_id), score=score)


def _make_vs(dense_mock: AsyncMock | None = None, sparse_mock: AsyncMock | None = None) -> MagicMock:
    vs = MagicMock()
    vs.search_dense = dense_mock or AsyncMock(return_value=[])
    vs.search_sparse = sparse_mock or AsyncMock(return_value=[])
    return vs


def _make_engine(
    dense_mock=None,
    sparse_mock=None,
    keyword_llm=None,
    multi_query_llm=None,
    search_mode=SearchMode.SEMANTIC,
    rag_config=None,
):
    if rag_config is None:
        rag_kwargs: dict = {"search_mode": search_mode}
        if keyword_llm is not None:
            rag_kwargs["keyword_extraction_llm"] = _LLM_CFG
        if multi_query_llm is not None:
            rag_kwargs["multi_query"] = MultiQueryConfig(llm=_LLM_CFG, num_variants=3)
        rag_config = RAGConfig(**rag_kwargs)
    return SearchEngine(
        rag_config=rag_config,
        vector_store=_make_vs(dense_mock, sparse_mock),
        keyword_llm=keyword_llm,
        multi_query_llm=multi_query_llm,
    )


class TestRrf:
    def test_node_in_both_legs_scores_higher(self):
        dense = [[_search_result(_N1, 0.9), _search_result(_N2, 0.8)]]
        sparse = [[_search_result(_N1, 0.5), _search_result(_N3, 0.4)]]
        results = _rrf(dense, sparse)
        assert results[0].node_id == UUID(_N1)

    def test_rrf_returns_all_nodes(self):
        uids = [f"00000000-0000-0000-0000-00000000000{i}" for i in range(4)]
        dense = [[_search_result(uid) for uid in uids]]
        results = _rrf(dense, [])
        assert len(results) == 4

    def test_rrf_empty_returns_empty(self):
        assert _rrf([], []) == []

    def test_duplicate_node_id_appears_once(self):
        # same node at two ranks in one leg — should appear once in output
        results = _rrf([[_search_result(_N1, 0.9), _search_result(_N1, 0.8)]], [])
        assert len([r for r in results if r.node_id == UUID(_N1)]) == 1


class TestSearchEngineSemantic:
    async def test_delegates_to_search_dense(self):
        expected = [_search_result(_N1)]
        dense_mock = AsyncMock(return_value=expected)
        engine = _make_engine(dense_mock=dense_mock, search_mode=SearchMode.SEMANTIC)
        result = await engine.search("query")
        assert result == expected
        dense_mock.assert_awaited_once()

    async def test_passes_node_filter_and_exclude_job_id(self):
        dense_mock = AsyncMock(return_value=[])
        engine = _make_engine(dense_mock=dense_mock, search_mode=SearchMode.SEMANTIC)
        nf = NodeFilter()
        await engine.search("q", node_filter=nf, exclude_job_id="job-1")
        _, kwargs = dense_mock.call_args
        assert kwargs["node_filter"] is nf
        assert kwargs["exclude_job_id"] == "job-1"


class TestSearchEngineKeyword:
    async def test_keyword_llm_called_before_search(self):
        keyword_llm = make_structured_llm(return_value=_Keywords(keywords=["redis", "cache"]))
        sparse_mock = AsyncMock(return_value=[_search_result(_N1)])
        engine = _make_engine(sparse_mock=sparse_mock, keyword_llm=keyword_llm, search_mode=SearchMode.KEYWORD)
        await engine.search("Use Redis for caching sessions")
        keyword_llm.with_structured_output.assert_called_once()
        sparse_mock.assert_awaited_once()

    async def test_no_keyword_llm_passes_query_directly(self):
        sparse_mock = AsyncMock(return_value=[])
        engine = _make_engine(sparse_mock=sparse_mock, search_mode=SearchMode.KEYWORD)
        await engine.search("some query")
        args, kwargs = sparse_mock.call_args
        assert kwargs.get("query", args[0] if args else None) == "some query"


class TestSearchEngineMultiQuery:
    async def test_multi_query_fans_out_and_fuses(self):
        multi_query_llm = make_structured_llm(
            return_value=_QueryVariants(variants=["variant 1", "variant 2", "variant 3"])
        )
        dense_mock = AsyncMock(side_effect=[[_search_result(_N1)], [_search_result(_N2)], [], []])
        rag = RAGConfig(
            search_mode=SearchMode.SEMANTIC,
            multi_query=MultiQueryConfig(llm=_LLM_CFG, num_variants=3),
        )
        engine = SearchEngine(
            rag_config=rag,
            vector_store=_make_vs(dense_mock=dense_mock),
            keyword_llm=None,
            multi_query_llm=multi_query_llm,
        )
        results = await engine.search("query about caching")
        assert dense_mock.call_count == 4  # original + 3 variants
        node_ids = {r.node_id for r in results}
        assert UUID(_N1) in node_ids
        assert UUID(_N2) in node_ids

    async def test_multi_query_llm_failure_falls_back_to_single_query(self, caplog):
        multi_query_llm = make_structured_llm(side_effect=RuntimeError("llm down"))
        dense_mock = AsyncMock(return_value=[_search_result(_N1)])
        rag = RAGConfig(
            search_mode=SearchMode.SEMANTIC,
            multi_query=MultiQueryConfig(llm=_LLM_CFG),
        )
        engine = SearchEngine(
            rag_config=rag,
            vector_store=_make_vs(dense_mock=dense_mock),
            keyword_llm=None,
            multi_query_llm=multi_query_llm,
        )
        with caplog.at_level(logging.WARNING, logger="seshat.app.pipeline.extraction.search_engine"):
            results = await engine.search("fallback query")

        assert dense_mock.call_count == 1
        assert any("multi" in r.message.lower() or "fallback" in r.message.lower() for r in caplog.records)
        assert results == [_search_result(_N1)]


class TestSearchEngineAgentMode:
    async def test_agent_mode_raises_value_error(self):
        engine = _make_engine(search_mode=SearchMode.AGENT)
        with pytest.raises(ValueError, match="agent"):
            await engine.search("query")


class TestSearchEngineFingerprint:
    def test_fingerprint_is_stable(self):
        engine = _make_engine(search_mode=SearchMode.SEMANTIC)
        assert engine.fingerprint() == engine.fingerprint()

    def test_different_modes_give_different_fingerprints(self):
        e1 = _make_engine(search_mode=SearchMode.SEMANTIC)
        e2 = _make_engine(search_mode=SearchMode.KEYWORD)
        assert e1.fingerprint() != e2.fingerprint()
