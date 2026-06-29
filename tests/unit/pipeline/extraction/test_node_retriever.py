import logging
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from seshat.config.settings import RAGConfig
from seshat.models.api_graph import NodeFilter, SearchResult
from seshat.models.enums import ConceptType, NodeStatus
from seshat.pipeline.extraction.node_retriever import NodeRetriever
from tests.helpers import make_node


def _make_retriever(
    search_results=None,
    kb_nodes=None,
    neighbour_nodes=None,
    top_k=3,
    max_context_tokens=None,
    search_mode=None,
    reranker=None,
):

    rag_kwargs: dict[str, Any] = {"top_k": top_k}
    if max_context_tokens is not None:
        rag_kwargs["max_context_tokens"] = max_context_tokens
    if search_mode is not None:
        rag_kwargs["search_mode"] = search_mode
    rag_config = RAGConfig(**rag_kwargs)

    vector_store = MagicMock()
    vector_store.search = AsyncMock(return_value=search_results or [])

    kb_store = MagicMock()
    kb_store.get_node = AsyncMock(side_effect=lambda nid: next((n for n in (kb_nodes or []) if str(n.id) == nid), None))
    kb_store.get_neighbours = AsyncMock(return_value=neighbour_nodes or [])

    return NodeRetriever(rag_config=rag_config, kb_store=kb_store, vector_store=vector_store, reranker=reranker)


class TestNodeRetriever:
    async def test_returns_matched_kb_nodes(self):
        candidate = make_node("n2", title="Use Redis")
        search_results = [SearchResult(node_id=str(candidate.id), score=0.9)]
        retriever = _make_retriever(search_results=search_results, kb_nodes=[candidate])

        source = make_node("n1")
        result = await retriever.retrieve(source)
        assert any(n.id == candidate.id for n in result)

    async def test_excludes_source_node_from_results(self):
        source = make_node("n1")
        search_results = [SearchResult(node_id=str(source.id), score=0.99)]
        retriever = _make_retriever(search_results=search_results, kb_nodes=[source])

        result = await retriever.retrieve(source)
        assert not any(n.id == source.id for n in result)

    async def test_includes_neighbours(self):
        candidate = make_node("n2", title="Use Redis")
        neighbour = make_node("n3", title="Redis Caching Decision")
        search_results = [SearchResult(node_id=str(candidate.id), score=0.9)]
        retriever = _make_retriever(
            search_results=search_results,
            kb_nodes=[candidate],
            neighbour_nodes=[neighbour],
        )

        source = make_node("n1")
        result = await retriever.retrieve(source)
        assert any(n.id == neighbour.id for n in result)

    async def test_caller_node_filter_overrides_default_status(self):
        retriever = _make_retriever()
        source = make_node("n1")
        override = NodeFilter(status=NodeStatus.PENDING_REVIEW)

        await retriever.retrieve(source, node_filter=override)

        call_kwargs = retriever._vs.search.call_args.kwargs
        node_filter: NodeFilter = call_kwargs["node_filter"]
        assert node_filter.status == NodeStatus.PENDING_REVIEW

    async def test_orphan_vector_result_is_silently_skipped(self):
        # vector store returns a hit, but the KB has no matching node
        orphan_id = str(make_node("orphan").id)
        search_results = [SearchResult(node_id=orphan_id, score=0.9)]
        retriever = _make_retriever(search_results=search_results, kb_nodes=[])

        source = make_node("n1")
        result = await retriever.retrieve(source)

        assert result == []

    async def test_exclude_job_id_forwarded_to_vector_search(self):
        retriever = _make_retriever()
        source = make_node("n1")

        await retriever.retrieve(source, exclude_job_id="job-42")

        call_kwargs = retriever._vs.search.call_args.kwargs
        assert call_kwargs["exclude_job_id"] == "job-42"

    async def test_fetch_loop_stops_at_cap_without_fetching_remaining_results(self):
        # top_k=1 → cap=2; three vector hits — only 2 KB fetches should happen
        candidates = [make_node(f"n{i}", title=f"Node {i}") for i in range(2, 5)]
        search_results = [SearchResult(node_id=str(c.id), score=0.9) for c in candidates]
        retriever = _make_retriever(search_results=search_results, kb_nodes=candidates, top_k=1)

        await retriever.retrieve(make_node("n1"))

        assert retriever._kb.get_node.call_count == 2

    async def test_token_budget_stops_fetch_before_top_k_cap(self):
        # each node costs ~9 tokens (title + description); budget of 18 allows 2 nodes
        # top_k=10 → cap=20, so token budget is the binding constraint here
        candidates = [make_node(f"n{i}", title=f"Node {i}") for i in range(2, 6)]
        search_results = [SearchResult(node_id=str(c.id), score=0.9) for c in candidates]
        retriever = _make_retriever(
            search_results=search_results,
            kb_nodes=candidates,
            top_k=10,
            max_context_tokens=18,
        )

        result = await retriever.retrieve(make_node("n1"))

        assert len(result) == 2
        assert retriever._kb.get_node.call_count == 2

    async def test_cap_limits_neighbour_expansion(self):
        # top_k=1 → cap=2; one vector hit plus three neighbours — only 2 total should be kept
        candidate = make_node("n2", title="Use Redis")
        neighbours = [make_node(f"n{i}", title=f"Neighbour {i}") for i in range(3, 7)]
        search_results = [SearchResult(node_id=str(candidate.id), score=0.9)]
        retriever = _make_retriever(
            search_results=search_results,
            kb_nodes=[candidate],
            neighbour_nodes=neighbours,
            top_k=1,
        )

        source = make_node("n1")
        result = await retriever.retrieve(source)

        assert len(result) <= 2  # cap = top_k * 2

    @pytest.mark.parametrize(
        ("override_filter", "expected_type"),
        [
            (None, ConceptType.DECISION),  # default: uses source node type
            (NodeFilter(node_type=ConceptType.RISK), ConceptType.RISK),  # explicit override
            (NodeFilter(node_type=None), None),  # explicit None override
        ],
    )
    async def test_node_type_forwarded_to_vector_search(self, override_filter, expected_type):
        retriever = _make_retriever()
        source = make_node("n1", type=ConceptType.DECISION)

        await retriever.retrieve(source, node_filter=override_filter)

        call_kwargs = retriever._vs.search.call_args.kwargs
        node_filter = call_kwargs["node_filter"]
        assert node_filter.node_type == expected_type

    async def test_duplicate_vector_result_logged_and_skipped(self, caplog):
        candidate = make_node("n2", title="Use Redis")
        dup_id = str(candidate.id)
        search_results = [
            SearchResult(node_id=dup_id, score=0.9),
            SearchResult(node_id=dup_id, score=0.8),  # duplicate
        ]
        retriever = _make_retriever(search_results=search_results, kb_nodes=[candidate])

        with caplog.at_level(logging.WARNING, logger="seshat.pipeline.extraction.node_retriever"):
            result = await retriever.retrieve(make_node("n1"))

        ids = [n.id for n in result]
        assert ids.count(candidate.id) == 1
        assert any("Duplicate" in r.message for r in caplog.records)

    async def test_source_node_excluded_when_appearing_as_neighbour(self):
        source = make_node("n1")
        candidate = make_node("n2", title="Use Redis")
        search_results = [SearchResult(node_id=str(candidate.id), score=0.9)]
        retriever = _make_retriever(
            search_results=search_results,
            kb_nodes=[candidate],
            neighbour_nodes=[source],
        )

        result = await retriever.retrieve(source)

        assert not any(n.id == source.id for n in result)

    async def test_over_budget_neighbour_does_not_block_cheaper_sibling(self):
        # fat_neighbour has a 200-char title (~55 tokens); thin_neighbour is tiny (~8).
        # budget=20 (hard cap=22): candidate consumes ~8 tokens as a direct hit,
        # leaving ~14 under the hard cap — enough for thin but not fat.
        candidate = make_node("n2", title="A")
        fat_neighbour = make_node("n3", title="B" * 200)
        thin_neighbour = make_node("n4", title="C")
        search_results = [SearchResult(node_id=str(candidate.id), score=0.9)]
        retriever = _make_retriever(
            search_results=search_results,
            kb_nodes=[candidate],
            neighbour_nodes=[fat_neighbour, thin_neighbour],
            top_k=10,
            max_context_tokens=20,
        )

        result = await retriever.retrieve(make_node("n1"))

        ids = {n.id for n in result}
        assert thin_neighbour.id in ids
        assert fat_neighbour.id not in ids

    async def test_exhausted_budget_skips_get_neighbours_call(self):
        candidate = make_node("n2", title="Use Redis")
        search_results = [SearchResult(node_id=str(candidate.id), score=0.9)]
        retriever = _make_retriever(
            search_results=search_results,
            kb_nodes=[candidate],
            top_k=10,
            max_context_tokens=1,
        )

        await retriever.retrieve(make_node("n1"))

        assert retriever._kb.get_neighbours.call_count == 0

    async def test_no_duplicates_in_result(self):
        candidate = make_node("n2", title="Use Redis")
        search_results = [SearchResult(node_id=str(candidate.id), score=0.9)]
        # neighbour is the same node as candidate — should not appear twice
        retriever = _make_retriever(
            search_results=search_results,
            kb_nodes=[candidate],
            neighbour_nodes=[candidate],
        )

        source = make_node("n1")
        result = await retriever.retrieve(source)
        ids = [n.id for n in result]
        assert len(ids) == len(set(ids))

    async def test_search_mode_forwarded_to_vector_search(self):
        from seshat.models.enums import SearchMode

        retriever = _make_retriever(search_mode=SearchMode.HYBRID)
        await retriever.retrieve(make_node("n1"))
        call_kwargs = retriever._vs.search.call_args.kwargs
        assert call_kwargs["mode"] == SearchMode.HYBRID

    async def test_search_mode_defaults_semantic(self):
        from seshat.models.enums import SearchMode

        retriever = _make_retriever()
        await retriever.retrieve(make_node("n1"))
        call_kwargs = retriever._vs.search.call_args.kwargs
        assert call_kwargs["mode"] == SearchMode.SEMANTIC

    async def test_reranker_called_with_query_and_results(self):
        candidate = make_node("n2", title="Use Kafka")
        search_results = [SearchResult(node_id=str(candidate.id), score=0.9)]
        reranker = MagicMock()
        reranker.rerank = AsyncMock(return_value=search_results)
        retriever = _make_retriever(search_results=search_results, kb_nodes=[candidate], reranker=reranker)

        source = make_node("n1", title="Kafka consumer lag")
        await retriever.retrieve(source)

        reranker.rerank.assert_awaited_once()
        called_query = reranker.rerank.call_args.args[0]
        assert "kafka" in called_query.lower()

    async def test_reranker_result_used_by_fetch_direct_hits(self):
        n2 = make_node("n2", title="Kafka consumer")
        n3 = make_node("n3", title="Kafka topic")
        search_results = [
            SearchResult(node_id=str(n2.id), score=0.9),
            SearchResult(node_id=str(n3.id), score=0.8),
        ]
        reranked = [
            SearchResult(node_id=str(n3.id), score=0.95),
            SearchResult(node_id=str(n2.id), score=0.85),
        ]
        reranker = MagicMock()
        reranker.rerank = AsyncMock(return_value=reranked)
        retriever = _make_retriever(
            search_results=search_results,
            kb_nodes=[n2, n3],
            reranker=reranker,
            top_k=1,
        )

        result = await retriever.retrieve(make_node("n1"))

        assert result[0].id == n3.id

    async def test_reranker_none_skipped(self):
        retriever = _make_retriever()
        assert retriever._reranker is None
