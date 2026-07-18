from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from seshat.app.pipeline.extraction.reranker import AbstractReranker, CohereReranker, VoyageReranker, reranker_factory
from seshat.app.platform.observability.usage_tracker import (
    TokenBudgetCallback,
    UsageTracker,
    set_run_tracker,
)
from seshat.core.config.settings import RerankerConfig
from seshat.core.models.enums import RerankerProvider
from tests.helpers import make_node


class _FakeReranker(AbstractReranker):
    async def _rerank(self, query, nodes):
        return nodes

    async def ping(self):
        pass


def _nodes(*seeds: str):
    return [make_node(s) for s in seeds]


class TestAbstractRerankerTopN:
    async def test_top_n_none_returns_all(self):
        cfg = RerankerConfig(provider=RerankerProvider.COHERE, model="x", top_n=None)
        out = await _FakeReranker(cfg, "key").rerank("q", _nodes("a", "b", "c"))
        assert len(out) == 3

    async def test_top_n_truncates(self):
        cfg = RerankerConfig(provider=RerankerProvider.COHERE, model="x", top_n=2)
        out = await _FakeReranker(cfg, "key").rerank("q", _nodes("a", "b", "c"))
        assert len(out) == 2


class TestBuildReranker:
    def test_cohere_returns_cohere_reranker(self):
        cfg = RerankerConfig(provider=RerankerProvider.COHERE, model="rerank-v3.5")
        assert isinstance(reranker_factory(cfg, "key"), CohereReranker)

    def test_voyage_returns_voyage_reranker(self):
        cfg = RerankerConfig(provider=RerankerProvider.VOYAGE, model="rerank-2")
        assert isinstance(reranker_factory(cfg, "key"), VoyageReranker)


def _cohere_response(nodes, input_tokens: int):
    result = MagicMock()
    result.results = [MagicMock(index=i) for i in range(len(nodes))]
    result.meta = MagicMock()
    result.meta.tokens = MagicMock()
    result.meta.tokens.input_tokens = input_tokens
    return result


def _voyage_response(nodes, total_tokens: int):
    result = MagicMock()
    result.results = [MagicMock(index=i) for i in range(len(nodes))]
    result.total_tokens = total_tokens
    return result


class TestCohereRerankerUsageTracking:
    async def test_records_reranker_tokens_when_tracker_active(self):
        nodes = _nodes("a", "b", "c")
        cfg = RerankerConfig(provider=RerankerProvider.COHERE, model="rerank-v3.5")

        tracker = UsageTracker(max_input_tokens=10_000, max_output_tokens=10_000)
        set_run_tracker(TokenBudgetCallback(tracker))

        mock_response = _cohere_response(nodes, input_tokens=150)
        with patch("cohere.AsyncClientV2") as mock_cls:
            mock_client = AsyncMock()
            mock_client.rerank = AsyncMock(return_value=mock_response)
            mock_cls.return_value = mock_client
            reranker = CohereReranker(cfg, "key")
            await reranker.rerank("query", nodes)

        assert tracker.reranker_input_tokens == 150

    async def test_skips_tracking_when_no_tracker_set(self):
        nodes = _nodes("a", "b")
        cfg = RerankerConfig(provider=RerankerProvider.COHERE, model="rerank-v3.5")

        set_run_tracker(None)  # type: ignore[arg-type]

        mock_response = _cohere_response(nodes, input_tokens=80)
        with patch("cohere.AsyncClientV2") as mock_cls:
            mock_client = AsyncMock()
            mock_client.rerank = AsyncMock(return_value=mock_response)
            mock_cls.return_value = mock_client
            reranker = CohereReranker(cfg, "key")
            await reranker.rerank("query", nodes)  # should not raise


class TestCohereRerankerPing:
    async def test_reachable_does_not_raise(self):
        cfg = RerankerConfig(provider=RerankerProvider.COHERE, model="rerank-v3.5")
        with patch("cohere.AsyncClientV2") as mock_cls:
            mock_cls.return_value.rerank = AsyncMock()
            reranker = CohereReranker(cfg, "key")
            await reranker.ping()  # must not raise
        mock_cls.return_value.rerank.assert_awaited_once_with(
            model="rerank-v3.5", query="ping", documents=["ping"], top_n=1
        )

    async def test_unreachable_raises(self):
        cfg = RerankerConfig(provider=RerankerProvider.COHERE, model="rerank-v3.5")
        with patch("cohere.AsyncClientV2") as mock_cls:
            mock_cls.return_value.rerank = AsyncMock(side_effect=RuntimeError("unreachable"))
            reranker = CohereReranker(cfg, "key")
            with pytest.raises(RuntimeError, match="unreachable"):
                await reranker.ping()


class TestVoyageRerankerPing:
    async def test_reachable_does_not_raise(self):
        cfg = RerankerConfig(provider=RerankerProvider.VOYAGE, model="rerank-2")
        with patch("voyageai.AsyncClient") as mock_cls:
            mock_cls.return_value.rerank = AsyncMock()
            reranker = VoyageReranker(cfg, "key")
            await reranker.ping()  # must not raise
        mock_cls.return_value.rerank.assert_awaited_once_with(
            query="ping", documents=["ping"], model="rerank-2", top_k=1
        )

    async def test_unreachable_raises(self):
        cfg = RerankerConfig(provider=RerankerProvider.VOYAGE, model="rerank-2")
        with patch("voyageai.AsyncClient") as mock_cls:
            mock_cls.return_value.rerank = AsyncMock(side_effect=RuntimeError("unreachable"))
            reranker = VoyageReranker(cfg, "key")
            with pytest.raises(RuntimeError, match="unreachable"):
                await reranker.ping()


class TestVoyageRerankerUsageTracking:
    async def test_records_reranker_tokens_when_tracker_active(self):
        nodes = _nodes("a", "b", "c")
        cfg = RerankerConfig(provider=RerankerProvider.VOYAGE, model="rerank-2")

        tracker = UsageTracker(max_input_tokens=10_000, max_output_tokens=10_000)
        set_run_tracker(TokenBudgetCallback(tracker))

        mock_response = _voyage_response(nodes, total_tokens=200)
        with patch("voyageai.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.rerank = AsyncMock(return_value=mock_response)
            mock_cls.return_value = mock_client
            reranker = VoyageReranker(cfg, "key")
            await reranker.rerank("query", nodes)

        assert tracker.reranker_input_tokens == 200

    async def test_skips_tracking_when_no_tracker_set(self):
        nodes = _nodes("a", "b")
        cfg = RerankerConfig(provider=RerankerProvider.VOYAGE, model="rerank-2")

        set_run_tracker(None)  # type: ignore[arg-type]

        mock_response = _voyage_response(nodes, total_tokens=60)
        with patch("voyageai.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.rerank = AsyncMock(return_value=mock_response)
            mock_cls.return_value = mock_client
            reranker = VoyageReranker(cfg, "key")
            await reranker.rerank("query", nodes)  # should not raise
