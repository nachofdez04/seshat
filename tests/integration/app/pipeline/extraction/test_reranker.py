from __future__ import annotations

import os

import pytest

from seshat.app.pipeline.extraction.reranker import CohereReranker, VoyageReranker
from seshat.core.config.settings import RerankerConfig
from seshat.core.models.enums import RerankerProvider
from tests.helpers import make_node
from tests.integration.conftest import SKIP_IF_NO_COHERE_API, SKIP_IF_NO_VOYAGE_API

pytestmark = [pytest.mark.integration, pytest.mark.llm, pytest.mark.reranker]


def _redis_nodes():
    return [
        make_node("n1", title="PostgreSQL schema migration approach"),
        make_node("n2", title="Redis cache eviction policy decision"),
        make_node("n3", title="Redis memory limit configuration"),
    ]


@SKIP_IF_NO_COHERE_API
class TestCohereReranker:
    async def test_orders_by_relevance(self):
        cfg = RerankerConfig(provider=RerankerProvider.COHERE, model="rerank-v3.5")
        reranker = CohereReranker(cfg, os.environ["COHERE_API_KEY"])

        result = await reranker.rerank("Redis memory management", _redis_nodes())

        assert len(result) == 3
        assert result[0].title in {"Redis cache eviction policy decision", "Redis memory limit configuration"}

    async def test_top_n_truncates_after_reranking(self):
        cfg = RerankerConfig(provider=RerankerProvider.COHERE, model="rerank-v3.5", top_n=2)
        reranker = CohereReranker(cfg, os.environ["COHERE_API_KEY"])

        result = await reranker.rerank("Redis memory management", _redis_nodes())

        assert len(result) == 2


@SKIP_IF_NO_VOYAGE_API
class TestVoyageReranker:
    async def test_orders_by_relevance(self):
        cfg = RerankerConfig(provider=RerankerProvider.VOYAGE, model="rerank-2")
        reranker = VoyageReranker(cfg, os.environ["VOYAGE_API_KEY"])

        result = await reranker.rerank("Redis memory management", _redis_nodes())

        assert len(result) == 3
        assert result[0].title in {"Redis cache eviction policy decision", "Redis memory limit configuration"}

    async def test_top_n_truncates_after_reranking(self):
        cfg = RerankerConfig(provider=RerankerProvider.VOYAGE, model="rerank-2", top_n=2)
        reranker = VoyageReranker(cfg, os.environ["VOYAGE_API_KEY"])

        result = await reranker.rerank("Redis memory management", _redis_nodes())

        assert len(result) == 2
