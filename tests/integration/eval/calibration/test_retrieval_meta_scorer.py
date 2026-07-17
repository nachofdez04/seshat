from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from seshat.app.pipeline.extraction.search_engine import SearchEngine
from seshat.core.config.settings import RAGConfig
from seshat.core.models.api_graph import NodeFilter, SearchResult
from seshat.eval.calibration.retrieval_meta_scorer import RetrievalMetaScorer
from seshat.infra.vector_store.base_store import AbstractVectorStore
from tests.integration.eval.helpers import make_eval_config

if TYPE_CHECKING:
    from pathlib import Path

    from seshat.core.config.eval_settings import EvalConfig


pytestmark = [pytest.mark.integration, pytest.mark.eval]


class _StubVectorStore(AbstractVectorStore):
    """Echoes upserted node ids back as search results with a fixed score."""

    def __init__(self) -> None:
        self._stored: list[str] = []

    @staticmethod
    def get_supported_filter_fields() -> frozenset[str]:
        return frozenset({"node_type"})

    async def upsert(self, node_id: str, text: str, metadata: dict) -> None:
        self._stored.append(node_id)

    async def search_dense(
        self,
        query: str,
        top_k: int,
        node_filter: NodeFilter | None = None,
        exclude_job_id: str | None = None,
        score_threshold: float | None = None,
    ) -> list[SearchResult]:
        results = [SearchResult(node_id=nid, score=0.8) for nid in self._stored[:top_k]]
        self._stored.clear()
        return results

    async def search_sparse(
        self,
        query: str,
        top_k: int,
        node_filter: NodeFilter | None = None,
        exclude_job_id: str | None = None,
    ) -> list[SearchResult]:
        results = [SearchResult(node_id=nid, score=0.8) for nid in self._stored[:top_k]]
        self._stored.clear()
        return results

    async def delete(self, node_id: str) -> None:
        self._stored = [nid for nid in self._stored if nid != node_id]

    async def update_metadata(self, node_id: str, patch: dict) -> None:
        pass


def _make_scorer(eval_config: EvalConfig) -> RetrievalMetaScorer:
    rag_config = RAGConfig()
    stub_vs = _StubVectorStore()
    search_engine = SearchEngine(
        rag_config=rag_config,
        vector_store=stub_vs,
        keyword_llm=None,
        multi_query_llm=None,
    )
    return RetrievalMetaScorer(
        search_engine=search_engine,
        vector_store=stub_vs,
        config=eval_config,
        rag_config=rag_config,
        step=0.1,
    )


class TestRetrievalMetaScorerIntegration:
    async def test_sweep_end_to_end(self, tmp_path: Path) -> None:
        """Real corpus loader + stub vector store; verifies the full sweep path runs
        without errors and produces one result point per step."""
        eval_config = make_eval_config(tmp_path, "seshat-retrieval-meta-scorer")
        scorer = _make_scorer(eval_config)

        result = await scorer.sweep_threshold()

        corpus_files = list(eval_config.retrieval_corpus_dir.glob("*.yaml"))
        assert len(result.points) == 11  # step=0.1 → 0.0, 0.1, …, 1.0
        assert len(corpus_files) > 0
        assert result.suggested_threshold is not None
