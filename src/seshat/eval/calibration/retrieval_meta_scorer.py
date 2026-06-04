from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import numpy as np

from seshat.eval.calibration.models import RetrievalSweepPoint, RetrievalSweepResult
from seshat.eval.retrieval.corpus_loader import build_kb_nodes, load_corpus
from seshat.models.api import NodeFilter, SearchResult

if TYPE_CHECKING:
    from seshat.config.settings import EvalConfig
    from seshat.models.nodes import KBNode
    from seshat.vector_store.base_store import AbstractVectorStore

# corpus_id → (results sorted desc by score, expected_ids)
type _Cache = dict[str, tuple[list[SearchResult], list[str]]]

_TOP_K = 5


class RetrievalMetaScorer:
    def __init__(
        self,
        vector_store: AbstractVectorStore,
        config: EvalConfig,
        step: float = 0.01,
    ) -> None:
        self._vs = vector_store
        self._config = config
        self._step = step
        self._cache: _Cache | None = None

    async def build_cache(self) -> None:
        """Seed each corpus example once; store all results with score_threshold=None."""
        examples = load_corpus(self._config.retrieval_corpus_dir)
        cache: _Cache = {}

        for ex in examples:
            query_node, candidate_kb_nodes, slug_map = build_kb_nodes(ex)
            await self._seed(candidate_kb_nodes)
            try:
                query = f"{query_node.title} {query_node.description}"
                node_filter = NodeFilter(node_type=query_node.type)
                results = await self._vs.search(
                    query,
                    top_k=len(candidate_kb_nodes),
                    node_filter=node_filter,
                    score_threshold=None,
                )
                uuid_to_slug = {str(v): k for k, v in slug_map.items()}
                slug_results = [
                    SearchResult(node_id=uuid_to_slug.get(r.node_id, r.node_id), score=r.score) for r in results
                ]
                cache[ex.corpus_id] = (slug_results, list(ex.expected_relevant_ids))
            finally:
                await self._teardown(candidate_kb_nodes)

        self._cache = cache

    def sweep_threshold(self) -> RetrievalSweepResult:
        """Replay threshold cutoffs [0, 1] at step intervals against cached results."""
        if self._cache is None:
            raise RuntimeError("build_cache() must be called before sweep_threshold()")

        n_points = round(1 / self._step) + 1
        thresholds = np.linspace(0.0, 1.0, n_points).tolist()
        points: list[RetrievalSweepPoint] = []

        for t in thresholds:
            recalls: list[float] = []
            precisions: list[float] = []
            for _, (results, expected_ids) in self._cache.items():
                r, p = _compute_metrics(results, expected_ids, t)
                recalls.append(r)
                precisions.append(p)
            points.append(
                RetrievalSweepPoint(
                    threshold=round(t, 10),
                    recall_at_5=sum(recalls) / len(recalls),
                    precision_at_5=sum(precisions) / len(precisions),
                )
            )

        # argmax recall; ties → lower threshold (np.argmax returns first occurrence, grid is ascending)
        best_idx = int(np.argmax([p.recall_at_5 for p in points]))
        return RetrievalSweepResult(points=points, suggested_threshold=points[best_idx].threshold)

    async def _seed(self, nodes: list[KBNode]) -> None:
        async def _upsert(node: KBNode) -> None:
            metadata = {"node_type": node.type.value, "confidence": node.confidence}
            await self._vs.upsert(str(node.id), text=f"{node.title} {node.description}", metadata=metadata)

        await asyncio.gather(*(_upsert(n) for n in nodes))

    async def _teardown(self, nodes: list[KBNode]) -> None:
        await asyncio.gather(*(self._vs.delete(str(n.id)) for n in nodes))


def _compute_metrics(
    results: list[SearchResult],
    expected_ids: list[str],
    threshold: float,
) -> tuple[float, float]:
    """Return (recall_at_5, precision_at_5) for one corpus example at one threshold."""
    filtered = [r for r in results if r.score >= threshold][:_TOP_K]
    returned_ids = {r.node_id for r in filtered}

    if not expected_ids:
        # Negative example
        if returned_ids:
            return 1.0, 0.0
        return 1.0, 1.0

    tp = len(returned_ids & set(expected_ids))
    if not filtered:
        return 0.0, 0.0
    return tp / len(expected_ids), tp / _TOP_K
