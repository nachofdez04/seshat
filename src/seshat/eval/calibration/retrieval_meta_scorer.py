from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from seshat.app.platform.observability.usage_tracker import track_eval_usage
from seshat.eval.cache import build_cache_fp, read_or_run, sweep_stale_entries
from seshat.eval.calibration.models import RetrievalSweepPoint, RetrievalSweepResult
from seshat.eval.models import RetrievalScoredResult
from seshat.eval.retrieval.corpus_loader import load_corpus
from seshat.eval.retrieval.scorers import TOP_K

if TYPE_CHECKING:
    from seshat.app.pipeline.extraction.search_engine import SearchEngine
    from seshat.core.config.eval_settings import EvalConfig
    from seshat.core.config.settings import RAGConfig
    from seshat.infra.vector_store.base_store import AbstractVectorStore

type _Slug = str
type _ScoredResult = tuple[_Slug, float]  # (corpus slug, similarity score)
type _CacheEntry = tuple[list[_ScoredResult], list[_Slug]]  # (results desc by score, expected slugs)
type _Cache = dict[str, _CacheEntry]  # corpus_id → entry


class RetrievalMetaScorer:
    def __init__(
        self,
        search_engine: SearchEngine,
        vector_store: AbstractVectorStore,
        config: EvalConfig,
        rag_config: RAGConfig,
        step: float = 0.005,
    ) -> None:
        self._search_engine = search_engine
        self._vs = vector_store
        self._config = config
        self._rag_config = rag_config
        self._search_mode = rag_config.search_mode
        self._search_mode_hash = search_engine.fingerprint()
        self._step = step

    async def sweep_threshold(self) -> RetrievalSweepResult:
        """Load corpus results (file cache or vector store), then sweep thresholds [0, 1]."""
        cache = await self._build_cache()
        return self._compute_sweep(cache)

    def _compute_sweep(self, cache: _Cache) -> RetrievalSweepResult:
        n_points = round(1 / self._step) + 1
        thresholds = np.linspace(0.0, 1.0, n_points).tolist()
        points: list[RetrievalSweepPoint] = []

        for t in thresholds:
            recalls: list[float] = []
            precisions: list[float] = []
            scores: list[float] = []
            for _, (results, expected_ids) in cache.items():
                r, p, s = _compute_metrics(results, expected_ids, t)
                recalls.append(r)
                precisions.append(p)
                scores.append(s)
            points.append(
                RetrievalSweepPoint(
                    threshold=round(t, 5),
                    recall_at_5=round(sum(recalls) / len(recalls), 3),
                    precision_at_5=round(sum(precisions) / len(precisions), 3),
                    macro_f2=round(sum(scores) / len(scores), 3),
                )
            )

        best_idx = int(np.argmax([p.macro_f2 for p in points]))
        return RetrievalSweepResult(points=points, suggested_threshold=points[best_idx].threshold)

    @track_eval_usage("retrieval")
    async def _build_cache(self) -> _Cache:
        """Load scored results from the shared retrieval file cache; run vector store on miss."""
        from seshat.eval.retrieval.runner import RetrievalEvalRunner

        examples = load_corpus(self._config.retrieval_corpus_dir)
        runner = RetrievalEvalRunner(
            search_engine=self._search_engine,
            vector_store=self._vs,
            config=self._config,
            rag_config=self._rag_config,
        )
        cache: _Cache = {}
        touched = set()

        for ex in examples:
            cache_fp = build_cache_fp(self._config.retrieval_cache_dir, ex, agent_hash=self._search_mode_hash)
            scored, used, _cached = await read_or_run(
                cache_fp,
                RetrievalScoredResult,
                runner._fetch_example(ex),
            )
            cache[ex.corpus_id] = (list(scored.results), list(ex.expected_relevant_ids))
            touched.add(used)

        sweep_stale_entries(
            self._config.retrieval_cache_dir,
            corpus_ids=[ex.corpus_id for ex in examples],
            touched=touched,
            agent_hash=self._search_mode_hash,
        )
        return cache


def _compute_metrics(
    results: list[_ScoredResult],
    expected_ids: list[_Slug],
    threshold: float,
) -> tuple[float, float, float]:
    """Return (recall_at_5, precision_at_5, score) for one corpus example at one threshold.

    score is F2 for positive examples and specificity for negative examples.
    Both are in [0, 1] and feed the macro_f2 average on RetrievalSweepPoint.
    """
    filtered = [slug for slug, score in results if score >= threshold][:TOP_K]
    returned_ids = set(filtered)

    if not expected_ids:
        # Negative example: specificity = 1 if nothing returned, 0 otherwise.
        specificity = 0.0 if returned_ids else 1.0
        return 0.0, 0.0, specificity

    tp = len(returned_ids & set(expected_ids))
    if not filtered:
        return 0.0, 0.0, 0.0

    recall = tp / len(expected_ids)
    precision = tp / TOP_K
    # F2 weights recall twice as heavily as precision: beta=2
    f2 = (5 * precision * recall) / (4 * precision + recall) if (precision + recall) > 0 else 0.0
    return recall, precision, f2
