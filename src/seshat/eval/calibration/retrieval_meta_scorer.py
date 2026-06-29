from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from seshat.eval.cache import build_cache_fp, read_or_run, sweep_stale_entries
from seshat.eval.calibration.models import RetrievalSweepPoint, RetrievalSweepResult
from seshat.eval.models import RetrievalScoredResult
from seshat.eval.retrieval.corpus_loader import load_corpus
from seshat.eval.retrieval.scorers import TOP_K
from seshat.models.api_graph import SearchResult
from seshat.models.enums import SearchMode
from seshat.observability.usage_tracker import track_eval_usage
from seshat.utils.hashing import fingerprint

if TYPE_CHECKING:
    from seshat.config.eval_settings import EvalConfig
    from seshat.vector_store.base_store import AbstractVectorStore

# corpus_id → (results sorted desc by score, expected_ids)
type _Cache = dict[str, tuple[list[SearchResult], list[str]]]


class RetrievalMetaScorer:
    def __init__(
        self,
        vector_store: AbstractVectorStore,
        config: EvalConfig,
        search_mode: SearchMode = SearchMode.SEMANTIC,
        step: float = 0.005,
        extractor_model_id: str | None = None,
    ) -> None:
        self._vs = vector_store
        self._config = config
        self._search_mode = search_mode
        self._extractor_model_id = extractor_model_id or "none"
        self._search_mode_hash = fingerprint(f"{search_mode.value}:{self._extractor_model_id}")
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
            self._vs, self._config, search_mode=self._search_mode, extractor_model_id=self._extractor_model_id
        )
        cache: _Cache = {}
        touched = set()

        for ex in examples:
            cache_fp = build_cache_fp(self._config.retrieval_cache_dir, ex, agent_hash=self._search_mode_hash)
            scored, used = await read_or_run(
                cache_fp,
                RetrievalScoredResult,
                runner._fetch_example(ex),
            )
            slug_results = [SearchResult(node_id=slug, score=score) for slug, score in scored.results]
            cache[ex.corpus_id] = (slug_results, list(ex.expected_relevant_ids))
            touched.add(used)

        sweep_stale_entries(
            self._config.retrieval_cache_dir,
            corpus_ids=[ex.corpus_id for ex in examples],
            touched=touched,
            agent_hash=self._search_mode_hash,
        )
        return cache


def _compute_metrics(
    results: list[SearchResult],
    expected_ids: list[str],
    threshold: float,
) -> tuple[float, float, float]:
    """Return (recall_at_5, precision_at_5, score) for one corpus example at one threshold.

    score is F2 for positive examples and specificity for negative examples.
    Both are in [0, 1] and feed the macro_f2 average on RetrievalSweepPoint.
    """
    filtered = [r for r in results if r.score >= threshold][:TOP_K]
    returned_ids = {r.node_id for r in filtered}

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
