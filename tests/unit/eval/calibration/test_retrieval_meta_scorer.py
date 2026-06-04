from __future__ import annotations

import pytest

from seshat.eval.calibration.retrieval_meta_scorer import RetrievalMetaScorer
from seshat.models.api import SearchResult


def _make_scorer(cache: dict, step: float = 0.5) -> RetrievalMetaScorer:
    scorer = RetrievalMetaScorer.__new__(RetrievalMetaScorer)
    scorer._step = step  # coarse grid: 0.0, 0.5, 1.0
    scorer._cache = cache
    return scorer


def _results(scores: list[float]) -> list[SearchResult]:
    return [SearchResult(node_id=f"node-{i}", score=s) for i, s in enumerate(scores)]


class TestSweepThreshold:
    def test_all_relevant_returned(self) -> None:
        # 5 results, all relevant → recall=1.0, precision=5/5=1.0
        cache = {"ex1": (_results([0.9, 0.8, 0.7, 0.6, 0.5]), ["node-0", "node-1", "node-2", "node-3", "node-4"])}
        pt = _make_scorer(cache).sweep_threshold().points[0]  # threshold=0.0
        assert pt.recall_at_5 == pytest.approx(1.0)
        assert pt.precision_at_5 == pytest.approx(1.0)

    def test_none_relevant_returned(self) -> None:
        # positive example, results returned but none expected → recall=0.0, precision=0.0
        cache = {"ex1": (_results([0.9, 0.8, 0.7, 0.6, 0.5]), ["other-node"])}
        pt = _make_scorer(cache).sweep_threshold().points[0]  # threshold=0.0
        assert pt.recall_at_5 == pytest.approx(0.0)
        assert pt.precision_at_5 == pytest.approx(0.0)

    def test_threshold_filters_all_positive_example(self) -> None:
        # threshold=1.0 filters all results; positive example → recall=0.0, precision=0.0
        cache = {"ex1": (_results([0.9, 0.8, 0.7, 0.6, 0.5]), ["node-0"])}
        pt = _make_scorer(cache).sweep_threshold().points[-1]  # threshold=1.0
        assert pt.recall_at_5 == pytest.approx(0.0)
        assert pt.precision_at_5 == pytest.approx(0.0)

    def test_negative_example_results_above_threshold(self) -> None:
        # expected empty, results returned → recall=1.0, precision=0.0
        cache = {"ex1": (_results([0.9, 0.8]), [])}
        pt = _make_scorer(cache).sweep_threshold().points[0]  # threshold=0.0
        assert pt.recall_at_5 == pytest.approx(1.0)
        assert pt.precision_at_5 == pytest.approx(0.0)

    def test_negative_example_no_results_above_threshold(self) -> None:
        # expected empty, nothing passes threshold=1.0 → recall=1.0, precision=1.0
        cache = {"ex1": (_results([0.9, 0.8]), [])}
        pt = _make_scorer(cache).sweep_threshold().points[-1]  # threshold=1.0
        assert pt.recall_at_5 == pytest.approx(1.0)
        assert pt.precision_at_5 == pytest.approx(1.0)

    def test_partial_overlap(self) -> None:
        # 1 of 2 relevant returned in top-5 → recall=0.5, precision=1/5
        cache = {"ex1": (_results([0.9, 0.8, 0.7, 0.6, 0.5]), ["node-0", "node-5"])}
        pt = _make_scorer(cache).sweep_threshold().points[0]  # threshold=0.0
        assert pt.recall_at_5 == pytest.approx(0.5)
        assert pt.precision_at_5 == pytest.approx(1 / 5)

    def test_top_k_cap_at_5(self) -> None:
        # 7 results above threshold; positions 6 & 7 must be excluded
        cache = {
            "ex1": (
                _results([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]),
                ["node-5", "node-6"],
            )
        }
        pt = _make_scorer(cache).sweep_threshold().points[0]
        assert pt.recall_at_5 == pytest.approx(0.0)

    def test_precision_denominator_is_always_5(self) -> None:
        # Only 2 results survive threshold=0.5; TP=2 → precision=2/5, not 2/2
        cache = {"ex1": (_results([0.9, 0.8, 0.3, 0.2, 0.1]), ["node-0", "node-1"])}
        scorer = _make_scorer(cache)
        pt = next(p for p in scorer.sweep_threshold().points if p.threshold == pytest.approx(0.5))
        assert pt.precision_at_5 == pytest.approx(2 / 5)
        assert pt.recall_at_5 == pytest.approx(1.0)

    def test_multiple_examples_averaged(self) -> None:
        # ex1: all 5 relevant → recall=1.0; ex2: none relevant → recall=0.0; average=0.5
        cache = {
            "ex1": (_results([0.9, 0.8, 0.7, 0.6, 0.5]), ["node-0", "node-1", "node-2", "node-3", "node-4"]),
            "ex2": (_results([0.9, 0.8, 0.7, 0.6, 0.5]), ["other-a"]),
        }
        pt = _make_scorer(cache).sweep_threshold().points[0]  # threshold=0.0
        assert pt.recall_at_5 == pytest.approx(0.5)
        assert pt.precision_at_5 == pytest.approx(0.5)


class TestSuggestedThreshold:
    def test_strictly_best_threshold_selected(self) -> None:
        # node-1 (score=0.6) and node-2 (score=0.3) are both relevant.
        # t=0.0: all three returned → TP=2, recall=1.0
        # t=0.5: only node-0 and node-1 returned → TP=1, recall=0.5
        # t=0.0 strictly wins; no tie.
        cache = {"ex1": (_results([0.9, 0.6, 0.3]), ["node-1", "node-2"])}
        result = _make_scorer(cache).sweep_threshold()
        assert result.suggested_threshold == pytest.approx(0.0)

    def test_ties_go_to_lower_threshold(self) -> None:
        # Single result at 0.9; both t=0.0 and t=0.5 pass it → recall tie → lower wins
        cache = {"ex1": (_results([0.9]), ["node-0"])}
        result = _make_scorer(cache).sweep_threshold()
        assert result.suggested_threshold == pytest.approx(0.0)


class TestBuildCacheNotCalled:
    def test_sweep_before_build_cache_raises(self) -> None:
        scorer = RetrievalMetaScorer.__new__(RetrievalMetaScorer)
        scorer._step = 0.5
        scorer._cache = None  # type: ignore[assignment]
        with pytest.raises(RuntimeError, match="build_cache"):
            scorer.sweep_threshold()
