from __future__ import annotations

import pytest

from seshat.eval.calibration.retrieval_meta_scorer import RetrievalMetaScorer
from seshat.models.api_graph import SearchResult


def _make_scorer(cache: dict, step: float = 0.5) -> tuple[RetrievalMetaScorer, dict]:
    scorer = RetrievalMetaScorer.__new__(RetrievalMetaScorer)
    scorer._step = step  # coarse grid: 0.0, 0.5, 1.0
    return scorer, cache


def _results(scores: list[float]) -> list[SearchResult]:
    return [SearchResult(node_id=f"node-{i}", score=s) for i, s in enumerate(scores)]


class TestSweepThreshold:
    def test_all_relevant_returned(self) -> None:
        # 5 results, all relevant → recall=1.0, precision=5/5=1.0
        cache = {"ex1": (_results([0.9, 0.8, 0.7, 0.6, 0.5]), ["node-0", "node-1", "node-2", "node-3", "node-4"])}
        scorer, cache = _make_scorer(cache)
        pt = scorer._compute_sweep(cache).points[0]  # threshold=0.0
        assert pt.recall_at_5 == pytest.approx(1.0)
        assert pt.precision_at_5 == pytest.approx(1.0)

    def test_none_relevant_returned(self) -> None:
        # positive example, results returned but none expected → recall=0.0, precision=0.0
        cache = {"ex1": (_results([0.9, 0.8, 0.7, 0.6, 0.5]), ["other-node"])}
        scorer, cache = _make_scorer(cache)
        pt = scorer._compute_sweep(cache).points[0]  # threshold=0.0
        assert pt.recall_at_5 == pytest.approx(0.0)
        assert pt.precision_at_5 == pytest.approx(0.0)

    def test_threshold_filters_all_positive_example(self) -> None:
        # threshold=1.0 filters all results; positive example → recall=0.0, precision=0.0
        cache = {"ex1": (_results([0.9, 0.8, 0.7, 0.6, 0.5]), ["node-0"])}
        scorer, cache = _make_scorer(cache)
        pt = scorer._compute_sweep(cache).points[-1]  # threshold=1.0
        assert pt.recall_at_5 == pytest.approx(0.0)
        assert pt.precision_at_5 == pytest.approx(0.0)

    def test_negative_example_results_above_threshold(self) -> None:
        # expected empty, results returned → specificity=0.0 (returned something, should not have)
        cache = {"ex1": (_results([0.9, 0.8]), [])}
        scorer, cache = _make_scorer(cache)
        pt = scorer._compute_sweep(cache).points[0]  # threshold=0.0
        assert pt.macro_f2 == pytest.approx(0.0)

    def test_negative_example_no_results_above_threshold(self) -> None:
        # expected empty, nothing passes threshold=1.0 → specificity=1.0
        cache = {"ex1": (_results([0.9, 0.8]), [])}
        scorer, cache = _make_scorer(cache)
        pt = scorer._compute_sweep(cache).points[-1]  # threshold=1.0
        assert pt.macro_f2 == pytest.approx(1.0)

    def test_partial_overlap(self) -> None:
        # 1 of 2 relevant returned in top-5 → recall=0.5, precision=1/5
        cache = {"ex1": (_results([0.9, 0.8, 0.7, 0.6, 0.5]), ["node-0", "node-5"])}
        scorer, cache = _make_scorer(cache)
        pt = scorer._compute_sweep(cache).points[0]  # threshold=0.0
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
        scorer, cache = _make_scorer(cache)
        pt = scorer._compute_sweep(cache).points[0]
        assert pt.recall_at_5 == pytest.approx(0.0)

    def test_precision_denominator_is_always_5(self) -> None:
        # Only 2 results survive threshold=0.5; TP=2 → precision=2/5, not 2/2
        cache = {"ex1": (_results([0.9, 0.8, 0.3, 0.2, 0.1]), ["node-0", "node-1"])}
        scorer, cache = _make_scorer(cache)
        pt = next(p for p in scorer._compute_sweep(cache).points if p.threshold == pytest.approx(0.5))
        assert pt.precision_at_5 == pytest.approx(2 / 5)
        assert pt.recall_at_5 == pytest.approx(1.0)

    def test_multiple_examples_averaged(self) -> None:
        # ex1: all 5 relevant → recall=1.0; ex2: none relevant → recall=0.0; average=0.5
        cache = {
            "ex1": (_results([0.9, 0.8, 0.7, 0.6, 0.5]), ["node-0", "node-1", "node-2", "node-3", "node-4"]),
            "ex2": (_results([0.9, 0.8, 0.7, 0.6, 0.5]), ["other-a"]),
        }
        scorer, cache = _make_scorer(cache)
        pt = scorer._compute_sweep(cache).points[0]  # threshold=0.0
        assert pt.recall_at_5 == pytest.approx(0.5)
        assert pt.precision_at_5 == pytest.approx(0.5)


class TestMacroF2:
    def test_perfect_positive_example(self) -> None:
        # recall=1.0, precision=1.0 → F2=1.0
        cache = {"ex1": (_results([0.9, 0.8, 0.7, 0.6, 0.5]), ["node-0", "node-1", "node-2", "node-3", "node-4"])}
        scorer, cache = _make_scorer(cache)
        pt = scorer._compute_sweep(cache).points[0]
        assert pt.macro_f2 == pytest.approx(1.0)

    def test_zero_recall_zero_f2(self) -> None:
        # positive example with 0 TP → F2=0.0
        cache = {"ex1": (_results([0.9, 0.8, 0.7, 0.6, 0.5]), ["other-node"])}
        scorer, cache = _make_scorer(cache)
        pt = scorer._compute_sweep(cache).points[0]
        assert pt.macro_f2 == pytest.approx(0.0)

    def test_f2_formula_recall_weighted(self) -> None:
        # recall=1.0, precision=1/5 (1 TP, 5 returned, 1 expected)
        # F2 = (5 * (1/5) * 1.0) / (4 * (1/5) + 1.0) = 1.0 / 1.8 ≈ 0.556 (rounded to 3dp)
        cache = {"ex1": (_results([0.9, 0.8, 0.7, 0.6, 0.5]), ["node-0"])}
        scorer, cache = _make_scorer(cache)
        pt = scorer._compute_sweep(cache).points[0]
        assert pt.macro_f2 == pytest.approx(1.0 / 1.8, abs=5e-4)

    def test_negative_example_specificity_zero_at_threshold_zero(self) -> None:
        # negative example, everything passes threshold=0 → specificity=0.0 → macro_f2=0.0
        cache = {"ex1": (_results([0.9, 0.8, 0.7]), [])}
        scorer, cache = _make_scorer(cache)
        pt = scorer._compute_sweep(cache).points[0]  # threshold=0.0
        assert pt.macro_f2 == pytest.approx(0.0)

    def test_negative_example_specificity_one_at_threshold_one(self) -> None:
        # negative example, nothing passes threshold=1.0 → specificity=1.0 → macro_f2=1.0
        cache = {"ex1": (_results([0.9, 0.8, 0.7]), [])}
        scorer, cache = _make_scorer(cache)
        pt = scorer._compute_sweep(cache).points[-1]  # threshold=1.0
        assert pt.macro_f2 == pytest.approx(1.0)

    def test_macro_averages_positive_and_negative(self) -> None:
        # positive: recall=1.0, precision=1.0 → F2=1.0
        # negative: results returned → specificity=0.0
        # macro_f2 = (1.0 + 0.0) / 2 = 0.5
        cache = {
            "pos": (_results([0.9, 0.8, 0.7, 0.6, 0.5]), ["node-0", "node-1", "node-2", "node-3", "node-4"]),
            "neg": (_results([0.9, 0.8]), []),
        }
        scorer, cache = _make_scorer(cache)
        pt = scorer._compute_sweep(cache).points[0]  # threshold=0.0
        assert pt.macro_f2 == pytest.approx(0.5)


class TestSuggestedThreshold:
    def test_threshold_zero_not_selected_when_negatives_present(self) -> None:
        # At t=0.0: positive gets F2>0 but negative gets specificity=0.0 → macro_f2 pulled down.
        # At t=0.5: node-0 (score=0.9) still passes, positive recall=1.0 still; negative has
        # nothing above 0.5 → specificity=1.0. macro_f2 is higher at t=0.5.
        cache = {
            "pos": (_results([0.9]), ["node-0"]),
            "neg": (_results([0.4, 0.3]), []),
        }
        scorer, cache = _make_scorer(cache)
        result = scorer._compute_sweep(cache)
        assert result.suggested_threshold == pytest.approx(0.5)

    def test_argmax_macro_f2_selects_threshold(self) -> None:
        # Only positive example, perfect recall & precision at t=0.0 → macro_f2=1.0 everywhere
        # ties go to lower threshold
        cache = {"ex1": (_results([0.9, 0.8, 0.7, 0.6, 0.5]), ["node-0", "node-1", "node-2", "node-3", "node-4"])}
        scorer, cache = _make_scorer(cache)
        result = scorer._compute_sweep(cache)
        assert result.suggested_threshold == pytest.approx(0.0)

    def test_ties_resolve_to_lower_threshold(self) -> None:
        # Positive example: recall=1.0 at all thresholds that keep node-0 (score=0.9).
        # t=0.0 and t=0.5 both keep node-0; same F2. Lower threshold (0.0) wins.
        cache = {"ex1": (_results([0.9]), ["node-0"])}
        scorer, cache = _make_scorer(cache)
        result = scorer._compute_sweep(cache)
        assert result.suggested_threshold == pytest.approx(0.0)

    def test_higher_threshold_wins_when_it_filters_noise_for_negatives(self) -> None:
        # Two examples: positive and negative.
        # At t=0.0: pos F2>0, neg specificity=0 → average penalised.
        # At t=0.5: pos still gets full recall (node-0 at 0.9 passes), neg now empty → specificity=1.
        # t=0.5 wins.
        cache = {
            "pos": (_results([0.9, 0.3, 0.2]), ["node-0"]),  # node-0=0.9 always passes; rest are noise
            "neg": (_results([0.4, 0.3, 0.2]), []),  # all below 0.5
        }
        scorer, cache = _make_scorer(cache)
        result = scorer._compute_sweep(cache)
        assert result.suggested_threshold == pytest.approx(0.5)
