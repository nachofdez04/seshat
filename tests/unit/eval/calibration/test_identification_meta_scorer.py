from __future__ import annotations

from datetime import date

import pytest

from seshat.eval.calibration.identification_meta_scorer import IdentificationMetaScorer
from seshat.eval.models import IdentificationCorpusExample, IdentificationCorpusNode
from seshat.models.enums import ConceptType
from seshat.models.nodes import ConfidenceBreakdown, IdentificationResult, KBNode, NodeMetadata
from tests.helpers import make_node
from tests.unit.eval.identification.helpers import corpus_node

# ── helpers ─────────────────────────────────────────────────────────────────


def _make_kb_node(ctype: ConceptType, title: str, final: float) -> KBNode:
    """KBNode with no quote anchors and a calibrated final score; matching uses title similarity."""
    metadata = NodeMetadata(
        job_id="test",
        meeting_date=date(2026, 1, 1),
        confidence_breakdown=ConfidenceBreakdown(heuristics=final, final=final),
    )
    return make_node(
        node_id=title,
        title=title,
        description=f"Description of {title}",
        confidence=final,
        type=ctype,
        metadata=metadata,
        quote_anchors=[],
    )


def _make_example(
    corpus_id: str, transcript: str, nodes: list[IdentificationCorpusNode]
) -> IdentificationCorpusExample:
    return IdentificationCorpusExample(corpus_id=corpus_id, transcript=transcript, expected_nodes=nodes)


def _make_result(job_id: str, nodes: list[KBNode]) -> IdentificationResult:
    breakdowns = {n.id: n.metadata.confidence_breakdown for n in nodes if n.metadata.confidence_breakdown}
    return IdentificationResult(job_id=job_id, nodes=nodes, confidence_breakdowns=breakdowns)


def _make_scorer(cache: dict, step: float = 0.5) -> IdentificationMetaScorer:
    scorer = IdentificationMetaScorer.__new__(IdentificationMetaScorer)
    scorer._step = step  # coarse grid: 0.0, 0.5, 1.0
    scorer._cache = cache
    return scorer


TRANSCRIPT = "decided to use Kafka"
CORPUS_ID = "ex1"


# ── sweep cases ──────────────────────────────────────────────────────────────


class TestSweepThreshold:
    def test_all_above_threshold_all_matched(self) -> None:
        # final=0.9 above threshold=0.0; title match → TP=1, P=1, R=1, macro_f1=1
        node = _make_kb_node(ConceptType.DECISION, "Use Kafka", 0.9)
        ex = _make_example(CORPUS_ID, TRANSCRIPT, [corpus_node(TRANSCRIPT, ConceptType.DECISION, title="Use Kafka")])
        scorer = _make_scorer({CORPUS_ID: (_make_result(CORPUS_ID, [node]), ex)})

        pt = scorer.sweep_threshold().points[0]  # threshold=0.0
        assert pt.metrics[ConceptType.DECISION].precision == pytest.approx(1.0)
        assert pt.metrics[ConceptType.DECISION].recall == pytest.approx(1.0)
        assert pt.macro_f1 == pytest.approx(1.0)

    def test_threshold_above_all_scores_nothing_accepted(self) -> None:
        # final=0.3 below threshold=1.0 → no nodes accepted → FN=1, P=0, R=0
        node = _make_kb_node(ConceptType.DECISION, "Use Kafka", 0.3)
        ex = _make_example(CORPUS_ID, TRANSCRIPT, [corpus_node(TRANSCRIPT, ConceptType.DECISION, title="Use Kafka")])
        scorer = _make_scorer({CORPUS_ID: (_make_result(CORPUS_ID, [node]), ex)})

        pt = scorer.sweep_threshold().points[-1]  # threshold=1.0
        assert pt.metrics[ConceptType.DECISION].recall == pytest.approx(0.0)
        assert pt.metrics[ConceptType.DECISION].precision == pytest.approx(0.0)

    def test_partial_threshold_mixed_precision_recall(self) -> None:
        # Two nodes: final=0.8 accepted, final=0.2 filtered at t=0.5
        # Both expected → TP=1, FN=1, FP=0 → P=1.0, R=0.5
        node_a = _make_kb_node(ConceptType.DECISION, "Use Kafka", 0.8)
        node_b = _make_kb_node(ConceptType.DECISION, "Use Postgres", 0.2)
        transcript = "decided to use Kafka decided to use Postgres"
        ex = _make_example(
            CORPUS_ID,
            transcript,
            [
                corpus_node("decided to use Kafka", ConceptType.DECISION, title="Use Kafka"),
                corpus_node("decided to use Postgres", ConceptType.DECISION, title="Use Postgres"),
            ],
        )
        scorer = _make_scorer({CORPUS_ID: (_make_result(CORPUS_ID, [node_a, node_b]), ex)})

        pt = next(p for p in scorer.sweep_threshold().points if p.threshold == pytest.approx(0.5))
        assert pt.metrics[ConceptType.DECISION].recall == pytest.approx(0.5)
        assert pt.metrics[ConceptType.DECISION].precision == pytest.approx(1.0)

    def test_spurious_node_penalises_precision(self) -> None:
        # Predicted node, no expected → FP=1 → precision=0, recall=0
        node = _make_kb_node(ConceptType.DECISION, "Use Kafka", 0.9)
        ex = _make_example(CORPUS_ID, TRANSCRIPT, [])
        scorer = _make_scorer({CORPUS_ID: (_make_result(CORPUS_ID, [node]), ex)})

        d = scorer.sweep_threshold().points[0].metrics[ConceptType.DECISION]
        assert d.precision == pytest.approx(0.0)
        assert d.recall == pytest.approx(0.0)

    def test_multiple_examples_aggregated_globally(self) -> None:
        # Metrics accumulate globally, not per-example averaged.
        # ex1: DECISION TP=1; ex2: predicted RISK node → DECISION FN=1
        # Totals: DECISION TP=1, FP=0, FN=1 → P=1.0, R=0.5
        node_a = _make_kb_node(ConceptType.DECISION, "Use Kafka", 0.9)
        ex1 = _make_example("ex1", "use Kafka", [corpus_node("use Kafka", ConceptType.DECISION, title="Use Kafka")])

        ex2 = _make_example(
            "ex2", "use Postgres", [corpus_node("use Postgres", ConceptType.DECISION, title="Use Postgres")]
        )
        node_b_wrong = _make_kb_node(ConceptType.RISK, "Some risk", 0.9)

        scorer = _make_scorer(
            {
                "ex1": (_make_result("ex1", [node_a]), ex1),
                "ex2": (_make_result("ex2", [node_b_wrong]), ex2),
            }
        )

        pt = scorer.sweep_threshold().points[0]
        assert pt.metrics[ConceptType.DECISION].precision == pytest.approx(1.0)
        assert pt.metrics[ConceptType.DECISION].recall == pytest.approx(0.5)


class TestSuggestedThreshold:
    def test_strictly_best_threshold_selected(self) -> None:
        # At t=0.0 a spurious node (final=0.2) leaks in → FP=1, F1=0.667
        # At t=0.5 only the matched node passes → FP=0, F1=1.0 ← strictly better
        node_good = _make_kb_node(ConceptType.DECISION, "Use Kafka", 0.8)
        node_spurious = _make_kb_node(ConceptType.DECISION, "Use Postgres", 0.2)
        ex = _make_example(
            CORPUS_ID,
            "use Kafka use Postgres",
            [
                corpus_node("use Kafka", ConceptType.DECISION, title="Use Kafka"),
            ],
        )
        scorer = _make_scorer({CORPUS_ID: (_make_result(CORPUS_ID, [node_good, node_spurious]), ex)})

        assert scorer.sweep_threshold().suggested_threshold == pytest.approx(0.5)

    def test_ties_go_to_lower_threshold(self) -> None:
        # Single node at final=0.9; t=0.0 and t=0.5 both pass it → F1 tie → lower wins
        node = _make_kb_node(ConceptType.DECISION, "Use Kafka", 0.9)
        ex = _make_example(CORPUS_ID, TRANSCRIPT, [corpus_node(TRANSCRIPT, ConceptType.DECISION, title="Use Kafka")])
        scorer = _make_scorer({CORPUS_ID: (_make_result(CORPUS_ID, [node]), ex)})

        assert scorer.sweep_threshold().suggested_threshold == pytest.approx(0.0)


class TestFitWeights:
    def test_raises_not_implemented(self) -> None:
        scorer = IdentificationMetaScorer.__new__(IdentificationMetaScorer)
        scorer._step = 0.01
        scorer._cache = {}
        with pytest.raises(NotImplementedError):
            scorer.fit_weights()


class TestBuildCacheNotCalled:
    def test_sweep_before_build_cache_raises(self) -> None:
        scorer = IdentificationMetaScorer.__new__(IdentificationMetaScorer)
        scorer._step = 0.5
        scorer._cache = None  # type: ignore[assignment]
        with pytest.raises(RuntimeError, match="build_cache"):
            scorer.sweep_threshold()
