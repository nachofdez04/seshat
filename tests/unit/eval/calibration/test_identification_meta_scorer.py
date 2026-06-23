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


def _make_kb_node(ctype: ConceptType, title: str, heuristics: float) -> KBNode:
    """KBNode with no quote anchors and a heuristics score; matching uses title similarity."""
    metadata = NodeMetadata(
        job_id="test",
        meeting_date=date(2026, 1, 1),
        confidence_breakdown=ConfidenceBreakdown(heuristics=heuristics),
    )
    return make_node(
        node_id=title,
        title=title,
        description=f"Description of {title}",
        confidence=heuristics,
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


def _make_scorer(cache: dict, step: float = 0.5) -> tuple[IdentificationMetaScorer, dict]:
    scorer = IdentificationMetaScorer.__new__(IdentificationMetaScorer)
    scorer._step = step  # coarse grid: 0.0, 0.5, 1.0
    return scorer, cache


TRANSCRIPT = "decided to use Kafka"
CORPUS_ID = "ex1"


# ── sweep cases ──────────────────────────────────────────────────────────────


class TestSweepThreshold:
    def test_all_correct_approved_precision_1(self) -> None:
        # final=0.9 above threshold=0.0; matched → TP=1, FP=0 → precision=1.0, coverage=1.0
        node = _make_kb_node(ConceptType.DECISION, "Use Kafka", 0.9)
        ex = _make_example(CORPUS_ID, TRANSCRIPT, [corpus_node(TRANSCRIPT, ConceptType.DECISION, title="Use Kafka")])
        scorer, cache = _make_scorer({CORPUS_ID: (_make_result(CORPUS_ID, [node]), ex)})

        pt = scorer._compute_sweep(cache).points[0]  # threshold=0.0
        assert pt.precision_approved == pytest.approx(1.0)
        assert pt.coverage == pytest.approx(1.0)
        assert pt.per_type[ConceptType.DECISION].precision_approved == pytest.approx(1.0)
        assert pt.per_type[ConceptType.DECISION].coverage == pytest.approx(1.0)

    def test_threshold_above_all_scores_nothing_approved(self) -> None:
        # final=0.3 below threshold=1.0 → no nodes approved → coverage=0.0, precision=1.0 (no FP)
        node = _make_kb_node(ConceptType.DECISION, "Use Kafka", 0.3)
        ex = _make_example(CORPUS_ID, TRANSCRIPT, [corpus_node(TRANSCRIPT, ConceptType.DECISION, title="Use Kafka")])
        scorer, cache = _make_scorer({CORPUS_ID: (_make_result(CORPUS_ID, [node]), ex)})

        pt = scorer._compute_sweep(cache).points[-1]  # threshold=1.0
        assert pt.coverage == pytest.approx(0.0)
        assert pt.precision_approved == pytest.approx(1.0)

    def test_partial_threshold_correct_node_approved(self) -> None:
        # node_a (final=0.8) passes at t=0.5, node_b (final=0.2) filtered
        # node_a is matched (TP=1), node_b is not approved → coverage=0.5, precision=1.0
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
        scorer, cache = _make_scorer({CORPUS_ID: (_make_result(CORPUS_ID, [node_a, node_b]), ex)})

        pt = next(p for p in scorer._compute_sweep(cache).points if p.threshold == pytest.approx(0.5))
        assert pt.precision_approved == pytest.approx(1.0)
        assert pt.coverage == pytest.approx(0.5)

    def test_spurious_node_reduces_precision(self) -> None:
        # Predicted node not in expected → FP=1, TP=0, gold=0 → precision=0.0, coverage=0.0
        node = _make_kb_node(ConceptType.DECISION, "Use Kafka", 0.9)
        ex = _make_example(CORPUS_ID, TRANSCRIPT, [])
        scorer, cache = _make_scorer({CORPUS_ID: (_make_result(CORPUS_ID, [node]), ex)})

        pt = scorer._compute_sweep(cache).points[0]
        assert pt.precision_approved == pytest.approx(0.0)
        assert pt.coverage == pytest.approx(0.0)

    def test_multiple_examples_aggregated_globally(self) -> None:
        # ex1: DECISION TP=1; ex2: predicted RISK (not in expected DECISION) → RISK FP=1
        # DECISION: TP=1, FP=0, gold=2 → precision=1.0, coverage=0.5 (1 of 2 gold matched)
        # RISK: TP=0, FP=1, gold=0 → precision=0.0, coverage=0.0
        # aggregate: TP=1, FP=1, gold=2 → precision=0.5, coverage=0.5
        node_a = _make_kb_node(ConceptType.DECISION, "Use Kafka", 0.9)
        ex1 = _make_example("ex1", "use Kafka", [corpus_node("use Kafka", ConceptType.DECISION, title="Use Kafka")])

        ex2 = _make_example(
            "ex2", "use Postgres", [corpus_node("use Postgres", ConceptType.DECISION, title="Use Postgres")]
        )
        node_b_wrong = _make_kb_node(ConceptType.RISK, "Some risk", 0.9)

        scorer, cache = _make_scorer(
            {
                "ex1": (_make_result("ex1", [node_a]), ex1),
                "ex2": (_make_result("ex2", [node_b_wrong]), ex2),
            }
        )

        pt = scorer._compute_sweep(cache).points[0]
        assert pt.per_type[ConceptType.DECISION].precision_approved == pytest.approx(1.0)
        assert pt.per_type[ConceptType.DECISION].coverage == pytest.approx(0.5)
        assert pt.per_type[ConceptType.RISK].precision_approved == pytest.approx(0.0)
        assert pt.per_type[ConceptType.RISK].coverage == pytest.approx(0.0)
        assert pt.precision_approved == pytest.approx(0.5)
        assert pt.coverage == pytest.approx(0.5)


# ── suggested threshold ──────────────────────────────────────────────────────


class TestSuggestedThreshold:
    def test_picks_threshold_meeting_p_target(self) -> None:
        # At t=0.0 spurious node leaks in → precision=0.5 (fails p_target=0.95)
        # At t=0.5 only matched node passes → precision=1.0, coverage=0.5 ← selected
        node_good = _make_kb_node(ConceptType.DECISION, "Use Kafka", 0.8)
        node_spurious = _make_kb_node(ConceptType.DECISION, "Use Postgres", 0.2)
        ex = _make_example(
            CORPUS_ID,
            "use Kafka use Postgres",
            [corpus_node("use Kafka", ConceptType.DECISION, title="Use Kafka")],
        )
        scorer, cache = _make_scorer({CORPUS_ID: (_make_result(CORPUS_ID, [node_good, node_spurious]), ex)})

        assert scorer._compute_sweep(cache, p_target=0.95).suggested_threshold == pytest.approx(0.5)

    def test_ties_go_to_lower_threshold(self) -> None:
        # Single matched node at final=0.9; t=0.0 and t=0.5 both yield precision=1.0,
        # coverage=1.0 → tie → lower threshold wins
        node = _make_kb_node(ConceptType.DECISION, "Use Kafka", 0.9)
        ex = _make_example(CORPUS_ID, TRANSCRIPT, [corpus_node(TRANSCRIPT, ConceptType.DECISION, title="Use Kafka")])
        scorer, cache = _make_scorer({CORPUS_ID: (_make_result(CORPUS_ID, [node]), ex)})

        assert scorer._compute_sweep(cache, p_target=0.95).suggested_threshold == pytest.approx(0.0)

    def test_fallback_to_argmax_precision_when_p_target_unachievable(self) -> None:
        # All nodes are spurious → precision never exceeds 0.0; fallback picks argmax precision
        # At t=1.0 nothing approved → precision=1.0 (no FP); should be selected
        node = _make_kb_node(ConceptType.DECISION, "Use Kafka", 0.4)
        ex = _make_example(CORPUS_ID, TRANSCRIPT, [])  # no expected nodes → every predicted is FP
        scorer, cache = _make_scorer({CORPUS_ID: (_make_result(CORPUS_ID, [node]), ex)})

        result = scorer._compute_sweep(cache, p_target=0.95)
        # At t=0.5 and t=1.0, node (final=0.4) is filtered → precision=1.0; lower threshold wins
        assert result.suggested_threshold == pytest.approx(0.5)


# ── gate logic (_filter_by_threshold) ────────────────────────────────────────


def _make_kb_node_with_grounding(ctype: ConceptType, title: str, heuristics: float, grounding_passed: bool) -> KBNode:
    metadata = NodeMetadata(
        job_id="test",
        meeting_date=date(2026, 1, 1),
        confidence_breakdown=ConfidenceBreakdown(
            grounding_enabled=True, heuristics=heuristics, grounding_passed=grounding_passed
        ),
    )
    return make_node(
        node_id=title,
        title=title,
        description=f"Description of {title}",
        confidence=heuristics,
        type=ctype,
        metadata=metadata,
        quote_anchors=[],
    )


def _make_result_with_grounding(job_id: str, nodes: list[KBNode]) -> IdentificationResult:
    breakdowns = {n.id: n.metadata.confidence_breakdown for n in nodes if n.metadata.confidence_breakdown}
    return IdentificationResult(job_id=job_id, nodes=nodes, confidence_breakdowns=breakdowns)


class TestFilterByThreshold:
    def test_grounding_disabled_passes_on_heuristics_alone(self) -> None:
        # No grounding score → heuristics-only path
        node = _make_kb_node(ConceptType.DECISION, "Use Kafka", 0.8)
        result = _make_result(CORPUS_ID, [node])

        from seshat.eval.calibration.identification_meta_scorer import _filter_by_threshold

        assert _filter_by_threshold(result, 0.7) == [node]
        assert _filter_by_threshold(result, 0.9) == []

    def test_grounding_pass_and_heuristics_pass_approved(self) -> None:
        node = _make_kb_node_with_grounding(ConceptType.DECISION, "Use Kafka", heuristics=0.8, grounding_passed=True)
        result = _make_result_with_grounding(CORPUS_ID, [node])

        from seshat.eval.calibration.identification_meta_scorer import _filter_by_threshold

        assert _filter_by_threshold(result, 0.7) == [node]

    def test_grounding_fail_blocks_regardless_of_heuristics(self) -> None:
        # grounding_passed=False → blocked even with heuristics=0.99
        node = _make_kb_node_with_grounding(ConceptType.DECISION, "Use Kafka", heuristics=0.99, grounding_passed=False)
        result = _make_result_with_grounding(CORPUS_ID, [node])

        from seshat.eval.calibration.identification_meta_scorer import _filter_by_threshold

        assert _filter_by_threshold(result, 0.0) == []

    def test_grounding_pass_but_heuristics_below_threshold_blocked(self) -> None:
        node = _make_kb_node_with_grounding(ConceptType.DECISION, "Use Kafka", heuristics=0.3, grounding_passed=True)
        result = _make_result_with_grounding(CORPUS_ID, [node])

        from seshat.eval.calibration.identification_meta_scorer import _filter_by_threshold

        assert _filter_by_threshold(result, 0.5) == []
