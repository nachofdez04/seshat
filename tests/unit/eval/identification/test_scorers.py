from datetime import date

import pytest

from seshat.eval.identification.scorers import scorer
from seshat.models.enums import ConceptType, IngestionSource, NodeStatus
from seshat.models.nodes import NodeMetadata
from tests.helpers import make_node
from tests.unit.eval.identification.helpers import corpus_node

TRANSCRIPT = (
    "We decided to use PostgreSQL for all operational data. There is a risk that replication lag could affect reads."
)

_DECISION_QUOTE = "We decided to use PostgreSQL for all operational data."
_RISK_QUOTE = "There is a risk that replication lag could affect reads."


def _node_with_concept_fields(quote: str, ctype: ConceptType, concept_fields: dict) -> dict:
    metadata = NodeMetadata(
        job_id="job-1",
        meeting_date=date(2026, 4, 21),
        ingestion_source=IngestionSource.JOB,
        concept_fields=concept_fields,
    )
    return make_node(
        quote=quote,
        transcript=TRANSCRIPT,
        type=ctype,
        status=NodeStatus.APPROVED,
        metadata=metadata,
    ).model_dump(mode="json")


class TestIdentificationScorer:
    def test_perfect_precision_recall_decision(self):
        inputs = {"transcript": TRANSCRIPT, "corpus_id": "001"}
        outputs = {
            "nodes": [
                make_node(quote=_DECISION_QUOTE, transcript=TRANSCRIPT, type=ConceptType.DECISION).model_dump(
                    mode="json"
                )
            ]
        }
        expectations = {"expected_nodes": [corpus_node(_DECISION_QUOTE, ConceptType.DECISION).model_dump(mode="json")]}
        feedbacks = scorer(inputs=inputs, outputs=outputs, expectations=expectations)
        by_name = {f.name: f.value for f in feedbacks}
        assert by_name["decision.precision"] == pytest.approx(1.0)
        assert by_name["decision.recall"] == pytest.approx(1.0)

    def test_missed_node_gives_zero_recall(self):
        inputs = {"transcript": TRANSCRIPT, "corpus_id": "001"}
        outputs = {"nodes": []}
        expectations = {"expected_nodes": [corpus_node(_RISK_QUOTE, ConceptType.RISK).model_dump(mode="json")]}
        feedbacks = scorer(inputs=inputs, outputs=outputs, expectations=expectations)
        by_name = {f.name: f.value for f in feedbacks}
        assert by_name["risk.recall"] == pytest.approx(0.0)

    def test_spurious_node_gives_zero_precision(self):
        inputs = {"transcript": TRANSCRIPT, "corpus_id": "001"}
        outputs = {
            "nodes": [
                make_node(quote=_DECISION_QUOTE, transcript=TRANSCRIPT, type=ConceptType.ACTION_ITEM).model_dump(
                    mode="json"
                )
            ]
        }
        expectations = {"expected_nodes": []}
        feedbacks = scorer(inputs=inputs, outputs=outputs, expectations=expectations)
        by_name = {f.name: f.value for f in feedbacks}
        assert by_name["action_item.precision"] == pytest.approx(0.0)

    def test_mixed_type_decision_hit_and_risk_miss(self):
        # DECISION matches → precision=1, recall=1; RISK expected but not predicted → recall=0
        inputs = {"transcript": TRANSCRIPT, "corpus_id": "001"}
        outputs = {
            "nodes": [
                make_node(quote=_DECISION_QUOTE, transcript=TRANSCRIPT, type=ConceptType.DECISION).model_dump(
                    mode="json"
                )
            ]
        }
        expectations = {
            "expected_nodes": [
                corpus_node(_DECISION_QUOTE, ConceptType.DECISION).model_dump(mode="json"),
                corpus_node(_RISK_QUOTE, ConceptType.RISK).model_dump(mode="json"),
            ]
        }
        feedbacks = scorer(inputs=inputs, outputs=outputs, expectations=expectations)
        by_name = {f.name: f.value for f in feedbacks}
        assert by_name["decision.precision"] == pytest.approx(1.0)
        assert by_name["decision.recall"] == pytest.approx(1.0)
        assert by_name["risk.recall"] == pytest.approx(0.0)
        assert "action_item.precision" not in by_name
        assert "open_question.precision" not in by_name


class TestFieldAccuracyFeedback:
    def test_decision_set_field_alternatives_considered_perfect(self):
        # DECISION matched; alternatives_considered in extra_fields → set-field score = 1.0
        inputs = {"transcript": TRANSCRIPT, "corpus_id": "001"}
        outputs = {
            "nodes": [
                _node_with_concept_fields(
                    _DECISION_QUOTE, ConceptType.DECISION, {"alternatives_considered": ["MySQL", "SQLite"]}
                )
            ]
        }
        expectations = {
            "expected_nodes": [
                corpus_node(
                    _DECISION_QUOTE, ConceptType.DECISION, extra_fields={"alternatives_considered": ["MySQL", "SQLite"]}
                ).model_dump(mode="json")
            ]
        }
        feedbacks = scorer(inputs=inputs, outputs=outputs, expectations=expectations)
        by_name = {f.name: f.value for f in feedbacks}
        assert by_name["decision.alternatives_considered"] == pytest.approx(1.0)

    def test_decision_set_field_alternatives_considered_partial(self):
        # Only one of two expected alternatives predicted → score = 0.5
        inputs = {"transcript": TRANSCRIPT, "corpus_id": "001"}
        outputs = {
            "nodes": [
                _node_with_concept_fields(_DECISION_QUOTE, ConceptType.DECISION, {"alternatives_considered": ["MySQL"]})
            ]
        }
        expectations = {
            "expected_nodes": [
                corpus_node(
                    _DECISION_QUOTE, ConceptType.DECISION, extra_fields={"alternatives_considered": ["MySQL", "SQLite"]}
                ).model_dump(mode="json")
            ]
        }
        feedbacks = scorer(inputs=inputs, outputs=outputs, expectations=expectations)
        by_name = {f.name: f.value for f in feedbacks}
        assert by_name["decision.alternatives_considered"] == pytest.approx(0.5)

    def test_risk_exact_field_type_match(self):
        # RISK matched; type exact field correct → score = 1.0
        inputs = {"transcript": TRANSCRIPT, "corpus_id": "001"}
        outputs = {"nodes": [_node_with_concept_fields(_RISK_QUOTE, ConceptType.RISK, {"type": "future"})]}
        expectations = {
            "expected_nodes": [
                corpus_node(_RISK_QUOTE, ConceptType.RISK, extra_fields={"type": "future"}).model_dump(mode="json")
            ]
        }
        feedbacks = scorer(inputs=inputs, outputs=outputs, expectations=expectations)
        by_name = {f.name: f.value for f in feedbacks}
        assert by_name["risk.type"] == pytest.approx(1.0)

    def test_risk_exact_field_type_mismatch(self):
        # RISK matched but predicted wrong type value → score = 0.0
        inputs = {"transcript": TRANSCRIPT, "corpus_id": "001"}
        outputs = {"nodes": [_node_with_concept_fields(_RISK_QUOTE, ConceptType.RISK, {"type": "blocker"})]}
        expectations = {
            "expected_nodes": [
                corpus_node(_RISK_QUOTE, ConceptType.RISK, extra_fields={"type": "future"}).model_dump(mode="json")
            ]
        }
        feedbacks = scorer(inputs=inputs, outputs=outputs, expectations=expectations)
        by_name = {f.name: f.value for f in feedbacks}
        assert by_name["risk.type"] == pytest.approx(0.0)

    def test_action_item_fuzzy_field_assignee_present(self):
        # ACTION_ITEM matched; assignee fuzzy score > 0 when names match
        action_quote = "We decided to use PostgreSQL for all operational data."
        inputs = {"transcript": TRANSCRIPT, "corpus_id": "001"}
        outputs = {"nodes": [_node_with_concept_fields(action_quote, ConceptType.ACTION_ITEM, {"assignee": "Alice"})]}
        expectations = {
            "expected_nodes": [
                corpus_node(action_quote, ConceptType.ACTION_ITEM, extra_fields={"assignee": "Alice"}).model_dump(
                    mode="json"
                )
            ]
        }
        feedbacks = scorer(inputs=inputs, outputs=outputs, expectations=expectations)
        by_name = {f.name: f.value for f in feedbacks}
        assert by_name["action_item.assignee"] == pytest.approx(1.0)
