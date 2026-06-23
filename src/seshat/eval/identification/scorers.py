from __future__ import annotations

from collections import defaultdict
from typing import Any

import mlflow.genai
from mlflow.entities import Feedback
from rapidfuzz import fuzz

from seshat.eval.identification.matcher import MatchedNode, MatchResult, match_nodes
from seshat.eval.models import IdentificationCorpusNode
from seshat.models.enums import ConceptType
from seshat.models.nodes import KBNode

_FUZZY_FIELDS: dict[ConceptType, list[str]] = {
    ConceptType.ACTION_ITEM: ["assignee", "due"],
    ConceptType.DECISION: ["rationale"],
    ConceptType.OPEN_QUESTION: ["context"],
}
_EXACT_FIELDS: dict[ConceptType, list[str]] = {
    ConceptType.RISK: ["type"],
}
_SET_FIELDS: dict[ConceptType, list[str]] = {
    ConceptType.DECISION: ["alternatives_considered"],
}
# Corpus fields `task`, `decision`, and `question` are intentionally absent from the
# field dicts above — they are paraphrases of quote/title and would duplicate the
# quote-based fuzzy matching already performed by match_nodes.


@mlflow.genai.scorer
def scorer(inputs: dict, outputs: dict, expectations: dict) -> list[Feedback]:
    """Deterministic precision/recall scorer for identification quality. No LLM calls."""
    transcript = inputs["transcript"]
    expected = [IdentificationCorpusNode(**n) for n in expectations["expected_nodes"]]
    predicted = [KBNode(**n) for n in outputs["nodes"]]

    result = match_nodes(transcript, expected, predicted)
    feedbacks = _precision_recall_feedback(result)
    feedbacks += _field_accuracy_feedback(result.matched)
    return feedbacks


def _precision_recall_feedback(result: MatchResult) -> list[Feedback]:
    true_positives: dict[ConceptType, int] = defaultdict(int)
    false_positives: dict[ConceptType, int] = defaultdict(int)
    false_negatives: dict[ConceptType, int] = defaultdict(int)

    for matched_node in result.matched:
        true_positives[matched_node.expected.type] += 1

    for spurious_node in result.spurious:
        false_positives[spurious_node.type] += 1

    for missed_node in result.missed:
        false_negatives[missed_node.type] += 1

    feedbacks: list[Feedback] = []
    for ctype in ConceptType:
        predicted_count = true_positives[ctype] + false_positives[ctype]
        gold_count = true_positives[ctype] + false_negatives[ctype]
        if predicted_count == 0 and gold_count == 0:
            continue

        precision = true_positives[ctype] / predicted_count if predicted_count else 0.0
        recall = true_positives[ctype] / gold_count if gold_count else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        feedbacks.extend(
            [
                Feedback(name=f"{ctype}.precision", value=precision),
                Feedback(name=f"{ctype}.recall", value=recall),
                Feedback(name=f"{ctype}.f1", value=f1),
            ]
        )
    return feedbacks


def _field_accuracy_feedback(matched: list[MatchedNode]) -> list[Feedback]:
    feedbacks: list[Feedback] = []
    for match in matched:
        ctype = match.expected.type
        expected_fields: dict[str, Any] = match.expected.extra_fields
        predicted_fields: dict[str, Any] = match.predicted.metadata.concept_fields or {}

        if not expected_fields:
            continue

        feedbacks += _field_accuracy_fuzzy_fields_feedback(ctype, expected_fields, predicted_fields)
        feedbacks += _field_accuracy_exact_fields_feedback(ctype, expected_fields, predicted_fields)
        feedbacks += _field_accuracy_set_fields_feedback(ctype, expected_fields, predicted_fields)

    return feedbacks


def _field_accuracy_fuzzy_fields_feedback(
    ctype: ConceptType, expected_fields: dict[str, Any], predicted_fields: dict[str, Any]
) -> list[Feedback]:
    feedbacks: list[Feedback] = []

    for field in _FUZZY_FIELDS.get(ctype, []):
        if field not in expected_fields:
            continue

        exp_val = expected_fields.get(field)
        pred_val = predicted_fields.get(field)
        if exp_val is None:
            # corpus explicitly marks this field as absent; penalise hallucination
            score = 0.0 if pred_val is not None else 1.0
        elif pred_val is None:
            score = 0.0
        else:
            score = fuzz.token_set_ratio(str(exp_val), str(pred_val)) / 100.0
        feedbacks.append(Feedback(name=f"{ctype.value}.{field}", value=score))

    return feedbacks


def _field_accuracy_exact_fields_feedback(
    ctype: ConceptType, expected_fields: dict[str, Any], predicted_fields: dict[str, Any]
) -> list[Feedback]:
    feedbacks: list[Feedback] = []

    for field in _EXACT_FIELDS.get(ctype, []):
        if field not in expected_fields:
            continue

        exp_val = expected_fields.get(field)
        pred_val = predicted_fields.get(field)
        if exp_val is None:
            score = 0.0 if pred_val is not None else 1.0
        elif pred_val is None:
            score = 0.0
        else:
            score = 1.0 if str(exp_val).strip().lower() == str(pred_val).strip().lower() else 0.0
        feedbacks.append(Feedback(name=f"{ctype.value}.{field}", value=score))

    return feedbacks


def _field_accuracy_set_fields_feedback(
    ctype: ConceptType, expected_fields: dict[str, Any], predicted_fields: dict[str, Any]
) -> list[Feedback]:
    feedbacks: list[Feedback] = []

    for field in _SET_FIELDS.get(ctype, []):
        exp_val = expected_fields.get(field)
        if not exp_val:
            continue

        pred_val = predicted_fields.get(field)
        exp_set = {str(v).strip().lower() for v in exp_val}
        pred_set = {str(v).strip().lower() for v in (pred_val or [])}
        score = len(exp_set & pred_set) / len(exp_set)
        feedbacks.append(Feedback(name=f"{ctype.value}.{field}", value=score))

    return feedbacks


def _nli_faithfulness_feedback(result: MatchResult) -> list[Feedback]:
    # TODO: implement using a local cross-encoder NLI model
    return []
