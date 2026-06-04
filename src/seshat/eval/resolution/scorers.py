from __future__ import annotations

from collections import defaultdict

import mlflow.genai
from mlflow.entities import Feedback

from seshat.models.enums import ConceptType


@mlflow.genai.scorer
def scorer(inputs: dict, outputs: dict, expectations: dict) -> list[Feedback]:
    """Precision/recall scorer for resolution quality, broken down by source node ConceptType."""
    slug_to_uuid: dict[str, str] = expectations["slug_to_uuid"]
    slug_to_type: dict[str, str] = expectations["slug_to_type"]

    expected_triples: set[tuple[str, str, str]] = {
        (slug_to_uuid[r["source"]], slug_to_uuid[r["target"]], r["rel_type"])
        for r in expectations["expected_relations"]
        if r["source"] in slug_to_uuid and r["target"] in slug_to_uuid
    }
    predicted_triples: set[tuple[str, str, str]] = {
        (str(r["source_id"]), str(r["target_id"]), r["rel_type"]) for r in outputs["relationships"]
    }

    uuid_to_type: dict[str, str] = {str(v): slug_to_type[k] for k, v in slug_to_uuid.items() if k in slug_to_type}
    tp, fp, fn = _count_by_type(expected_triples, predicted_triples, uuid_to_type)
    return _precision_recall_feedbacks(tp, fp, fn)


def _count_by_type(
    expected: set[tuple[str, str, str]],
    predicted: set[tuple[str, str, str]],
    uuid_to_type: dict[str, str],
) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
    tp: dict[str, int] = defaultdict(int)
    fp: dict[str, int] = defaultdict(int)
    fn: dict[str, int] = defaultdict(int)

    for triple in expected & predicted:
        tp[uuid_to_type.get(triple[0], "")] += 1
    for triple in predicted - expected:
        fp[uuid_to_type.get(triple[0], "")] += 1
    for triple in expected - predicted:
        fn[uuid_to_type.get(triple[0], "")] += 1

    return tp, fp, fn


def _precision_recall_feedbacks(
    tp: dict[str, int],
    fp: dict[str, int],
    fn: dict[str, int],
) -> list[Feedback]:
    feedbacks: list[Feedback] = []
    for ctype in ConceptType:
        t, f_p, f_n = tp[ctype.value], fp[ctype.value], fn[ctype.value]
        if t == 0 and f_p == 0 and f_n == 0:
            continue

        precision = t / (t + f_p) if (t + f_p) else (1.0 if not f_n else 0.0)
        recall = t / (t + f_n) if (t + f_n) else 1.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        feedbacks.extend(
            [
                Feedback(name=f"{ctype}.precision", value=precision),
                Feedback(name=f"{ctype}.recall", value=recall),
                Feedback(name=f"{ctype}.f1", value=f1),
            ]
        )

    return feedbacks
