from __future__ import annotations

import mlflow.genai
from mlflow.entities import Feedback


@mlflow.genai.scorer
def scorer(inputs: dict, outputs: dict, expectations: dict) -> list[Feedback]:
    """Precision/recall scorer for resolution quality. TP = (source_id, target_id, rel_type) exact match."""
    slug_to_uuid: dict[str, str] = expectations["slug_to_uuid"]
    expected_triples: set[tuple[str, str, str]] = {
        (slug_to_uuid[r["source"]], slug_to_uuid[r["target"]], r["rel_type"])
        for r in expectations["expected_relations"]
        if r["source"] in slug_to_uuid and r["target"] in slug_to_uuid
    }
    predicted_triples: set[tuple[str, str, str]] = {
        (str(r["source_id"]), str(r["target_id"]), r["rel_type"]) for r in outputs["relationships"]
    }

    tp = len(expected_triples & predicted_triples)
    fp = len(predicted_triples - expected_triples)
    fn = len(expected_triples - predicted_triples)

    precision = tp / (tp + fp) if (tp + fp) else (1.0 if not expected_triples else 0.0)
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return [
        Feedback(name="precision", value=precision),
        Feedback(name="recall", value=recall),
        Feedback(name="f1", value=f1),
    ]
