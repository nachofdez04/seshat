from __future__ import annotations

import mlflow.genai
from mlflow.entities import Feedback


@mlflow.genai.scorer
def scorer(inputs: dict, outputs: dict, expectations: dict) -> list[Feedback]:
    """recall@5 and precision@5 for retrieval quality."""
    retrieved_ids: list[str] = outputs.get("retrieved_ids", [])
    expected_ids: set[str] = set(expectations.get("expected_relevant_ids", []))
    top_5 = set(retrieved_ids[:5])

    if not expected_ids:
        # negative case: any retrieved result is a false positive
        recall = 0.0 if top_5 else 1.0
        return [Feedback(name="recall_at_5", value=recall)]
    recall = len(expected_ids & top_5) / len(expected_ids)
    precision = len(expected_ids & top_5) / len(top_5) if top_5 else 0.0

    return [
        Feedback(name="recall_at_5", value=recall),
        Feedback(name="precision_at_5", value=precision),
    ]
