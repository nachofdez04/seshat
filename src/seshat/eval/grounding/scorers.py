from __future__ import annotations

import mlflow.genai
from mlflow.entities import Feedback


@mlflow.genai.scorer
def scorer(inputs: dict, outputs: dict, expectations: dict) -> list[Feedback]:
    """Confusion-matrix scorer for the grounding agent. No LLM calls."""
    expected: bool = expectations["expected_supported"]
    predicted: bool = outputs["supported"]

    if expected and predicted:
        return [Feedback(name="grounding.tp", value=1.0)]
    if not expected and predicted:
        return [Feedback(name="grounding.fp", value=1.0)]
    if expected and not predicted:
        return [Feedback(name="grounding.fn", value=1.0)]
    return [Feedback(name="grounding.tn", value=1.0)]
