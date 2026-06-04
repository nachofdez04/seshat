from __future__ import annotations

import mlflow.genai
from mlflow.entities import Feedback


@mlflow.genai.scorer
def scorer(inputs: dict, outputs: dict, expectations: dict) -> list[Feedback]:
    """Two grouping quality signals. Order-independent (frozenset comparison).

    grouping.exact_match  — 1.0 if every predicted group exactly matches every expected group.
                            Strict; a single wrong group makes the whole example fail.
    grouping.group_hit_rate — fraction of expected groups that appear exactly in the predicted set.
                              Partial credit: getting k of n groups right scores k/n.
    """
    expected = frozenset(frozenset(g) for g in expectations["expected_groups"])
    predicted = frozenset(frozenset(g) for g in outputs["predicted_groups"])

    exact_match = 1.0 if predicted == expected else 0.0

    hits = sum(1 for g in expected if g in predicted)
    group_hit_rate = hits / len(expected) if expected else 1.0

    return [
        Feedback(name="grouping.exact_match", value=exact_match),
        Feedback(name="grouping.group_hit_rate", value=group_hit_rate),
    ]
