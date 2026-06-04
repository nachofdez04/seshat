import pytest

from seshat.eval.resolution.scorers import scorer
from seshat.models.enums import ConceptType

SRC = "00000000-0000-0000-0000-000000000001"
TGT = "00000000-0000-0000-0000-000000000002"

# src node is action_item; tgt is decision
_SLUG_TO_TYPE = {"src": ConceptType.ACTION_ITEM, "tgt": ConceptType.DECISION}


def _rel(source_id: str, target_id: str, rel_type: str) -> dict:
    from datetime import UTC, datetime

    return {
        "source_id": source_id,
        "target_id": target_id,
        "rel_type": rel_type,
        "job_id": "eval",
        "created_at": datetime.now(UTC).isoformat(),
    }


def _slug_to_uuid() -> dict[str, str]:
    return {"src": SRC, "tgt": TGT}


def _expectations(relations: list[dict], slug_to_uuid: dict | None = None) -> dict:
    return {
        "expected_relations": relations,
        "slug_to_uuid": slug_to_uuid or _slug_to_uuid(),
        "slug_to_type": _SLUG_TO_TYPE,
    }


class TestResolutionScorer:
    def test_both_empty_no_feedbacks(self):
        # tp=fp=fn=0 for all types: no data to score
        feedbacks = scorer(
            inputs={},
            outputs={"relationships": []},
            expectations=_expectations([]),
        )
        assert feedbacks == []

    def test_perfect_precision_recall(self):
        feedbacks = scorer(
            inputs={},
            outputs={"relationships": [_rel(SRC, TGT, "amends")]},
            expectations=_expectations([{"source": "src", "target": "tgt", "rel_type": "amends"}]),
        )
        by_name: dict[str, float] = {f.name: float(f.value) for f in feedbacks}
        assert by_name["action_item.precision"] == pytest.approx(1.0)
        assert by_name["action_item.recall"] == pytest.approx(1.0)

    def test_missed_relation_zero_recall(self):
        feedbacks = scorer(
            inputs={},
            outputs={"relationships": []},
            expectations=_expectations([{"source": "src", "target": "tgt", "rel_type": "amends"}]),
        )
        by_name: dict[str, float] = {f.name: float(f.value) for f in feedbacks}
        assert by_name["action_item.recall"] == pytest.approx(0.0)

    def test_spurious_relation_zero_precision(self):
        feedbacks = scorer(
            inputs={},
            outputs={"relationships": [_rel(SRC, TGT, "supersedes")]},
            expectations=_expectations([]),
        )
        by_name: dict[str, float] = {f.name: float(f.value) for f in feedbacks}
        assert by_name["action_item.precision"] == pytest.approx(0.0)

    def test_wrong_rel_type_is_fp_and_fn(self):
        feedbacks = scorer(
            inputs={},
            outputs={"relationships": [_rel(SRC, TGT, "supersedes")]},
            expectations=_expectations([{"source": "src", "target": "tgt", "rel_type": "amends"}]),
        )
        by_name: dict[str, float] = {f.name: float(f.value) for f in feedbacks}
        assert by_name["action_item.precision"] == pytest.approx(0.0)
        assert by_name["action_item.recall"] == pytest.approx(0.0)
