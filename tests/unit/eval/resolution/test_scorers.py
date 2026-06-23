import pytest

from seshat.eval.resolution.scorers import scorer

SRC = "00000000-0000-0000-0000-000000000001"
TGT = "00000000-0000-0000-0000-000000000002"


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
    }


class TestResolutionScorer:
    def test_both_empty_precision_is_one(self):
        # tp=fp=fn=0: the guard yields precision=1.0, recall=1.0 by convention
        feedbacks = scorer(
            inputs={},
            outputs={"relationships": []},
            expectations=_expectations([]),
        )
        by_name = {f.name: f.value for f in feedbacks}
        assert by_name["precision"] == pytest.approx(1.0)
        assert by_name["recall"] == pytest.approx(1.0)

    def test_perfect_precision_recall(self):
        feedbacks = scorer(
            inputs={},
            outputs={"relationships": [_rel(SRC, TGT, "amends")]},
            expectations=_expectations([{"source": "src", "target": "tgt", "rel_type": "amends"}]),
        )
        by_name = {f.name: f.value for f in feedbacks}
        assert by_name["precision"] == pytest.approx(1.0)
        assert by_name["recall"] == pytest.approx(1.0)

    def test_missed_relation_zero_recall(self):
        feedbacks = scorer(
            inputs={},
            outputs={"relationships": []},
            expectations=_expectations([{"source": "src", "target": "tgt", "rel_type": "amends"}]),
        )
        by_name = {f.name: f.value for f in feedbacks}
        assert by_name["recall"] == pytest.approx(0.0)

    def test_spurious_relation_zero_precision(self):
        feedbacks = scorer(
            inputs={},
            outputs={"relationships": [_rel(SRC, TGT, "supersedes")]},
            expectations=_expectations([]),
        )
        by_name = {f.name: f.value for f in feedbacks}
        assert by_name["precision"] == pytest.approx(0.0)

    def test_wrong_rel_type_is_fp_and_fn(self):
        feedbacks = scorer(
            inputs={},
            outputs={"relationships": [_rel(SRC, TGT, "supersedes")]},
            expectations=_expectations([{"source": "src", "target": "tgt", "rel_type": "amends"}]),
        )
        by_name = {f.name: f.value for f in feedbacks}
        assert by_name["precision"] == pytest.approx(0.0)
        assert by_name["recall"] == pytest.approx(0.0)
