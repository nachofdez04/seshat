import pytest

from seshat.eval.verification.scorers import scorer


def _run(expected_supported: bool, predicted_supported: bool) -> list:
    return scorer(
        inputs={"title": "T", "description": "D", "quote": "Q"},
        outputs={"supported": predicted_supported},
        expectations={"expected_supported": expected_supported},
    )


class TestVerificationScorer:
    @pytest.mark.parametrize(
        ("expected", "predicted", "key"),
        [
            (True, True, "verification.tp"),
            (False, True, "verification.fp"),
            (True, False, "verification.fn"),
            (False, False, "verification.tn"),
        ],
    )
    def test_outcome_label(self, expected, predicted, key):
        by_name = {f.name: f.value for f in _run(expected, predicted)}
        assert by_name[key] == pytest.approx(1.0)

    def test_exactly_one_feedback_emitted(self):
        for exp in (True, False):
            for pred in (True, False):
                assert len(_run(exp, pred)) == 1
