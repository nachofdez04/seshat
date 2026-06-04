import pytest

from seshat.eval.grouping.scorers import scorer


def _score(expected_groups: list[list[str]], predicted_groups: list[list[str]]) -> dict[str, float]:
    feedbacks = scorer(
        inputs={"corpus_id": "test"},
        outputs={"predicted_groups": predicted_groups},
        expectations={"expected_groups": expected_groups},
    )
    return {fb.name: fb.value for fb in feedbacks}


class TestGroupingScorer:
    def test_exact_match_both_metrics_one(self):
        scores = _score([["a", "b"], ["c"]], [["a", "b"], ["c"]])
        assert scores["grouping.exact_match"] == pytest.approx(1.0)
        assert scores["grouping.group_hit_rate"] == pytest.approx(1.0)

    def test_group_and_member_order_do_not_matter(self):
        scores = _score([["a", "b"], ["c"]], [["c"], ["b", "a"]])
        assert scores["grouping.exact_match"] == pytest.approx(1.0)
        assert scores["grouping.group_hit_rate"] == pytest.approx(1.0)

    def test_merged_groups_exact_match_zero_hit_rate_partial(self):
        # expected: two singletons; predicted: one merged group
        # exact_match=0 (not identical); group_hit_rate=0 (neither expected group found)
        scores = _score([["a"], ["b"]], [["a", "b"]])
        assert scores["grouping.exact_match"] == pytest.approx(0.0)
        assert scores["grouping.group_hit_rate"] == pytest.approx(0.0)

    def test_split_group_exact_match_zero_hit_rate_zero(self):
        # expected: one group; predicted: split — neither matches
        scores = _score([["a", "b"]], [["a"], ["b"]])
        assert scores["grouping.exact_match"] == pytest.approx(0.0)
        assert scores["grouping.group_hit_rate"] == pytest.approx(0.0)

    def test_partial_hit_exact_match_zero_hit_rate_partial(self):
        # 3 expected groups; agent gets 2 right, 1 wrong (split)
        # exact_match=0; group_hit_rate=2/3
        scores = _score(
            [["a", "b"], ["c", "d"], ["e"]],
            [["a", "b"], ["c"], ["d"], ["e"]],  # ["c","d"] split into two
        )
        assert scores["grouping.exact_match"] == pytest.approx(0.0)
        assert scores["grouping.group_hit_rate"] == pytest.approx(2.0 / 3)

    def test_single_item_correct(self):
        scores = _score([["a"]], [["a"]])
        assert scores["grouping.exact_match"] == pytest.approx(1.0)
        assert scores["grouping.group_hit_rate"] == pytest.approx(1.0)

    def test_exactly_two_feedbacks_emitted(self):
        feedbacks = scorer(
            inputs={"corpus_id": "test"},
            outputs={"predicted_groups": [["a"]]},
            expectations={"expected_groups": [["a"]]},
        )
        assert len(feedbacks) == 2
        names = {fb.name for fb in feedbacks}
        assert names == {"grouping.exact_match", "grouping.group_hit_rate"}
