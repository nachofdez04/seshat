import pytest

from seshat.eval.retrieval.scorers import scorer as retrieval_scorer


class TestRetrievalScorer:
    def test_perfect_recall(self):
        feedbacks = retrieval_scorer(
            inputs={},
            outputs={"retrieved_ids": ["id-1", "id-2"]},
            expectations={"expected_relevant_ids": ["id-1", "id-2"]},
        )
        by_name = {f.name: f.value for f in feedbacks}
        assert by_name["recall_at_5"] == pytest.approx(1.0)

    def test_missed_relevant_node(self):
        feedbacks = retrieval_scorer(
            inputs={},
            outputs={"retrieved_ids": ["id-2"]},
            expectations={"expected_relevant_ids": ["id-1", "id-2"]},
        )
        by_name = {f.name: f.value for f in feedbacks}
        assert by_name["recall_at_5"] == pytest.approx(0.5)

    def test_no_relevant_nodes_retrieved(self):
        feedbacks = retrieval_scorer(
            inputs={},
            outputs={"retrieved_ids": ["id-3"]},
            expectations={"expected_relevant_ids": ["id-1"]},
        )
        by_name = {f.name: f.value for f in feedbacks}
        assert by_name["recall_at_5"] == pytest.approx(0.0)

    def test_precision_at_5(self):
        feedbacks = retrieval_scorer(
            inputs={},
            outputs={"retrieved_ids": ["id-1", "id-2", "id-3"]},
            expectations={"expected_relevant_ids": ["id-1"]},
        )
        by_name = {f.name: f.value for f in feedbacks}
        assert by_name["precision_at_5"] == pytest.approx(1 / 3)

    def test_empty_retrieved_ids_gives_zero_scores(self):
        # retrieved_ids=[] with non-empty expected → recall=0, precision=0, no error
        feedbacks = retrieval_scorer(
            inputs={},
            outputs={"retrieved_ids": []},
            expectations={"expected_relevant_ids": ["id-1"]},
        )
        by_name = {f.name: f.value for f in feedbacks}
        assert by_name["recall_at_5"] == pytest.approx(0.0)
        assert by_name["precision_at_5"] == pytest.approx(0.0)

    def test_negative_case_nothing_retrieved(self):
        feedbacks = retrieval_scorer(
            inputs={},
            outputs={"retrieved_ids": []},
            expectations={"expected_relevant_ids": []},
        )
        by_name = {f.name: f.value for f in feedbacks}
        assert by_name["recall_at_5"] == pytest.approx(1.0)
        assert "precision_at_5" not in by_name

    def test_negative_case_spurious_result(self):
        feedbacks = retrieval_scorer(
            inputs={},
            outputs={"retrieved_ids": ["id-1"]},
            expectations={"expected_relevant_ids": []},
        )
        by_name = {f.name: f.value for f in feedbacks}
        assert by_name["recall_at_5"] == pytest.approx(0.0)
