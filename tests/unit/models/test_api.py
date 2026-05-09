from datetime import date, timedelta

import pytest
from pydantic import ValidationError

from seshat.models.api import ApproveRequest, BulkApproveRule, NodeFilter, SearchResult


class TestNodeFilterValidation:
    def test_min_confidence_above_1_raises(self):
        with pytest.raises(ValidationError):
            NodeFilter(min_confidence=1.1)

    def test_min_confidence_below_0_raises(self):
        with pytest.raises(ValidationError):
            NodeFilter(min_confidence=-0.1)

    def test_meeting_date_to_in_future_raises(self):
        with pytest.raises(ValidationError):
            NodeFilter(meeting_date_to=date.today() + timedelta(days=1))

    def test_meeting_date_from_after_to_raises(self):
        with pytest.raises(ValidationError):
            NodeFilter(
                meeting_date_from=date(2026, 5, 1),
                meeting_date_to=date(2026, 4, 1),
            )

    def test_no_constraints_accepted(self):
        f = NodeFilter()
        assert f.min_confidence is None
        assert f.meeting_date_from is None


class TestSearchResultValidation:
    def test_score_above_1_raises(self):
        with pytest.raises(ValidationError):
            SearchResult(node_id="n1", score=1.1)

    def test_score_below_0_raises(self):
        with pytest.raises(ValidationError):
            SearchResult(node_id="n1", score=-0.1)


class TestBulkApproveRuleValidation:
    def test_threshold_below_05_raises(self):
        with pytest.raises(ValidationError):
            BulkApproveRule(threshold=0.4)

    def test_threshold_equal_1_raises(self):
        with pytest.raises(ValidationError):
            BulkApproveRule(threshold=1.0)


class TestApproveRequestValidation:
    def test_empty_payload_raises(self):
        with pytest.raises(ValidationError, match="ApproveRequest"):
            ApproveRequest()
