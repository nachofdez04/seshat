from uuid import uuid4

import pytest
from pydantic import ValidationError

from seshat.models.enums import RelationshipType
from seshat.models.nodes import (
    ConfidenceBreakdown,
    ExtractionResult,
    ResolutionCandidate,
)
from seshat.models.transcript import Turn
from tests.helpers import make_node as _make_node


class TestConfidenceBreakdown:
    def test_null_heuristics_raises(self):
        with pytest.raises(ValidationError):
            ConfidenceBreakdown(heuristics=None, final=0.5)  # type: ignore[arg-type]


class TestExtractionResult:
    def test_resolution_candidates_ordered_by_confidence_desc(self):
        node_id = uuid4()
        candidates = [
            ResolutionCandidate(
                node_id=uuid4(),
                rel_type=RelationshipType.SUPERSEDES,
                candidate_title="A",
                target_node_confidence=0.6,
            ),
            ResolutionCandidate(
                node_id=uuid4(),
                rel_type=RelationshipType.AMENDS,
                candidate_title="B",
                target_node_confidence=0.9,
            ),
            ResolutionCandidate(
                node_id=uuid4(),
                rel_type=RelationshipType.CONFLICTS_WITH,
                candidate_title="C",
                target_node_confidence=0.75,
            ),
        ]
        sorted_candidates = sorted(candidates, key=lambda c: c.target_node_confidence, reverse=True)
        result = ExtractionResult(
            job_id="job-1",
            nodes=[],
            relationships=[],
            confidence_breakdowns={},
            resolution_candidates={node_id: sorted_candidates},
        )
        scores = [c.target_node_confidence for c in result.resolution_candidates[node_id]]
        assert scores == sorted(scores, reverse=True)


class TestTurnValidation:
    def test_negative_start_seconds_raises(self):
        with pytest.raises(ValidationError):
            Turn(text="hello", start_seconds=-1.0)

    def test_negative_end_seconds_raises(self):
        with pytest.raises(ValidationError):
            Turn(text="hello", end_seconds=-0.1)

    def test_none_offsets_accepted(self):
        t = Turn(text="hello")
        assert t.start_seconds is None
        assert t.end_seconds is None


class TestConfidenceRange:
    def test_node_confidence_above_1_raises(self):
        with pytest.raises(ValidationError):
            _make_node(confidence=1.1)

    def test_node_confidence_below_0_raises(self):
        with pytest.raises(ValidationError):
            _make_node(confidence=-0.1)

    def test_breakdown_final_above_1_raises(self):
        with pytest.raises(ValidationError):
            ConfidenceBreakdown(heuristics=0.8, final=1.5)

    def test_breakdown_heuristics_below_0_raises(self):
        with pytest.raises(ValidationError):
            ConfidenceBreakdown(heuristics=-0.1, final=0.5)

    def test_candidate_confidence_out_of_range_raises(self):
        with pytest.raises(ValidationError):
            ResolutionCandidate(
                node_id=uuid4(),
                rel_type=RelationshipType.SUPERSEDES,
                candidate_title="A",
                target_node_confidence=1.1,
            )
