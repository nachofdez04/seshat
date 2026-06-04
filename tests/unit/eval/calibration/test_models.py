import pytest

from seshat.eval.calibration.models import (
    IdentificationSweepResult,
    RetrievalSweepPoint,
    RetrievalSweepResult,
    SweepPoint,
    TypeMetrics,
)
from seshat.models.enums import ConceptType


class TestIdentificationSweepResult:
    def test_round_trips(self):
        point = SweepPoint(
            threshold=0.5,
            metrics={ConceptType.DECISION: TypeMetrics(precision=0.8, recall=0.7, f1=0.747)},
            macro_f1=0.747,
        )
        result = IdentificationSweepResult(points=[point], suggested_threshold=0.5)
        assert result.suggested_threshold == pytest.approx(0.5)
        assert result.points[0].threshold == pytest.approx(0.5)
        assert result.points[0].macro_f1 == pytest.approx(0.747)

    def test_empty_points_allowed(self):
        result = IdentificationSweepResult(points=[], suggested_threshold=0.0)
        assert result.points == []


class TestRetrievalSweepResult:
    def test_round_trips(self):
        point = RetrievalSweepPoint(threshold=0.3, recall_at_5=0.8, precision_at_5=0.4)
        result = RetrievalSweepResult(points=[point], suggested_threshold=0.3)
        assert result.suggested_threshold == pytest.approx(0.3)
        assert result.points[0].recall_at_5 == pytest.approx(0.8)
        assert result.points[0].precision_at_5 == pytest.approx(0.4)

    def test_multiple_points_ordered(self):
        points = [
            RetrievalSweepPoint(threshold=0.0, recall_at_5=1.0, precision_at_5=0.2),
            RetrievalSweepPoint(threshold=0.5, recall_at_5=0.6, precision_at_5=0.5),
            RetrievalSweepPoint(threshold=1.0, recall_at_5=0.0, precision_at_5=0.0),
        ]
        result = RetrievalSweepResult(points=points, suggested_threshold=0.0)
        assert len(result.points) == 3
        assert result.points[0].threshold == pytest.approx(0.0)
        assert result.points[2].threshold == pytest.approx(1.0)
