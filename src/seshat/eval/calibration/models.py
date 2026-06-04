from __future__ import annotations

from pydantic import BaseModel

from seshat.models.enums import ConceptType


class TypeMetrics(BaseModel):
    precision: float
    recall: float
    f1: float


class SweepPoint(BaseModel):
    threshold: float
    metrics: dict[ConceptType, TypeMetrics]
    macro_f1: float


class IdentificationSweepResult(BaseModel):
    points: list[SweepPoint]
    suggested_threshold: float  # argmax(macro_f1); ties → lower threshold


class RetrievalSweepPoint(BaseModel):
    threshold: float
    recall_at_5: float
    precision_at_5: float


class RetrievalSweepResult(BaseModel):
    points: list[RetrievalSweepPoint]  # sorted ascending by threshold
    suggested_threshold: float  # argmax(recall_at_5); ties → lower threshold
