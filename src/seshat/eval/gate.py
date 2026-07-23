from __future__ import annotations

import json
from pathlib import Path  # noqa: TC003
from typing import TYPE_CHECKING

from seshat.core.models.enums import ConceptType
from seshat.eval.models import GateResult, MetricEntry
from seshat.eval.thresholds import (
    GROUNDING_PRECISION,
    GROUNDING_RECALL,
    GROUPING_GROUP_HIT_RATE,
    IDENTIFICATION_PRECISION,
    IDENTIFICATION_RECALL,
    IDENTIFICATION_SPURIOUS_RATE,
    RESOLUTION_PRECISION,
    RESOLUTION_RECALL,
    RETRIEVAL_MRR_AT_5,
    RETRIEVAL_RECALL_AT_5,
    TRANSCRIPTION_WER_MAX,
)

if TYPE_CHECKING:
    from collections.abc import Callable


def write_gate(result: GateResult, gate_path: Path) -> None:
    gate_path.parent.mkdir(parents=True, exist_ok=True)
    gate_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")


def read_gate(gate_path: Path) -> GateResult:
    raw = gate_path.read_text(encoding="utf-8")
    data = json.loads(raw)
    stored_hash = data.pop("validation_hash", "")
    data.pop("passed", None)
    expected_hash = GateResult.compute_validation_hash(data)
    if stored_hash != expected_hash:
        raise ValueError(f"eval_gate.json has been modified outside the pipeline: {gate_path}")

    result = GateResult.model_validate_json(raw)
    return result


def upsert_gate(
    gate_path: Path,
    run_id: str,
    identification_metrics: dict[str, float] | None = None,
    resolution_metrics: dict[str, float] | None = None,
    retrieval_metrics: dict[str, float] | None = None,
    grounding_metrics: dict[str, float] | None = None,
    grouping_metrics: dict[str, float] | None = None,
    transcription_metrics: dict[str, float] | None = None,
) -> GateResult:
    """Update only the supplied metric blocks; carry over the rest from the existing file."""
    id_entries = identification_entries(identification_metrics) if identification_metrics is not None else None
    res_entries = resolution_entries(resolution_metrics) if resolution_metrics is not None else None
    ret_entries = retrieval_entries(retrieval_metrics) if retrieval_metrics is not None else None
    grd_entries = grounding_entries(grounding_metrics) if grounding_metrics is not None else None
    grp_entries = grouping_entries(grouping_metrics) if grouping_metrics is not None else None
    trn_entries = transcription_entries(transcription_metrics) if transcription_metrics is not None else None

    if gate_path.exists():
        existing = read_gate(gate_path)
        if id_entries is None:
            id_entries = existing.identification_metrics
        if res_entries is None:
            res_entries = existing.resolution_metrics
        if ret_entries is None:
            ret_entries = existing.retrieval_metrics
        if grd_entries is None:
            grd_entries = existing.grounding_metrics
        if grp_entries is None:
            grp_entries = existing.grouping_metrics
        if trn_entries is None:
            trn_entries = existing.transcription_metrics

    result = GateResult(
        run_id=run_id,
        identification_metrics=id_entries,
        resolution_metrics=res_entries,
        retrieval_metrics=ret_entries,
        grounding_metrics=grd_entries,
        grouping_metrics=grp_entries,
        transcription_metrics=trn_entries,
    )
    write_gate(result, gate_path)
    return result


# ── Per-group converters (also exported for test use) ────────────────────────


def identification_entries(metrics: dict[str, float]) -> dict[str, MetricEntry]:
    def identification_gate_judge(key: str, value: float) -> bool | None:
        ctype_str, metric = key.rsplit(".", maxsplit=1)
        ctype = ConceptType(ctype_str)
        match metric:
            case "precision":
                return value >= IDENTIFICATION_PRECISION[ctype]
            case "recall":
                return value >= IDENTIFICATION_RECALL[ctype]
            case "spurious_rate":
                return value <= IDENTIFICATION_SPURIOUS_RATE[ctype]
            case _:  # non-gated metric; logged for observability, not checked
                return None

    return _harness_entries(metrics, identification_gate_judge)


def resolution_entries(metrics: dict[str, float]) -> dict[str, MetricEntry]:
    def resolution_gate_judge(key: str, value: float) -> bool | None:
        ctype_str, metric = key.rsplit(".", maxsplit=1)
        ctype = ConceptType(ctype_str)
        match metric:
            case "precision":
                return value >= RESOLUTION_PRECISION[ctype]
            case "recall":
                return value >= RESOLUTION_RECALL[ctype]
            case _:  # non-gated metric; logged for observability, not checked
                return None

    return _harness_entries(metrics, resolution_gate_judge)


def retrieval_entries(metrics: dict[str, float]) -> dict[str, MetricEntry]:
    def retrieval_gate_judge(key: str, value: float) -> bool | None:
        match key:
            case "recall_at_5":
                return value >= RETRIEVAL_RECALL_AT_5
            case "mrr_at_5":
                return value >= RETRIEVAL_MRR_AT_5
            case _:
                return None

    return _harness_entries(metrics, retrieval_gate_judge)


def grounding_entries(metrics: dict[str, float]) -> dict[str, MetricEntry]:
    def grounding_gate_judge(key: str, value: float) -> bool | None:
        match key:
            case "precision":
                return value >= GROUNDING_PRECISION
            case "recall":
                return value >= GROUNDING_RECALL
            case _:  # non-gated metric; logged for observability, not checked
                return None

    return _harness_entries(metrics, grounding_gate_judge)


def grouping_entries(metrics: dict[str, float]) -> dict[str, MetricEntry]:
    def grouping_gate_judge(key: str, value: float) -> bool | None:
        return value >= GROUPING_GROUP_HIT_RATE if key == "group_hit_rate" else None

    return _harness_entries(metrics, grouping_gate_judge)


def transcription_entries(metrics: dict[str, float]) -> dict[str, MetricEntry]:
    def transcription_gate_judge(key: str, value: float) -> bool | None:
        # WER is lower-is-better: the threshold is an upper bound, inverting the comparison
        # used by every other harness here.
        return value <= TRANSCRIPTION_WER_MAX if key == "wer" else None

    return _harness_entries(metrics, transcription_gate_judge)


def _harness_entries(
    metrics: dict[str, float], gate_judge: Callable[[str, float], bool | None]
) -> dict[str, MetricEntry]:
    """Build a MetricEntry map from a `gate_judge` fn returning True/False for gated metrics, None for non-gated."""
    result: dict[str, MetricEntry] = {}
    for k, v in metrics.items():
        verdict = gate_judge(k, v)
        result[k] = MetricEntry(value=round(v, 3), gated=verdict is not None, passed=verdict)

    return result
