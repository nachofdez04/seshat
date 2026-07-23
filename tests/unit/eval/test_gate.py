import hashlib
import json

import pytest
from pydantic import ValidationError

from seshat.eval.gate import (
    grounding_entries,
    grouping_entries,
    identification_entries,
    read_gate,
    resolution_entries,
    retrieval_entries,
    transcription_entries,
    upsert_gate,
    write_gate,
)
from seshat.eval.models import GateResult, MetricEntry
from seshat.eval.thresholds import TRANSCRIPTION_WER_MAX


def _passing_resolution() -> dict[str, float]:
    return {
        "action_item.precision": 0.82,
        "action_item.recall": 0.80,
        "decision.precision": 0.82,
        "decision.recall": 0.80,
        "open_question.precision": 0.82,
        "open_question.recall": 0.80,
        "risk.precision": 0.82,
        "risk.recall": 0.80,
    }


def _passing_identification() -> dict[str, float]:
    return {
        "decision.precision": 0.85,
        "decision.recall": 0.82,
        "risk.precision": 0.77,
        "risk.recall": 0.81,
        "open_question.precision": 0.76,
        "open_question.recall": 0.76,
        "action_item.precision": 0.90,
        "action_item.recall": 0.88,
    }


class TestMetricEntry:
    def test_gated_entry_carries_passed(self):
        entry = MetricEntry(value=0.9, passed=True, gated=True)
        assert entry.gated is True
        assert entry.passed is True

    def test_non_gated_entry_has_passed_none(self):
        entry = MetricEntry(value=0.42, gated=False, passed=None)
        assert entry.gated is False
        assert entry.passed is None

    def test_gated_true_requires_passed_not_none(self):
        with pytest.raises(ValidationError, match="passed"):
            MetricEntry(value=0.9, gated=True, passed=None)

    @pytest.mark.parametrize(
        ("builder", "gated_key", "non_gated_key"),
        [
            (identification_entries, "decision.precision", "decision.f1"),
            (resolution_entries, "decision.precision", "decision.f1"),
            (retrieval_entries, "recall_at_5", "precision_at_5"),
            (grounding_entries, "precision", "accuracy"),
            (grouping_entries, "group_hit_rate", "exact_match"),
            (transcription_entries, "wer", "wer_macro"),
        ],
    )
    def test_builder_labels_gated_vs_non_gated(self, builder, gated_key, non_gated_key):
        entries = builder({gated_key: 0.9, non_gated_key: 0.9})
        assert entries[gated_key].gated is True
        assert entries[gated_key].passed is not None
        assert entries[non_gated_key].gated is False
        assert entries[non_gated_key].passed is None

    def test_non_gated_entry_survives_write_read_round_trip(self, tmp_path):
        # a non-gated entry (passed=None) must serialise and re-validate through the gate file
        gate_path = tmp_path / "gate.json"
        result = GateResult(run_id="r", grouping_metrics=grouping_entries({"group_hit_rate": 0.85, "exact_match": 0.0}))
        write_gate(result, gate_path)
        loaded = read_gate(gate_path)

        assert loaded.grouping_metrics is not None
        exact = loaded.grouping_metrics["exact_match"]
        assert exact.gated is False
        assert exact.passed is None


class TestGateReadWrite:
    def test_round_trip(self, tmp_path):
        result = GateResult(
            run_id="run-123",
            identification_metrics=identification_entries(_passing_identification()),
            resolution_metrics=resolution_entries(_passing_resolution()),
            retrieval_metrics=retrieval_entries({"recall_at_5": 0.75, "precision_at_5": 0.60}),
        )
        gate_path = tmp_path / "eval_gate.json"
        write_gate(result, gate_path)
        loaded = read_gate(gate_path)

        assert loaded.passed is True
        assert loaded.identification_metrics is not None
        assert loaded.resolution_metrics is not None
        assert loaded.retrieval_metrics is not None
        assert loaded.identification_metrics["decision.precision"].value == 0.85
        assert loaded.resolution_metrics["action_item.precision"].value == 0.82
        assert loaded.retrieval_metrics["recall_at_5"].value == 0.75

    def test_round_trip_with_none_blocks(self, tmp_path):
        result = GateResult(
            run_id="run-456",
            identification_metrics=identification_entries(_passing_identification()),
        )
        gate_path = tmp_path / "eval_gate.json"
        write_gate(result, gate_path)
        loaded = read_gate(gate_path)

        assert loaded.resolution_metrics is None
        assert loaded.retrieval_metrics is None

    def test_read_gate_accepts_a_valid_file_from_the_previous_schema(self, tmp_path):
        result = GateResult(
            run_id="legacy-run",
            identification_metrics=identification_entries(_passing_identification()),
        )
        gate_path = tmp_path / "eval_gate.json"
        data = result.model_dump()
        data.pop("transcription_metrics")
        payload = {key: value for key, value in data.items() if key not in {"passed", "validation_hash"}}
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        data["validation_hash"] = hashlib.sha256(serialized.encode()).hexdigest()[:16]
        gate_path.write_text(json.dumps(data), encoding="utf-8")

        loaded = read_gate(gate_path)

        assert loaded.run_id == "legacy-run"
        assert loaded.transcription_metrics is None

    def test_read_gate_raises_on_tampered_file(self, tmp_path):
        result = GateResult(
            run_id="run-789",
            identification_metrics=identification_entries(_passing_identification()),
        )
        gate_path = tmp_path / "eval_gate.json"
        write_gate(result, gate_path)

        data = json.loads(gate_path.read_text())
        data["identification_metrics"]["decision.precision"]["value"] = 0.99
        gate_path.write_text(json.dumps(data))

        with pytest.raises(ValueError, match="modified outside the pipeline"):
            read_gate(gate_path)


class TestGateResultPassed:
    def test_all_none_fails(self):
        assert GateResult(run_id="r").passed is False

    def test_identification_all_targets_met(self):
        gate_result = GateResult(run_id="r", identification_metrics=identification_entries(_passing_identification()))
        assert gate_result.passed is True

    def test_identification_single_type_below_precision_fails(self):
        m = _passing_identification()
        m["decision.precision"] = 0.60
        gate_result = GateResult(run_id="r", identification_metrics=identification_entries(m))
        assert gate_result.passed is False

    def test_identification_single_type_below_recall_fails(self):
        m = _passing_identification()
        m["risk.recall"] = 0.50
        gate_result = GateResult(run_id="r", identification_metrics=identification_entries(m))
        assert gate_result.passed is False

    def test_identification_spurious_rate_above_threshold_fails(self):
        m = _passing_identification()
        m["decision.spurious_rate"] = 0.20
        gate_result = GateResult(run_id="r", identification_metrics=identification_entries(m))
        assert gate_result.passed is False

    def test_identification_at_exact_threshold_passes(self):
        # decision.precision threshold is 0.80 — exactly at the boundary should pass
        m = _passing_identification()
        m["decision.precision"] = 0.80
        gate_result = GateResult(run_id="r", identification_metrics=identification_entries(m))
        assert gate_result.passed is True

    def test_identification_unknown_metric_not_gated(self):
        # e.g. f1 keys may be logged but are not gated
        m = _passing_identification()
        m["decision.f1"] = 0.0
        gate_result = GateResult(run_id="r", identification_metrics=identification_entries(m))
        assert gate_result.passed is True

    def test_identification_entries_bad_key_raises(self):
        with pytest.raises(ValueError):  # noqa: PT011
            identification_entries({"notatype.precision": 0.90})

    def test_resolution_below_target_fails(self):
        m = _passing_resolution()
        m["action_item.precision"] = 0.50
        gate_result = GateResult(run_id="r", resolution_metrics=resolution_entries(m))
        assert gate_result.passed is False

    def test_retrieval_below_target_fails(self):
        gate_result = GateResult(
            run_id="r", retrieval_metrics=retrieval_entries({"recall_at_5": 0.50, "precision_at_5": 0.80})
        )
        assert gate_result.passed is False

    def test_retrieval_low_precision_at_5_is_not_gated(self):
        # precision_at_5 is tracked but deliberately not a gate condition
        gate_result = GateResult(
            run_id="r", retrieval_metrics=retrieval_entries({"recall_at_5": 0.75, "precision_at_5": 0.10})
        )
        assert gate_result.passed is True

    def test_retrieval_mrr_at_5_below_threshold_fails(self):
        gate_result = GateResult(
            run_id="r", retrieval_metrics=retrieval_entries({"recall_at_5": 0.75, "mrr_at_5": 0.50})
        )
        assert gate_result.passed is False

    def test_retrieval_mrr_at_5_above_threshold_passes(self):
        gate_result = GateResult(
            run_id="r", retrieval_metrics=retrieval_entries({"recall_at_5": 0.75, "mrr_at_5": 0.80})
        )
        assert gate_result.passed is True

    def test_none_blocks_not_gated(self):
        gate_result = GateResult(
            run_id="r",
            identification_metrics=identification_entries(_passing_identification()),
            resolution_metrics=None,
            retrieval_metrics=None,
        )
        assert gate_result.passed is True

    def test_grounding_below_precision_fails(self):
        gate_result = GateResult(run_id="r", grounding_metrics=grounding_entries({"precision": 0.50, "recall": 0.85}))
        assert gate_result.passed is False

    def test_grounding_below_recall_fails(self):
        gate_result = GateResult(run_id="r", grounding_metrics=grounding_entries({"precision": 0.90, "recall": 0.70}))
        assert gate_result.passed is False

    def test_grounding_meets_targets_passes(self):
        gate_result = GateResult(run_id="r", grounding_metrics=grounding_entries({"precision": 0.90, "recall": 0.85}))
        assert gate_result.passed is True

    def test_grouping_below_group_hit_rate_fails(self):
        gate_result = GateResult(
            run_id="r", grouping_metrics=grouping_entries({"group_hit_rate": 0.70, "exact_match": 0.50})
        )
        assert gate_result.passed is False

    def test_grouping_meets_group_hit_rate_passes(self):
        gate_result = GateResult(
            run_id="r", grouping_metrics=grouping_entries({"group_hit_rate": 0.85, "exact_match": 0.70})
        )
        assert gate_result.passed is True

    def test_grouping_exact_match_not_gated(self):
        # exact_match=0 but group_hit_rate above threshold — should pass
        gate_result = GateResult(
            run_id="r", grouping_metrics=grouping_entries({"group_hit_rate": 0.85, "exact_match": 0.0})
        )
        assert gate_result.passed is True

    def test_transcription_wer_below_upper_bound_passes(self):
        # WER is the one lower-is-better gated metric: the threshold is an upper bound.
        gate_result = GateResult(run_id="r", transcription_metrics=transcription_entries({"wer": 0.05}))
        assert gate_result.passed is True

    def test_transcription_wer_above_upper_bound_fails(self):
        gate_result = GateResult(run_id="r", transcription_metrics=transcription_entries({"wer": 0.90}))
        assert gate_result.passed is False

    def test_transcription_wer_at_exact_threshold_passes(self):
        entries = transcription_entries({"wer": TRANSCRIPTION_WER_MAX})
        gate_result = GateResult(run_id="r", transcription_metrics=entries)
        assert gate_result.passed is True

    def test_transcription_wer_macro_not_gated(self):
        # a terrible macro mean cannot fail the gate on its own
        gate_result = GateResult(
            run_id="r", transcription_metrics=transcription_entries({"wer": 0.05, "wer_macro": 0.99})
        )
        assert gate_result.passed is True


class TestGateResultHarnessPassed:
    def test_passing_block_is_true(self):
        gate_result = GateResult(run_id="r", identification_metrics=identification_entries(_passing_identification()))
        assert gate_result.harness_passed("identification") is True

    def test_failing_block_is_false(self):
        m = _passing_identification()
        m["decision.precision"] = 0.10
        gate_result = GateResult(run_id="r", identification_metrics=identification_entries(m))
        assert gate_result.harness_passed("identification") is False

    def test_absent_block_is_false(self):
        gate_result = GateResult(run_id="r", identification_metrics=identification_entries(_passing_identification()))
        assert gate_result.harness_passed("resolution") is False

    def test_harness_can_pass_while_overall_gate_fails(self):
        # identification passes on its own even though a failing resolution block
        # drags the overall gate to False — this is the whole point of the per-harness metric.
        m = _passing_resolution()
        m["action_item.precision"] = 0.10
        gate_result = GateResult(
            run_id="r",
            identification_metrics=identification_entries(_passing_identification()),
            resolution_metrics=resolution_entries(m),
        )
        assert gate_result.passed is False
        assert gate_result.harness_passed("identification") is True
        assert gate_result.harness_passed("resolution") is False

    def test_bad_non_gated_metric_does_not_fail_harness(self):
        # a terrible non-gated metric (f1) must not drag harness_passed to False
        m = _passing_identification()
        m["decision.f1"] = 0.0
        gate_result = GateResult(run_id="r", identification_metrics=identification_entries(m))
        assert gate_result.harness_passed("identification") is True


class TestUpsertGate:
    def test_upsert_preserves_existing_blocks(self, tmp_path):
        gate_path = tmp_path / "gate.json"
        write_gate(
            GateResult(run_id="r1", identification_metrics=identification_entries(_passing_identification())),
            gate_path,
        )
        result = upsert_gate(gate_path, run_id="r2", resolution_metrics=_passing_resolution())

        assert result.identification_metrics is not None
        assert result.resolution_metrics is not None
        assert result.identification_metrics["decision.precision"].value == 0.85
        assert result.resolution_metrics["action_item.precision"].value == 0.82

    def test_upsert_overwrites_existing_block(self, tmp_path):
        gate_path = tmp_path / "gate.json"
        write_gate(
            GateResult(run_id="r1", identification_metrics=identification_entries(_passing_identification())),
            gate_path,
        )
        new_id = _passing_identification()
        new_id["decision.precision"] = 0.95
        result = upsert_gate(gate_path, run_id="r2", identification_metrics=new_id)

        assert result.identification_metrics is not None
        assert result.identification_metrics["decision.precision"].value == 0.95

    def test_upsert_transcription_block_preserves_others(self, tmp_path):
        gate_path = tmp_path / "gate.json"
        write_gate(
            GateResult(run_id="r1", identification_metrics=identification_entries(_passing_identification())),
            gate_path,
        )
        result = upsert_gate(gate_path, run_id="r2", transcription_metrics={"wer": 0.1})

        assert result.identification_metrics is not None
        assert result.transcription_metrics is not None
        assert result.transcription_metrics["wer"].passed is True

    def test_upsert_creates_file_if_missing(self, tmp_path):
        gate_path = tmp_path / "subdir" / "gate.json"
        result = upsert_gate(gate_path, run_id="r1", identification_metrics=_passing_identification())
        assert gate_path.exists()
        assert result.run_id == "r1"
