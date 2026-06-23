import tempfile
from pathlib import Path

from seshat.eval.gate import read_gate, upsert_gate, write_gate
from seshat.eval.models import GateResult


def _passing_identification() -> dict[str, float]:
    return {
        "decision.precision": 0.85,
        "decision.recall": 0.78,
        "risk.precision": 0.77,
        "risk.recall": 0.81,
        "open_question.precision": 0.76,
        "open_question.recall": 0.76,
        "action_item.precision": 0.90,
        "action_item.recall": 0.88,
    }


class TestGateReadWrite:
    def test_round_trip(self):
        result = GateResult(
            run_id="run-123",
            identification_metrics=_passing_identification(),
            resolution_metrics={"precision": 0.82, "recall": 0.80},
            retrieval_metrics={"recall_at_5": 0.75, "precision_at_5": 0.60},
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            gate_path = Path(tmpdir) / "eval_gate.json"
            write_gate(result, gate_path)
            loaded = read_gate(gate_path)

        assert loaded.passed is True
        assert loaded.identification_metrics is not None
        assert loaded.resolution_metrics is not None
        assert loaded.retrieval_metrics is not None
        assert loaded.identification_metrics["decision.precision"] == 0.85
        assert loaded.resolution_metrics["precision"] == 0.82
        assert loaded.retrieval_metrics["recall_at_5"] == 0.75

    def test_round_trip_with_none_blocks(self):
        result = GateResult(
            run_id="run-456",
            identification_metrics=_passing_identification(),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            gate_path = Path(tmpdir) / "eval_gate.json"
            write_gate(result, gate_path)
            loaded = read_gate(gate_path)

        assert loaded.resolution_metrics is None
        assert loaded.retrieval_metrics is None


class TestGateResultPassed:
    def test_all_none_fails(self):
        assert GateResult(run_id="r").passed is False

    def test_identification_all_targets_met(self):
        assert GateResult(run_id="r", identification_metrics=_passing_identification()).passed is True

    def test_identification_single_type_below_precision_fails(self):
        m = _passing_identification()
        m["decision.precision"] = 0.60
        assert GateResult(run_id="r", identification_metrics=m).passed is False

    def test_identification_single_type_below_recall_fails(self):
        m = _passing_identification()
        m["risk.recall"] = 0.50
        assert GateResult(run_id="r", identification_metrics=m).passed is False

    def test_resolution_below_target_fails(self):
        assert GateResult(run_id="r", resolution_metrics={"precision": 0.50, "recall": 0.85}).passed is False

    def test_retrieval_below_target_fails(self):
        assert GateResult(run_id="r", retrieval_metrics={"recall_at_5": 0.60, "precision_at_5": 0.80}).passed is False

    def test_retrieval_low_precision_at_5_is_not_gated(self):
        # precision_at_5 is tracked but deliberately not a gate condition
        assert GateResult(run_id="r", retrieval_metrics={"recall_at_5": 0.75, "precision_at_5": 0.10}).passed is True

    def test_none_blocks_not_gated(self):
        assert (
            GateResult(
                run_id="r",
                identification_metrics=_passing_identification(),
                resolution_metrics=None,
                retrieval_metrics=None,
            ).passed
            is True
        )


class TestUpsertGate:
    def test_upsert_preserves_existing_blocks(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            gate_path = Path(tmpdir) / "gate.json"
            write_gate(GateResult(run_id="r1", identification_metrics=_passing_identification()), gate_path)
            result = upsert_gate(gate_path, run_id="r2", resolution_metrics={"precision": 0.85, "recall": 0.82})

        assert result.identification_metrics is not None
        assert result.resolution_metrics is not None
        assert result.identification_metrics["decision.precision"] == 0.85
        assert result.resolution_metrics["precision"] == 0.85

    def test_upsert_creates_file_if_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            gate_path = Path(tmpdir) / "subdir" / "gate.json"
            result = upsert_gate(gate_path, run_id="r1", identification_metrics=_passing_identification())
            assert gate_path.exists()
            assert result.run_id == "r1"
