import pytest

from seshat.config.settings import EvalConfig, ObservabilityConfig
from seshat.eval.models import GateResult
from tests.integration.conftest import SKIP_IF_NO_LLM_API
from tests.integration.eval.helpers import CORPUS_BASE_DIR, make_identification_runner

pytestmark = [pytest.mark.integration, pytest.mark.llm, SKIP_IF_NO_LLM_API]


class TestIdentificationEvalRunner:
    async def test_run_produces_gate_result_and_file(self, tmp_path):
        config = EvalConfig(
            corpus_base_dir=CORPUS_BASE_DIR,
            gate_path=tmp_path / "eval_gate.json",
            observability=ObservabilityConfig(
                mlflow_tracking_uri="sqlite:///" + str(tmp_path / "mlflow.db"),
                mlflow_experiment_name="seshat-eval-test",
            ),
        )

        runner = make_identification_runner(config)
        result = await runner.run()

        assert isinstance(result, GateResult)
        assert result.run_id
        assert result.identification_metrics is not None
        assert "decision.precision" in result.identification_metrics
        assert "decision.recall" in result.identification_metrics
        assert "action_item.precision" in result.identification_metrics

        assert (tmp_path / "eval_gate.json").exists()
