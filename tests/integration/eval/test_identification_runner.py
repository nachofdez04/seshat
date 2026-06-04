import pytest

from seshat.eval.models import GateResult
from tests.integration.conftest import SKIP_IF_NO_LLM_API
from tests.integration.eval.helpers import make_eval_config, make_identification_runner

pytestmark = [pytest.mark.integration, pytest.mark.llm, pytest.mark.eval, SKIP_IF_NO_LLM_API]


class TestIdentificationEvalRunner:
    async def test_run_produces_gate_result_and_file(self, tmp_path):
        config = make_eval_config(tmp_path, "seshat-identification-eval-test")
        runner = make_identification_runner(config)
        result = await runner.run()

        assert isinstance(result, GateResult)
        assert result.run_id
        assert result.identification_metrics is not None
        assert "decision.precision" in result.identification_metrics
        assert "decision.recall" in result.identification_metrics
        assert "action_item.precision" in result.identification_metrics

        assert (tmp_path / "eval_gate.json").exists()
