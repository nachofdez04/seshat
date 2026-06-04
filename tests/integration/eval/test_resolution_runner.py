import pytest

from seshat.eval.models import GateResult
from tests.integration.conftest import SKIP_IF_NO_LLM_API
from tests.integration.eval.helpers import make_eval_config, make_resolution_runner

pytestmark = [pytest.mark.integration, pytest.mark.llm, pytest.mark.eval, SKIP_IF_NO_LLM_API]


class TestResolutionEvalRunner:
    async def test_run_produces_gate_result_with_resolution_metrics(self, tmp_path):
        config = make_eval_config(tmp_path, "seshat-resolution-eval-test")
        runner = make_resolution_runner(config)
        result = await runner.run()

        assert isinstance(result, GateResult)
        assert result.run_id
        assert result.resolution_metrics is not None
        assert "decision.precision" in result.resolution_metrics
        assert "decision.recall" in result.resolution_metrics
        assert (tmp_path / "eval_gate.json").exists()
