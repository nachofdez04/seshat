import pytest

from seshat.eval.models import GateResult
from tests.integration.conftest import SKIP_IF_NO_LLM_API
from tests.integration.eval.helpers import make_eval_config, make_grounding_runner

pytestmark = [pytest.mark.integration, pytest.mark.llm, pytest.mark.agents, pytest.mark.eval, SKIP_IF_NO_LLM_API]


class TestGroundingEvalRunner:
    async def test_run_produces_gate_result_and_file(self, tmp_path):
        config = make_eval_config(tmp_path, "seshat-grounding-eval-test")
        runner = make_grounding_runner(config)
        result = await runner.run()

        assert isinstance(result, GateResult)
        assert result.run_id
        assert result.grounding_metrics is not None
        assert "precision" in result.grounding_metrics
        assert "recall" in result.grounding_metrics

        assert (tmp_path / "eval_gate.json").exists()
