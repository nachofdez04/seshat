import pytest

from seshat.config.settings import EvalConfig, ObservabilityConfig
from seshat.eval.models import GateResult
from seshat.eval.retrieval.runner import RetrievalEvalRunner
from tests.integration.conftest import SKIP_IF_NO_EMBEDDINGS_API, SKIP_IF_NO_POSTGRES
from tests.integration.eval.helpers import CORPUS_BASE_DIR

pytestmark = [
    pytest.mark.integration,
    pytest.mark.embedding,
    SKIP_IF_NO_POSTGRES,
    SKIP_IF_NO_EMBEDDINGS_API,
]


class TestRetrievalEvalRunner:
    async def test_run_produces_gate_result_with_retrieval_metrics(self, vector_store, tmp_path):
        config = EvalConfig(
            corpus_base_dir=CORPUS_BASE_DIR,
            gate_path=tmp_path / "eval_gate.json",
            observability=ObservabilityConfig(
                mlflow_tracking_uri="sqlite:///" + str(tmp_path / "mlflow.db"),
                mlflow_experiment_name="seshat-retrieval-eval-test",
            ),
        )

        runner = RetrievalEvalRunner(vector_store=vector_store, config=config)
        result = await runner.run()

        assert isinstance(result, GateResult)
        assert result.run_id
        assert result.retrieval_metrics is not None
        assert "recall_at_5" in result.retrieval_metrics
        assert "precision_at_5" in result.retrieval_metrics
        assert (tmp_path / "eval_gate.json").exists()
