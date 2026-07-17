import pytest

from seshat.app.pipeline.extraction.search_engine import SearchEngine
from seshat.core.config.settings import RAGConfig
from seshat.eval.models import GateResult
from seshat.eval.retrieval.runner import RetrievalEvalRunner
from tests.integration.conftest import SKIP_IF_NO_EMBEDDINGS_API, SKIP_IF_NO_POSTGRES
from tests.integration.eval.helpers import make_eval_config

pytestmark = [
    pytest.mark.integration,
    pytest.mark.llm,
    pytest.mark.embedding,
    pytest.mark.eval,
    SKIP_IF_NO_POSTGRES,
    SKIP_IF_NO_EMBEDDINGS_API,
]


def _make_runner(vector_store, eval_config) -> RetrievalEvalRunner:
    rag_config = RAGConfig()
    search_engine = SearchEngine(
        rag_config=rag_config, vector_store=vector_store, keyword_llm=None, multi_query_llm=None
    )
    return RetrievalEvalRunner(
        search_engine=search_engine, vector_store=vector_store, config=eval_config, rag_config=rag_config
    )


class TestRetrievalEvalRunner:
    async def test_run_produces_gate_result_with_retrieval_metrics(self, vector_store, tmp_path):
        config = make_eval_config(tmp_path, "seshat-retrieval-eval-test")
        runner = _make_runner(vector_store, config)
        result = await runner.run()

        assert isinstance(result, GateResult)
        assert result.run_id
        assert result.retrieval_metrics is not None
        assert "recall_at_5" in result.retrieval_metrics
        assert "precision_at_5" in result.retrieval_metrics
        assert "mrr_at_5" in result.retrieval_metrics
        assert (tmp_path / "eval_gate.json").exists()
