from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import httpx
import mlflow
import mlflow.genai
import openai
import pandas as pd

from seshat.eval.cache import clear_cache_dir, read_or_run
from seshat.eval.common import log_breakdown_artifact
from seshat.eval.gate import upsert_gate
from seshat.eval.models import RetrievalResult
from seshat.eval.retrieval.corpus_loader import build_kb_nodes, load_corpus
from seshat.eval.retrieval.scorers import scorer
from seshat.models.api import NodeFilter
from seshat.utils.log import get_logger
from seshat.utils.retry import async_retry

if TYPE_CHECKING:
    from uuid import UUID

    from mlflow.genai.evaluation.entities import EvaluationResult

    from seshat.config.settings import EvalConfig
    from seshat.eval.models import GateResult, RetrievalCorpusExample
    from seshat.models.nodes import KBNode
    from seshat.vector_store.base_store import AbstractVectorStore


logger = get_logger(__name__)


class RetrievalEvalRunner:
    """Eval runner for the retrieval pass.

    The caller is responsible for passing a dedicated, empty vector store collection.
    Any pre-existing nodes in the collection will appear in search results and corrupt scores.
    """

    def __init__(
        self,
        vector_store: AbstractVectorStore,
        config: EvalConfig,
        model_id: str | None = None,
    ) -> None:
        self._vs = vector_store
        self._config = config
        self._model_id = model_id

    async def run(self) -> GateResult:
        mlflow.set_tracking_uri(self._config.observability.mlflow_tracking_uri)
        mlflow.set_experiment(self._config.observability.mlflow_experiment_name)

        examples = load_corpus(self._config.retrieval_corpus_dir)
        if not examples:
            return upsert_gate(self._config.gate_path, run_id="retrieval-no-corpus")

        # Build UUID maps once — build_kb_nodes calls uuid4() so must not be called twice.
        example_nodes = {ex.corpus_id: build_kb_nodes(ex) for ex in examples}
        result_cache = await self._run_all_predictions(examples, example_nodes)

        def _predict(corpus_id: str) -> dict:
            if corpus_id not in result_cache:
                raise KeyError(f"corpus_id {corpus_id!r} not found in result cache — mlflow unpacking mismatch")
            return {"retrieved_ids": result_cache[corpus_id]}

        df = _build_dataframe(examples, example_nodes)
        eval_result = mlflow.genai.evaluate(
            data=df,
            predict_fn=_predict,
            scorers=[scorer],
            model_id=self._model_id,
        )

        run_id = eval_result.run_id
        retrieval_metrics = _aggregate_metrics(eval_result)
        self._log_breakdown(eval_result, examples, example_nodes, result_cache, run_id)

        gate = upsert_gate(
            self._config.gate_path,
            run_id=run_id,
            retrieval_metrics=retrieval_metrics,
        )
        mlflow.log_metrics({**retrieval_metrics, "gate.passed": float(gate.passed)}, run_id=run_id)
        clear_cache_dir(self._config.retrieval_cache_dir)
        return gate

    async def _run_all_predictions(
        self,
        examples: list[RetrievalCorpusExample],
        example_nodes: dict[str, tuple[KBNode, list[KBNode], dict[str, UUID]]],
    ) -> dict[str, list[str]]:
        result_cache: dict[str, list[str]] = {}
        for ex in examples:
            query_node, candidate_kb_nodes, _ = example_nodes[ex.corpus_id]
            result = await read_or_run(
                self._config.retrieval_cache_dir / f"{ex.corpus_id}.json",
                RetrievalResult,
                self._fetch_example(query_node, candidate_kb_nodes),
            )
            result_cache[ex.corpus_id] = result.retrieved_ids
        return result_cache

    async def _fetch_example(self, query_node: KBNode, candidate_kb_nodes: list[KBNode]) -> RetrievalResult:
        seeded = await self._seed_candidates(candidate_kb_nodes)
        if not seeded:
            return RetrievalResult(retrieved_ids=[])
        try:
            query = f"{query_node.title} {query_node.description}"
            node_filter = NodeFilter(node_type=query_node.type)
            results = await self._search(query, node_filter)
            return RetrievalResult(retrieved_ids=[r.node_id for r in results])
        finally:
            await self._teardown_candidates(candidate_kb_nodes)

    @async_retry(retryable_exceptions=(httpx.ReadError, openai.APIConnectionError))
    async def _search(self, query: str, node_filter: NodeFilter) -> list:
        return await self._vs.search(
            query, top_k=5, node_filter=node_filter, score_threshold=self._config.retrieval_score_threshold
        )

    async def _seed_candidates(self, nodes: list[KBNode]) -> bool:
        """Upsert candidate nodes. Returns False if all nodes failed (example should be skipped)."""
        failures = 0

        async def _upsert(node: KBNode) -> None:
            nonlocal failures
            metadata = {"node_type": node.type.value, "confidence": node.confidence}
            try:
                await self._vs.upsert(str(node.id), text=f"{node.title} {node.description}", metadata=metadata)
            except Exception:
                failures += 1
                logger.warning("Failed to seed node %s; eval scores for this example will be inaccurate", node.id)

        await asyncio.gather(*(_upsert(node) for node in nodes))

        if failures == len(nodes):
            logger.error("All %d candidate nodes failed to seed — skipping corpus example", len(nodes))
            return False
        return True

    async def _teardown_candidates(self, nodes: list[KBNode]) -> None:
        async def _delete(node: KBNode) -> None:
            try:
                await self._vs.delete(str(node.id))
            except Exception:
                logger.warning(
                    "Failed to delete node %s during teardown; stale node may affect subsequent examples", node.id
                )

        await asyncio.gather(*(_delete(node) for node in nodes))

    def _log_breakdown(
        self,
        eval_result: EvaluationResult,
        examples: list[RetrievalCorpusExample],
        example_nodes: dict[str, tuple[KBNode, list[KBNode], dict[str, UUID]]],
        result_cache: dict[str, list[str]],
        run_id: str,
    ) -> None:
        log_breakdown_artifact(_build_breakdown(eval_result, examples, example_nodes, result_cache), run_id)


def _build_dataframe(
    examples: list[RetrievalCorpusExample],
    example_nodes: dict[str, tuple[KBNode, list[KBNode], dict[str, UUID]]],
) -> pd.DataFrame:
    rows = []
    for ex in examples:
        _, _, slug_map = example_nodes[ex.corpus_id]
        expected_uuids = [str(slug_map[s]) for s in ex.expected_relevant_ids if s in slug_map]
        rows.append(
            {
                "inputs": {"corpus_id": ex.corpus_id},
                "expectations": {"expected_relevant_ids": expected_uuids},
            }
        )
    return pd.DataFrame(rows)


def _aggregate_metrics(eval_result: EvaluationResult) -> dict[str, float]:
    result: dict[str, float] = {}
    for metric in ("recall_at_5", "precision_at_5"):
        v = eval_result.metrics.get(f"{metric}/mean")
        if v is not None:
            result[metric] = float(v)
    return result


def _build_breakdown(
    eval_result: EvaluationResult,
    examples: list[RetrievalCorpusExample],
    example_nodes: dict[str, tuple[KBNode, list[KBNode], dict[str, UUID]]],
    result_cache: dict[str, list[str]],
) -> dict:
    assert eval_result.result_df is not None
    breakdown: dict[str, dict] = {}
    for ex, (_, row) in zip(examples, eval_result.result_df.iterrows(), strict=True):
        scores: dict[str, float | None] = {}
        for metric in ("recall_at_5", "precision_at_5"):
            v = row.get(f"{metric}/value")
            scores[metric] = float(v) if not pd.isna(v) else None

        _, _, slug_map = example_nodes[ex.corpus_id]
        uuid_to_slug = {str(v): k for k, v in slug_map.items()}
        breakdown[ex.corpus_id] = {
            "scores": scores,
            "query": ex.query_node.id,
            "expected": ex.expected_relevant_ids,
            "retrieved": [uuid_to_slug.get(uid, uid) for uid in result_cache[ex.corpus_id]],
        }
    return breakdown
