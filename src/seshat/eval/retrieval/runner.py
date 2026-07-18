from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import httpx
import mlflow
import mlflow.genai
import openai
import pandas as pd

from seshat.app.platform.observability.usage_tracker import track_eval_usage
from seshat.core.models.api_graph import NodeFilter
from seshat.core.models.enums import SearchMode
from seshat.core.utils.log import get_logger, set_task_num
from seshat.core.utils.retry import async_retry
from seshat.eval.cache import build_cache_fp, read_or_run, sweep_stale_entries
from seshat.eval.gate import upsert_gate
from seshat.eval.mlflow_logging import log_eval_run_metadata
from seshat.eval.models import RetrievalScoredResult
from seshat.eval.retrieval.corpus_loader import build_kb_nodes, load_corpus
from seshat.eval.retrieval.scorers import TOP_K, scorer

if TYPE_CHECKING:
    from pathlib import Path

    from mlflow.genai.evaluation.entities import EvaluationResult

    from seshat.app.pipeline.extraction.search_engine import SearchEngine
    from seshat.core.config.eval_settings import EvalConfig
    from seshat.core.config.settings import RAGConfig
    from seshat.core.models.nodes import KBNode
    from seshat.eval.corpus_tags import CorpusTagFilter
    from seshat.eval.models import GateResult, RetrievalCorpusExample, RetrievalCorpusNode
    from seshat.infra.vector_store.base_store import AbstractVectorStore


logger = get_logger(__name__)


class RetrievalEvalRunner:
    """Eval runner for the retrieval pass.

    The caller is responsible for passing a dedicated, empty vector store collection.
    Any pre-existing nodes in the collection will appear in search results and corrupt scores.
    """

    def __init__(
        self,
        search_engine: SearchEngine,
        vector_store: AbstractVectorStore,
        config: EvalConfig,
        rag_config: RAGConfig,
    ) -> None:
        self._search_engine = search_engine
        self._vs = vector_store
        self._config = config
        self._rag_config = rag_config
        self._search_mode = rag_config.search_mode
        self._search_mode_hash = search_engine.fingerprint()

    async def run(self, tag_filter: CorpusTagFilter | None = None, model_id: str | None = None) -> GateResult:
        examples = load_corpus(self._config.retrieval_corpus_dir, tag_filter=tag_filter)
        if not examples:
            return upsert_gate(self._config.gate_path, run_id="retrieval-no-corpus")

        threshold = self._config.retrieval_score_thresholds.get(self._search_mode, 0.0)
        result_cache, touched, cache_hits = await self._run_all_predictions(examples, threshold)

        expected_by_id = {ex.corpus_id: ex.expected_relevant_ids for ex in examples}

        def _predict(corpus_id: str, _query_node: dict, _candidate_nodes: list[dict]) -> dict:
            if corpus_id not in result_cache:
                raise KeyError(f"corpus_id {corpus_id!r} not found in result cache — mlflow unpacking mismatch")

            return {
                # used by the scorer
                "retrieved_ids": result_cache[corpus_id],
                # used for debugging in the MLflow UI (shown in traces output); not part of the scorer input
                "expected_relevant_ids": expected_by_id[corpus_id],
            }

        df = _build_dataframe(examples)
        eval_result = mlflow.genai.evaluate(data=df, predict_fn=_predict, scorers=[scorer], model_id=model_id)

        run_id = eval_result.run_id
        retrieval_metrics = _aggregate_metrics(eval_result)

        gate = upsert_gate(
            self._config.gate_path,
            run_id=run_id,
            retrieval_metrics=retrieval_metrics,
        )
        log_eval_run_metadata(
            run_id=run_id,
            harness="retrieval",
            gate_passed=gate.passed,
            corpus_dir=self._config.retrieval_corpus_dir,
            corpus_examples=examples,
            breakdown_artifact=_build_breakdown(eval_result, examples, result_cache),
            tag_filter=tag_filter,
            cache_hits=cache_hits,
            total_predictions=len(examples),
            extra_params={
                "retrieval.search_mode": self._search_mode,
                "retrieval.score_threshold": str(threshold),
                "retrieval.keyword_extraction_provider": (
                    self._rag_config.keyword_extraction_llm.provider
                    if self._rag_config.keyword_extraction_llm
                    else "none"
                ),
                "retrieval.keyword_extraction_model": (
                    self._rag_config.keyword_extraction_llm.model if self._rag_config.keyword_extraction_llm else "none"
                ),
                "retrieval.multi_query_model": (
                    self._rag_config.multi_query.llm.model if self._rag_config.multi_query.llm else "none"
                ),
                "retrieval.multi_query_llm_provider": (
                    self._rag_config.multi_query.llm.provider if self._rag_config.multi_query.llm else "none"
                ),
                "retrieval.multi_query_num_variants": str(
                    self._rag_config.multi_query.num_variants if self._rag_config.multi_query else 0
                ),
                "retrieval.reranker_provider": (
                    self._rag_config.reranker.provider if self._rag_config.reranker else "none"
                ),
                "retrieval.reranker_model": (self._rag_config.reranker.model if self._rag_config.reranker else "none"),
            },
        )

        sweep_stale_entries(
            self._config.retrieval_cache_dir,
            corpus_ids=[ex.corpus_id for ex in examples],
            touched=touched,
            agent_hash=self._search_mode_hash,
        )
        return gate

    @track_eval_usage("retrieval")
    async def _run_all_predictions(
        self,
        examples: list[RetrievalCorpusExample],
        threshold: float,
    ) -> tuple[dict[str, list[str]], set[Path], int]:
        result_cache: dict[str, list[str]] = {}
        touched: set[Path] = set()
        cache_hits = 0
        for task_idx, ex in enumerate(examples):
            set_task_num(task_idx)
            cache_fp = build_cache_fp(self._config.retrieval_cache_dir, ex, agent_hash=self._search_mode_hash)
            scored, used, was_cached = await read_or_run(cache_fp, RetrievalScoredResult, self._fetch_example(ex))
            if was_cached:
                cache_hits += 1
            retrieved_ids = [slug for slug, score in scored.results if score >= threshold]
            result_cache[ex.corpus_id] = retrieved_ids[:TOP_K]
            touched.add(used)
        return result_cache, touched, cache_hits

    async def _fetch_example(self, ex: RetrievalCorpusExample) -> RetrievalScoredResult:
        query_node, candidate_kb_nodes, slug_map = build_kb_nodes(ex)
        seeded = await self._seed_candidates(candidate_kb_nodes)
        if not seeded:
            return RetrievalScoredResult(results=[])

        query = f"{query_node.title} {query_node.description}"
        node_filter = NodeFilter(node_type=None)
        uuid_to_slug = {str(v): k for k, v in slug_map.items()}
        try:
            results = await self._search(query, node_filter, top_k=len(candidate_kb_nodes))
            return RetrievalScoredResult(results=[(uuid_to_slug[str(r.node_id)], r.score) for r in results])
        finally:
            await self._teardown_candidates(candidate_kb_nodes)

    @async_retry(retryable_exceptions=(httpx.ReadError, openai.APIConnectionError))
    async def _search(self, query: str, node_filter: NodeFilter, top_k: int) -> list:
        # score_threshold=None: full unfiltered results are cached so the meta-scorer can
        # sweep post-RRF thresholds in memory without re-running searches.
        # Exception: for hybrid, the calibrated semantic threshold is used as a dense
        # pre-filter before RRF fusion, matching production behaviour (where
        # RAG__MIN_SIMILARITY_SCORE filters the dense leg before fusion). This embeds
        # the threshold in the cached result.
        # NOTE: the user must clear the hybrid cache if the semantic threshold is recalibrated.
        # NOTE: If EVAL__RETRIEVAL_SCORE_THRESHOLDS__SEMANTIC is absent, .get() returns None
        # and no dense pre-filter is applied (as with keyword and semantic).
        score_threshold = (
            self._config.retrieval_score_thresholds.get(SearchMode.SEMANTIC)
            if self._search_mode == SearchMode.HYBRID
            else None
        )
        return await self._search_engine.search(
            query, node_filter=node_filter, exclude_job_id=None, top_k=top_k, score_threshold=score_threshold
        )

    async def _seed_candidates(self, nodes: list[KBNode]) -> bool:
        """Upsert candidate nodes. Returns False if all nodes failed (example should be skipped).

        Note: upsert always calls the embedding model (LangChain's PGVector has no embed-free
        insert path), so embedding_input_tokens will be non-zero even for KEYWORD mode.
        This is seeding cost, not search cost — the search step itself makes no embedding call.
        """
        failures = 0

        async def _upsert(node: KBNode) -> None:
            nonlocal failures
            metadata = {"node_type": node.type.value, "confidence": node.confidence}
            try:
                await self._vs.upsert(str(node.id), text=node.vector_store_text, metadata=metadata)
            except Exception as exc:
                failures += 1
                logger.warning("Failed to seed node %s: %s: %s", node.id, type(exc).__name__, exc)

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


def _build_dataframe(examples: list[RetrievalCorpusExample]) -> pd.DataFrame:
    def _slim_node(n: RetrievalCorpusNode) -> dict:
        return {"id": n.id, "type": n.type.value, "title": n.title, "description": n.description}

    rows = []
    for ex in examples:
        rows.append(
            {
                "inputs": {
                    "corpus_id": ex.corpus_id,
                    "_query_node": _slim_node(ex.query_node),
                    "_candidate_nodes": [_slim_node(node) for node in ex.candidate_nodes],
                },
                "expectations": {"expected_relevant_ids": ex.expected_relevant_ids},
                "tags": {f"corpus.{k}": str(v) for k, v in ex.tags.items()},
            }
        )
    return pd.DataFrame(rows)


def _aggregate_metrics(eval_result: EvaluationResult) -> dict[str, float]:
    result: dict[str, float] = {}
    for metric in ("recall_at_5", "precision_at_5", "mrr_at_5"):
        v = eval_result.metrics.get(f"{metric}/mean")
        if v is not None:
            result[metric] = float(v)
    return result


def _build_breakdown(
    eval_result: EvaluationResult,
    examples: list[RetrievalCorpusExample],
    result_cache: dict[str, list[str]],
) -> dict:
    assert eval_result.result_df is not None
    breakdown: dict[str, dict] = {}
    for ex, (_, row) in zip(examples, eval_result.result_df.iterrows(), strict=True):
        scores: dict[str, float | None] = {}
        for metric in ("recall_at_5", "precision_at_5", "mrr_at_5"):
            v = row.get(f"{metric}/value")
            scores[metric] = float(v) if not pd.isna(v) else None

        breakdown[ex.corpus_id] = {
            "scores": scores,
            "query": ex.query_node.id,
            "expected": ex.expected_relevant_ids,
            "retrieved": result_cache[ex.corpus_id],
        }
    return breakdown
