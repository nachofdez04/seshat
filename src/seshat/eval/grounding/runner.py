from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import mlflow
import mlflow.genai
import pandas as pd

from seshat.app.agents.grounding import GroundingResult
from seshat.app.platform.observability.latency_tracker import track_eval_latency
from seshat.app.platform.observability.usage_tracker import track_eval_usage
from seshat.core.utils.log import set_task_num
from seshat.eval.cache import build_cache_fp, read_or_run, sweep_stale_entries
from seshat.eval.gate import upsert_gate
from seshat.eval.grounding.corpus_loader import load_corpus
from seshat.eval.grounding.scorers import scorer
from seshat.eval.mlflow_logging import configure_trace_processors, log_eval_run_metadata, make_input_redactor

if TYPE_CHECKING:
    from pathlib import Path

    from mlflow.genai.evaluation.entities import EvaluationResult

    from seshat.app.agents.grounding import GroundingAgent
    from seshat.core.config.eval_settings import EvalConfig
    from seshat.eval.corpus_tags import CorpusTagFilter
    from seshat.eval.grounding.corpus_loader import GroundingCorpusExample
    from seshat.eval.models import GateResult


class GroundingEvalRunner:
    def __init__(self, agent: GroundingAgent, config: EvalConfig) -> None:
        self._agent = agent
        self._config = config

    async def run(self, tag_filter: CorpusTagFilter | None = None, model_id: str | None = None) -> GateResult:
        examples = load_corpus(self._config.grounding_corpus_dir, tag_filter=tag_filter)
        if not examples:
            return upsert_gate(self._config.gate_path, run_id="grounding-no-corpus")

        result_cache, touched, cache_hits = await self._run_all_predictions(examples)

        expected_by_key: dict[tuple[str, int], bool] = {
            (ex.corpus_id, i): node.expected_supported for ex in examples for i, node in enumerate(ex.nodes)
        }

        def _predict(corpus_id: str, node_index: int, _title: str, _description: str, _quote: str) -> dict:
            key = (corpus_id, node_index)
            if key not in result_cache:
                raise KeyError(f"key {key!r} not in result cache — mlflow unpacking mismatch")

            return {
                # used by the scorer
                "supported": result_cache[key].supported,
                # used for debugging in the MLflow UI (shown in traces output); not part of the scorer input
                "expected_supported": expected_by_key[key],
            }

        configure_trace_processors(make_input_redactor(fields_to_exclude={"node_index"}))

        df = _build_dataframe(examples)
        eval_result = mlflow.genai.evaluate(data=df, predict_fn=_predict, scorers=[scorer], model_id=model_id)

        run_id = eval_result.run_id
        grounding_metrics = _aggregate_metrics(eval_result)

        gate = upsert_gate(
            self._config.gate_path,
            run_id=run_id,
            grounding_metrics=grounding_metrics,
        )
        log_eval_run_metadata(
            run_id=run_id,
            harness="grounding",
            gate_passed=gate.passed,
            corpus_dir=self._config.grounding_corpus_dir,
            corpus_examples=examples,
            breakdown_artifact=_build_breakdown(examples, result_cache),
            tag_filter=tag_filter,
            cache_hits=cache_hits,
            total_predictions=sum(len(ex.nodes) for ex in examples),
        )

        sweep_stale_entries(
            self._config.grounding_cache_dir,
            corpus_ids=[ex.corpus_id for ex in examples],
            touched=touched,
        )
        return gate

    @track_eval_usage("grounding")
    @track_eval_latency("grounding")
    async def _run_all_predictions(
        self, examples: list[GroundingCorpusExample]
    ) -> tuple[dict[tuple[str, int], GroundingResult], set[Path], int]:
        # Pre-populate before mlflow.genai.evaluate (sync) to avoid event-loop boundary issues.
        sem = asyncio.Semaphore(self._config.max_concurrent_predictions)
        agent_hash = self._agent.fingerprint()

        async def _run_one(
            task_idx: int, ex: GroundingCorpusExample, node_idx: int
        ) -> tuple[tuple[str, int], GroundingResult, Path, bool]:
            set_task_num(task_idx)
            cache_fp = build_cache_fp(self._config.grounding_cache_dir, ex, agent_hash=agent_hash, index=node_idx)
            node = ex.nodes[node_idx]

            async with sem:
                result, used, was_cached = await read_or_run(
                    cache_fp,
                    GroundingResult,
                    self._agent.verify(
                        title=node.title,
                        description=node.description,
                        quote=node.quote,
                        transcript=ex.transcript,
                    ),
                )
            return (ex.corpus_id, node_idx), result, used, was_cached

        flat = [(ex, node_idx) for ex in examples for node_idx in range(len(ex.nodes))]
        tasks = [_run_one(task_idx, ex, node_idx) for task_idx, (ex, node_idx) in enumerate(flat)]
        quads = await asyncio.gather(*tasks)
        results = {key: result for key, result, _, _ in quads}
        touched = {used for _, _, used, _ in quads}
        cache_hits = sum(1 for _, _, _, was_cached in quads if was_cached)
        return results, touched, cache_hits


def _build_dataframe(examples: list[GroundingCorpusExample]) -> pd.DataFrame:
    rows = []
    for ex in examples:
        for i, node in enumerate(ex.nodes):
            rows.append(
                {
                    "inputs": {
                        "corpus_id": ex.corpus_id,
                        "node_index": i,
                        "_title": node.title,
                        "_description": node.description,
                        "_quote": node.quote,
                    },
                    "expectations": {"expected_supported": node.expected_supported},
                    "tags": {f"corpus.{k}": str(v) for k, v in ex.tags.items()},
                }
            )
    return pd.DataFrame(rows)


def _aggregate_metrics(eval_result: EvaluationResult) -> dict[str, float]:
    counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    assert eval_result.result_df is not None
    for _, row in eval_result.result_df.iterrows():
        for k in counts:
            v = row.get(f"grounding.{k}/value")
            if v is not None and not pd.isna(v):
                counts[k] += int(v)

    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return {"precision": precision, "recall": recall}


def _build_breakdown(
    examples: list[GroundingCorpusExample],
    result_cache: dict[tuple[str, int], GroundingResult],
) -> dict:
    breakdown: dict = {}
    for ex in examples:
        nodes_out = []
        for i, node in enumerate(ex.nodes):
            result = result_cache.get((ex.corpus_id, i))
            nodes_out.append(
                {
                    "title": node.title,
                    "expected_supported": node.expected_supported,
                    "predicted_supported": result.supported if result else None,
                    "rationale": result.rationale if result else None,
                }
            )
        breakdown[ex.corpus_id] = {"tags": ex.tags, "nodes": nodes_out}
    return breakdown
