from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import mlflow
import mlflow.genai
import pandas as pd

from seshat.agents.verification import VerificationResult
from seshat.eval.cache import clear_cache_dir, read_or_run
from seshat.eval.common import log_breakdown_artifact
from seshat.eval.gate import upsert_gate
from seshat.eval.verification.corpus_loader import load_corpus
from seshat.eval.verification.scorers import scorer

if TYPE_CHECKING:
    from mlflow.genai.evaluation.entities import EvaluationResult

    from seshat.agents.verification import VerificationAgent
    from seshat.config.settings import EvalConfig
    from seshat.eval.models import GateResult
    from seshat.eval.verification.corpus_loader import VerificationCorpusExample


class VerificationEvalRunner:
    def __init__(self, agent: VerificationAgent, config: EvalConfig, model_id: str | None = None) -> None:
        self._agent = agent
        self._config = config
        self._model_id = model_id

    async def run(self, tag_filter: dict[str, str | list[str]] | None = None) -> GateResult:
        mlflow.set_tracking_uri(self._config.observability.mlflow_tracking_uri)
        mlflow.set_experiment(self._config.observability.mlflow_experiment_name)

        examples = load_corpus(self._config.verification_corpus_dir, tag_filter=tag_filter)
        if not examples:
            return upsert_gate(self._config.gate_path, run_id="verification-no-corpus")

        result_cache = await self._run_all_predictions(examples)

        def _predict(corpus_id: str, node_index: int) -> dict:
            key = (corpus_id, node_index)
            if key not in result_cache:
                raise KeyError(f"key {key!r} not in result cache — mlflow unpacking mismatch")
            return {"supported": result_cache[key].supported}

        df = _build_dataframe(examples)
        eval_result = mlflow.genai.evaluate(data=df, predict_fn=_predict, scorers=[scorer], model_id=self._model_id)

        run_id = eval_result.run_id
        verification_metrics = _aggregate_metrics(eval_result)
        self._log_breakdown(eval_result, examples, result_cache, run_id)

        gate = upsert_gate(
            self._config.gate_path,
            run_id=run_id,
            verification_metrics=verification_metrics,
        )
        mlflow.log_metrics({**verification_metrics, "gate.passed": float(gate.passed)}, run_id=run_id)
        if tag_filter:
            mlflow.log_params({f"tag_filter.{k}": str(v) for k, v in tag_filter.items()}, run_id=run_id)
        clear_cache_dir(self._config.verification_cache_dir)
        return gate

    async def _run_all_predictions(
        self, examples: list[VerificationCorpusExample]
    ) -> dict[tuple[str, int], VerificationResult]:
        # Pre-populate before mlflow.genai.evaluate (sync) to avoid event-loop boundary issues.
        sem = asyncio.Semaphore(self._config.max_concurrent_predictions)

        async def _run_one(ex: VerificationCorpusExample, i: int) -> tuple[tuple[str, int], VerificationResult]:
            node = ex.nodes[i]
            async with sem:
                result = await read_or_run(
                    self._config.verification_cache_dir / f"{ex.corpus_id}_{i}.json",
                    VerificationResult,
                    self._agent.verify(
                        title=node.title,
                        description=node.description,
                        quote=node.quote,
                        transcript=ex.transcript,
                    ),
                )
            return (ex.corpus_id, i), result

        tasks = [_run_one(ex, i) for ex in examples for i in range(len(ex.nodes))]
        pairs = await asyncio.gather(*tasks)
        return dict(pairs)

    def _log_breakdown(
        self,
        eval_result: EvaluationResult,
        examples: list[VerificationCorpusExample],
        result_cache: dict[tuple[str, int], VerificationResult],
        run_id: str,
    ) -> None:
        log_breakdown_artifact(_build_breakdown(examples, result_cache), run_id)


def _build_dataframe(examples: list[VerificationCorpusExample]) -> pd.DataFrame:
    rows = []
    for ex in examples:
        for i, node in enumerate(ex.nodes):
            rows.append(
                {
                    "inputs": {"corpus_id": ex.corpus_id, "node_index": i},
                    "expectations": {"expected_supported": node.expected_supported},
                }
            )
    return pd.DataFrame(rows)


def _aggregate_metrics(eval_result: EvaluationResult) -> dict[str, float]:
    counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    assert eval_result.result_df is not None
    for _, row in eval_result.result_df.iterrows():
        for k in counts:
            v = row.get(f"verification.{k}/value")
            if v is not None and not pd.isna(v):
                counts[k] += int(v)

    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return {"precision": precision, "recall": recall}


def _build_breakdown(
    examples: list[VerificationCorpusExample],
    result_cache: dict[tuple[str, int], VerificationResult],
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
