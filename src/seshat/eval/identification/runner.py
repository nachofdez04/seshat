from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import mlflow
import mlflow.genai
import pandas as pd

from seshat.eval.cache import clear_cache_dir, read_or_run
from seshat.eval.common import log_breakdown_artifact
from seshat.eval.gate import upsert_gate
from seshat.eval.identification.corpus_loader import IdentificationCorpusExample, load_corpus
from seshat.eval.identification.scorers import scorer
from seshat.models.enums import ConceptType
from seshat.models.nodes import IdentificationResult

if TYPE_CHECKING:
    from mlflow.genai.evaluation.entities import EvaluationResult

    from seshat.config.settings import EvalConfig
    from seshat.eval.models import GateResult
    from seshat.models.nodes import KBNode
    from seshat.pipeline.extraction.orchestrator import ExtractionOrchestrator


class IdentificationEvalRunner:
    def __init__(
        self,
        orchestrator: ExtractionOrchestrator,
        config: EvalConfig,
        model_id: str | None = None,
    ) -> None:
        self._orchestrator = orchestrator
        self._config = config
        self._model_id = model_id

    async def run(self, tag_filter: dict[str, str | list[str]] | None = None) -> GateResult:
        mlflow.set_tracking_uri(self._config.observability.mlflow_tracking_uri)
        mlflow.set_experiment(self._config.observability.mlflow_experiment_name)

        examples = load_corpus(self._config.identification_corpus_dir, tag_filter=tag_filter)
        result_cache = await self._run_all_predictions(examples)

        def _predict(transcript: str, corpus_id: str) -> dict:
            if corpus_id not in result_cache:
                raise KeyError(f"corpus_id {corpus_id!r} not found in result cache — mlflow unpacking mismatch")
            return {"nodes": [n.model_dump(mode="json") for n in result_cache[corpus_id].nodes]}

        df = _build_dataframe(examples)
        eval_result = mlflow.genai.evaluate(data=df, predict_fn=_predict, scorers=[scorer], model_id=self._model_id)

        run_id = eval_result.run_id
        identification_metrics = _aggregate_metrics(eval_result)
        self._log_breakdown(eval_result, examples, result_cache, run_id)

        gate = upsert_gate(
            self._config.gate_path,
            run_id=run_id,
            identification_metrics=identification_metrics,
        )
        mlflow.log_metrics({**identification_metrics, "gate.passed": float(gate.passed)}, run_id=run_id)
        if tag_filter:
            mlflow.log_params({f"tag_filter.{k}": str(v) for k, v in tag_filter.items()}, run_id=run_id)
        clear_cache_dir(self._config.identification_cache_dir)
        return gate

    async def _run_all_predictions(
        self, examples: list[IdentificationCorpusExample]
    ) -> dict[str, IdentificationResult]:
        # Pre-populate before mlflow.genai.evaluate (sync). Calling the orchestrator
        # inside _predict would cross event-loop boundaries — LangChain clients are
        # bound to the loop that created them and fail silently from a new thread.
        sem = asyncio.Semaphore(self._config.max_concurrent_predictions)

        async def _run_one(ex: IdentificationCorpusExample) -> tuple[str, IdentificationResult]:
            async with sem:
                result = await read_or_run(
                    self._config.identification_cache_dir / f"{ex.corpus_id}.json",
                    IdentificationResult,
                    self._orchestrator._run_identification(ex.transcript, ex.corpus_id, job_id=ex.corpus_id, hints={}),
                )
            return ex.corpus_id, result

        pairs = await asyncio.gather(*(_run_one(ex) for ex in examples))
        return dict(pairs)

    def _log_breakdown(
        self,
        eval_result: EvaluationResult,
        examples: list[IdentificationCorpusExample],
        result_cache: dict[str, IdentificationResult],
        run_id: str,
    ) -> None:
        log_breakdown_artifact(_build_breakdown(eval_result, examples, result_cache), run_id)


def _build_dataframe(examples: list[IdentificationCorpusExample]) -> pd.DataFrame:
    rows = []
    for ex in examples:
        rows.append(
            {
                "inputs": {"transcript": ex.transcript, "corpus_id": ex.corpus_id},
                "expectations": {"expected_nodes": [n.model_dump(mode="json") for n in ex.expected_nodes]},
            }
        )
    return pd.DataFrame(rows)


def _aggregate_metrics(eval_result: EvaluationResult) -> dict[str, float]:
    """Flatten per-type precision/recall/spurious_rate into dotted keys.

    Field-level scores (assignee, due, rationale, …) are logged to MLflow for diagnosis
    but intentionally excluded from the gate — they are observability signals, not pass/fail criteria.
    """
    result: dict[str, float] = {}
    for ctype in ConceptType:
        p = eval_result.metrics.get(f"{ctype}.precision/mean")
        r = eval_result.metrics.get(f"{ctype}.recall/mean")
        hr = eval_result.metrics.get(f"{ctype}.spurious_rate/mean")
        if p is not None:
            result[f"{ctype}.precision"] = float(p)
        if r is not None:
            result[f"{ctype}.recall"] = float(r)
        if hr is not None:
            result[f"{ctype}.spurious_rate"] = float(hr)
    return result


def _build_breakdown(
    eval_result: EvaluationResult,
    examples: list[IdentificationCorpusExample],
    result_cache: dict[str, IdentificationResult],
) -> dict:
    assert eval_result.result_df is not None
    breakdown: dict[str, dict] = {}
    for ex, (_, row) in zip(examples, eval_result.result_df.iterrows(), strict=True):
        scores: dict[ConceptType, dict[str, float | None]] = {}
        for ctype in ConceptType:
            p = row.get(f"{ctype}.precision/value")
            r = row.get(f"{ctype}.recall/value")
            if pd.isna(p) and pd.isna(r):
                continue
            scores[ctype] = {
                "precision": float(p) if not pd.isna(p) else None,
                "recall": float(r) if not pd.isna(r) else None,
            }

        predicted_nodes = result_cache[ex.corpus_id].nodes
        breakdown[ex.corpus_id] = {
            "tags": ex.tags,
            "scores": scores,
            "expected": [{"type": n.type, "title": n.title, "quote": n.quote} for n in ex.expected_nodes],
            "predicted": [
                {"type": n.type, "title": n.title, "quote": _first_quote(n, ex.transcript)} for n in predicted_nodes
            ],
        }

    return breakdown


def _first_quote(node: KBNode, transcript: str) -> str | None:
    if node.quote_anchors:
        a = node.quote_anchors[0]
        return transcript[a.char_start : a.char_end]
    return None
