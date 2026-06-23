from __future__ import annotations

import json
import tempfile
from typing import TYPE_CHECKING

import mlflow
import mlflow.genai
import pandas as pd

from seshat.eval.cache import clear_cache_dir, read_or_run
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
    ) -> None:
        self._orchestrator = orchestrator
        self._config = config

    async def run(self) -> GateResult:
        mlflow.set_tracking_uri(self._config.observability.mlflow_tracking_uri)
        mlflow.set_experiment(self._config.observability.mlflow_experiment_name)

        examples = load_corpus(self._config.identification_corpus_dir)
        result_cache = await self._run_all_predictions(examples)

        def _predict(transcript: str, corpus_id: str) -> dict:
            if corpus_id not in result_cache:
                raise KeyError(f"corpus_id {corpus_id!r} not found in result cache — mlflow unpacking mismatch")
            return {"nodes": [n.model_dump(mode="json") for n in result_cache[corpus_id].nodes]}

        df = _build_dataframe(examples)
        eval_result = mlflow.genai.evaluate(data=df, predict_fn=_predict, scorers=[scorer])

        run_id = eval_result.run_id
        identification_metrics = _aggregate_metrics(eval_result)
        self._log_breakdown(eval_result, examples, result_cache, run_id)

        gate = upsert_gate(
            self._config.gate_path,
            run_id=run_id,
            identification_metrics=identification_metrics,
        )
        mlflow.log_metrics({**identification_metrics, "gate.passed": float(gate.passed)}, run_id=run_id)
        clear_cache_dir(self._config.identification_cache_dir)
        return gate

    async def _run_all_predictions(
        self, examples: list[IdentificationCorpusExample]
    ) -> dict[str, IdentificationResult]:
        # Pre-populate before mlflow.genai.evaluate (sync). Calling the orchestrator
        # inside _predict would cross event-loop boundaries — LangChain clients are
        # bound to the loop that created them and fail silently from a new thread.
        results: dict[str, IdentificationResult] = {}
        for ex in examples:
            results[ex.corpus_id] = await read_or_run(
                self._config.identification_cache_dir / f"{ex.corpus_id}.json",
                IdentificationResult,
                self._orchestrator._run_identification(ex.transcript, ex.corpus_id, job_id=ex.corpus_id, hints={}),
            )
        return results

    def _log_breakdown(
        self,
        eval_result: EvaluationResult,
        examples: list[IdentificationCorpusExample],
        result_cache: dict[str, IdentificationResult],
        run_id: str,
    ) -> None:
        breakdown = _build_breakdown(eval_result, examples, result_cache)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(breakdown, f, indent=2)
            breakdown_path = f.name

        mlflow.log_artifact(breakdown_path, artifact_path="eval", run_id=run_id)


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
    """Flatten per-type precision/recall into dotted keys: '{ctype}.precision', '{ctype}.recall'.

    Field-level scores (assignee, due, rationale, …) are logged to MLflow for diagnosis
    but intentionally excluded from the gate — they are observability signals, not pass/fail criteria.
    """
    result: dict[str, float] = {}
    for ctype in ConceptType:
        p = eval_result.metrics.get(f"{ctype}.precision/mean")
        r = eval_result.metrics.get(f"{ctype}.recall/mean")
        if p is not None:
            result[f"{ctype}.precision"] = float(p)
        if r is not None:
            result[f"{ctype}.recall"] = float(r)
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
