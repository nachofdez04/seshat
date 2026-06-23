from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import mlflow
import mlflow.genai
import pandas as pd

from seshat.eval.cache import build_cache_fp, read_or_run, sweep_stale_entries
from seshat.eval.gate import upsert_gate
from seshat.eval.identification.corpus_loader import IdentificationCorpusExample, load_corpus
from seshat.eval.identification.scorers import scorer
from seshat.eval.mlflow_logging import configure_trace_processors, log_eval_run_metadata
from seshat.models.enums import ConceptType
from seshat.models.nodes import IdentificationResult
from seshat.observability.latency_tracker import track_eval_latency
from seshat.observability.usage_tracker import track_eval_usage
from seshat.utils.log import set_task_num

if TYPE_CHECKING:
    from pathlib import Path

    from mlflow.entities.span import LiveSpan
    from mlflow.genai.evaluation.entities import EvaluationResult

    from seshat.config.eval_settings import EvalConfig
    from seshat.eval.corpus_tags import CorpusTagFilter
    from seshat.eval.models import GateResult
    from seshat.models.nodes import KBNode
    from seshat.pipeline.extraction.orchestrator import ExtractionOrchestrator


class IdentificationEvalRunner:
    def __init__(self, orchestrator: ExtractionOrchestrator, config: EvalConfig) -> None:
        self._orchestrator = orchestrator
        self._config = config

    async def run(self, tag_filter: CorpusTagFilter | None = None, model_id: str | None = None) -> GateResult:
        examples = load_corpus(self._config.identification_corpus_dir, tag_filter=tag_filter)
        agent_hash = self._orchestrator._identification_registry.fingerprint()
        result_cache, touched = await self._run_all_predictions(examples, agent_hash=agent_hash)

        expected_by_id = {ex.corpus_id: ex.expected_nodes for ex in examples}

        def _predict(corpus_id: str, transcript: str) -> dict:
            if corpus_id not in result_cache:
                raise KeyError(f"corpus_id {corpus_id!r} not found in result cache — mlflow unpacking mismatch")

            return {
                # used by the scorer
                "nodes": [n.model_dump(mode="json") for n in result_cache[corpus_id].nodes],
                # used for debugging in the MLflow UI (shown in traces output); not part of the scorer input
                "expected_nodes": [n.model_dump(mode="json", exclude={"quote"}) for n in expected_by_id[corpus_id]],
            }

        configure_trace_processors(_slim_output_nodes)
        df = _build_dataframe(examples)
        eval_result = mlflow.genai.evaluate(data=df, predict_fn=_predict, scorers=[scorer], model_id=model_id)

        run_id = eval_result.run_id
        identification_metrics = _aggregate_metrics(eval_result)

        gate = upsert_gate(
            self._config.gate_path,
            run_id=run_id,
            identification_metrics=identification_metrics,
        )
        log_eval_run_metadata(
            run_id=run_id,
            harness="identification",
            gate_passed=gate.passed,
            corpus_dir=self._config.identification_corpus_dir,
            corpus_examples=examples,
            breakdown_artifact=_build_breakdown(eval_result, examples, result_cache),
            tag_filter=tag_filter,
        )

        sweep_stale_entries(
            self._config.identification_cache_dir,
            corpus_ids=[ex.corpus_id for ex in examples],
            touched=touched,
            agent_hash=agent_hash,
        )
        return gate

    @track_eval_usage("identification")
    @track_eval_latency("identification")
    async def _run_all_predictions(
        self, examples: list[IdentificationCorpusExample], agent_hash: str
    ) -> tuple[dict[str, IdentificationResult], set[Path]]:
        # Pre-populate before mlflow.genai.evaluate (sync). Calling the orchestrator
        # inside _predict would cross event-loop boundaries — LangChain clients are
        # bound to the loop that created them and fail silently from a new thread.
        sem = asyncio.Semaphore(self._config.max_concurrent_predictions)

        async def _run_one(task_idx: int, ex: IdentificationCorpusExample) -> tuple[str, IdentificationResult, Path]:
            set_task_num(task_idx)
            cache_fp = build_cache_fp(self._config.identification_cache_dir, ex, agent_hash=agent_hash)
            async with sem:
                result, used = await read_or_run(
                    cache_fp,
                    IdentificationResult,
                    self._orchestrator._run_identification(ex.transcript, ex.corpus_id, job_id=ex.corpus_id, hints={}),
                )
            return ex.corpus_id, result, used

        triples = await asyncio.gather(*(_run_one(i, ex) for i, ex in enumerate(examples)))
        results = {corpus_id: result for corpus_id, result, _ in triples}
        touched = {used for _, _, used in triples}
        return results, touched


def _slim_node(n: dict) -> dict:
    slim = {"type": n["type"], "title": n["title"], "description": n["description"]}

    confidence = n.get("confidence")
    if confidence is not None:
        slim["confidence"] = confidence

    return slim


def _slim_output_nodes(span: LiveSpan) -> None:
    if not span.outputs or not isinstance(span.outputs, dict):
        return

    slimmed = {}
    for k, v in span.outputs.items():
        if isinstance(v, list) and v and isinstance(v[0], dict):
            slimmed[k] = [_slim_node(n) for n in v]
        else:
            slimmed[k] = v
    if slimmed != span.outputs:
        span.set_outputs(slimmed)


def _build_dataframe(examples: list[IdentificationCorpusExample]) -> pd.DataFrame:
    rows = []
    for ex in examples:
        rows.append(
            {
                "inputs": {"transcript": ex.transcript, "corpus_id": ex.corpus_id},
                "expectations": {"expected_nodes": [n.model_dump(mode="json") for n in ex.expected_nodes]},
                "tags": {f"corpus.{k}": str(v) for k, v in ex.tags.items()},
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
