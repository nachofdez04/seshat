from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import mlflow
import mlflow.genai
import pandas as pd

from seshat.app.platform.observability.latency_tracker import track_eval_latency
from seshat.app.platform.observability.usage_tracker import track_eval_usage
from seshat.core.models.enums import ConceptType
from seshat.core.models.nodes import ResolutionResult
from seshat.core.utils.log import set_task_num
from seshat.eval.cache import build_cache_fp, read_or_run, sweep_stale_entries
from seshat.eval.gate import upsert_gate
from seshat.eval.mlflow_logging import log_eval_run_metadata
from seshat.eval.resolution.corpus_loader import build_kb_nodes, load_corpus
from seshat.eval.resolution.scorers import scorer

if TYPE_CHECKING:
    from pathlib import Path
    from uuid import UUID

    from mlflow.genai.evaluation.entities import EvaluationResult

    from seshat.app.pipeline.extraction.orchestrator import ExtractionOrchestrator
    from seshat.core.config.eval_settings import EvalConfig
    from seshat.core.models.nodes import KBNode
    from seshat.eval.corpus_tags import CorpusTagFilter
    from seshat.eval.models import GateResult, ResolutionCorpusExample, ResolutionCorpusNode


class ResolutionEvalRunner:
    def __init__(self, orchestrator: ExtractionOrchestrator, config: EvalConfig) -> None:
        self._orchestrator = orchestrator
        self._config = config
        self._kb_nodes: dict[str, dict[str, KBNode]] = {}
        self._slug_maps: dict[str, dict[str, UUID]] = {}

    async def run(self, tag_filter: CorpusTagFilter | None = None, model_id: str | None = None) -> GateResult:
        mlflow.autolog(disable=True)
        examples = load_corpus(self._config.resolution_corpus_dir, tag_filter=tag_filter)
        if not examples:
            return upsert_gate(self._config.gate_path, run_id="resolution-no-corpus")

        for ex in examples:
            kb_nodes, slug_map = build_kb_nodes(ex)
            self._kb_nodes[ex.corpus_id] = kb_nodes
            self._slug_maps[ex.corpus_id] = slug_map

        result_cache, touched, agent_hashes, cache_hits = await self._run_all_predictions(examples)

        expected_relations_by_id = {ex.corpus_id: ex.expected_relations for ex in examples}

        def _predict(corpus_id: str, _source_nodes: list[dict], _kb_nodes: list[dict]) -> dict:
            if corpus_id not in result_cache:
                raise KeyError(f"corpus_id {corpus_id!r} not found in result cache — mlflow unpacking mismatch")

            relationships = result_cache[corpus_id].relationships
            uuid_to_slug = {v: k for k, v in self._slug_maps[corpus_id].items()}
            return {
                # used by the scorer
                "relations": [
                    {
                        "source": uuid_to_slug.get(r.source_id, str(r.source_id)),
                        "target": uuid_to_slug.get(r.target_id, str(r.target_id)),
                        "rel_type": r.rel_type,
                    }
                    for r in relationships
                ],
                # used for debugging in the MLflow UI (shown in traces output); not part of the scorer input
                "expected_relations": [
                    {"source": r.source, "target": r.target, "rel_type": r.rel_type.value}
                    for r in expected_relations_by_id[corpus_id]
                ],
            }

        df = _build_dataframe(examples)
        eval_result = mlflow.genai.evaluate(data=df, predict_fn=_predict, scorers=[scorer], model_id=model_id)

        run_id = eval_result.run_id
        resolution_metrics = _aggregate_metrics(eval_result)

        gate = upsert_gate(
            self._config.gate_path,
            run_id=run_id,
            resolution_metrics=resolution_metrics,
        )
        log_eval_run_metadata(
            run_id=run_id,
            harness="resolution",
            gate_passed=gate.passed,
            corpus_dir=self._config.resolution_corpus_dir,
            corpus_examples=examples,
            breakdown_artifact=_build_breakdown(eval_result, examples, result_cache, self._slug_maps),
            tag_filter=tag_filter,
            cache_hits=cache_hits,
            total_predictions=len(examples),
        )

        corpus_ids = [ex.corpus_id for ex in examples]
        for h in agent_hashes:
            sweep_stale_entries(
                self._config.resolution_cache_dir,
                corpus_ids=corpus_ids,
                touched=touched,
                agent_hash=h,
            )
        return gate

    @track_eval_usage("resolution")
    @track_eval_latency("resolution")
    async def _run_all_predictions(
        self, examples: list[ResolutionCorpusExample]
    ) -> tuple[dict[str, ResolutionResult], set[Path], set[str], int]:
        sem = asyncio.Semaphore(self._config.max_concurrent_predictions)

        async def _run_one(task_idx: int, ex: ResolutionCorpusExample) -> tuple[str, ResolutionResult, Path, str, bool]:
            set_task_num(task_idx)
            kb_nodes = self._kb_nodes[ex.corpus_id]
            source_nodes = [kb_nodes[n.id] for n in ex.source_nodes]
            kb_target_nodes = [kb_nodes[n.id] for n in ex.kb_nodes]
            per_source_targets: dict[UUID, list[KBNode]] = {src.id: kb_target_nodes for src in source_nodes}

            agent_hash = self._orchestrator._resolution_registry.fingerprint_for_types(
                source_types={n.type for n in source_nodes}, target_types={n.type for n in kb_target_nodes}
            )
            cache_fp = build_cache_fp(self._config.resolution_cache_dir, ex, agent_hash=agent_hash)

            async with sem:
                result, used, was_cached = await read_or_run(
                    cache_fp,
                    ResolutionResult,
                    self._orchestrator._run_resolution(source_nodes, per_source_targets, job_id=ex.corpus_id),
                )
            return ex.corpus_id, result, used, agent_hash, was_cached

        quints = await asyncio.gather(*(_run_one(i, ex) for i, ex in enumerate(examples)))
        results = {corpus_id: result for corpus_id, result, _, _, _ in quints}
        touched = {used for _, _, used, _, _ in quints}
        agent_hashes = {h for _, _, _, h, _ in quints}
        cache_hits = sum(1 for _, _, _, _, was_cached in quints if was_cached)
        return results, touched, agent_hashes, cache_hits


def _slim_node(n: ResolutionCorpusNode) -> dict:
    return {"id": n.id, "type": n.type.value, "title": n.title, "description": n.description}


def _build_dataframe(examples: list[ResolutionCorpusExample]) -> pd.DataFrame:
    rows = []
    for ex in examples:
        slug_to_type = {n.id: n.type.value for n in ex.source_nodes + ex.kb_nodes}
        rows.append(
            {
                "inputs": {
                    "corpus_id": ex.corpus_id,
                    "_source_nodes": [_slim_node(n) for n in ex.source_nodes],
                    "_kb_nodes": [_slim_node(n) for n in ex.kb_nodes],
                },
                "expectations": {
                    "expected_relations": [
                        {"source": r.source, "target": r.target, "rel_type": r.rel_type.value}
                        for r in ex.expected_relations
                    ],
                    "slug_to_type": slug_to_type,
                },
                "tags": {f"corpus.{k}": str(v) for k, v in ex.tags.items()},
            }
        )
    return pd.DataFrame(rows)


def _aggregate_metrics(eval_result: EvaluationResult) -> dict[str, float]:
    """Flatten per-type precision/recall into dotted keys: '{ctype}.precision', '{ctype}.recall'."""
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
    examples: list[ResolutionCorpusExample],
    result_cache: dict[str, ResolutionResult],
    slug_maps: dict[str, dict[str, UUID]],
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

        uuid_to_slug = {v: k for k, v in slug_maps[ex.corpus_id].items()}
        breakdown[ex.corpus_id] = {
            "tags": ex.tags,
            "group": "same_type" if _is_same_type(ex) else "cross_type",
            "scores": scores,
            "expected": [
                {"source": r.source, "target": r.target, "rel_type": r.rel_type.value} for r in ex.expected_relations
            ],
            "predicted": [
                {
                    "source": uuid_to_slug.get(r.source_id, str(r.source_id)),
                    "target": uuid_to_slug.get(r.target_id, str(r.target_id)),
                    "rel_type": r.rel_type,
                }
                for r in result_cache[ex.corpus_id].relationships
            ],
        }
    return breakdown


def _is_same_type(example: ResolutionCorpusExample) -> bool:
    source_types = {n.type for n in example.source_nodes}
    kb_types = {n.type for n in example.kb_nodes}
    return bool(source_types & kb_types)
