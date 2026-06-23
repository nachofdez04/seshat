from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import mlflow
import mlflow.genai
import pandas as pd
from pydantic import BaseModel

from seshat.agents.identification.base import AnchoredConcept, ConceptModel
from seshat.eval.cache import build_cache_fp, read_or_run, sweep_stale_entries
from seshat.eval.gate import upsert_gate
from seshat.eval.grouping.corpus_loader import load_corpus
from seshat.eval.grouping.scorers import scorer
from seshat.eval.mlflow_logging import log_eval_run_metadata
from seshat.models.enums import ConceptType
from seshat.observability.latency_tracker import track_eval_latency
from seshat.observability.usage_tracker import track_eval_usage
from seshat.utils.log import set_task_num

if TYPE_CHECKING:
    from pathlib import Path

    from mlflow.genai.evaluation.entities import EvaluationResult

    from seshat.agents.identification.grouping import GroupingAgent
    from seshat.config.eval_settings import EvalConfig
    from seshat.eval.corpus_tags import CorpusTagFilter
    from seshat.eval.grouping.corpus_loader import GroupingCorpusExample, GroupingCorpusItem
    from seshat.eval.models import GateResult


class _GroupingCacheEntry(BaseModel):
    """Serialisable cache entry: the predicted grouping as lists of corpus item IDs."""

    groups: list[list[str]]


class _EvalConceptModel(ConceptModel):
    """Minimal ConceptModel populated from a corpus item for eval purposes only."""


class GroupingEvalRunner:
    def __init__(self, agent: GroupingAgent, config: EvalConfig) -> None:
        self._agent = agent
        self._config = config

    async def run(self, tag_filter: CorpusTagFilter | None = None, model_id: str | None = None) -> GateResult:
        examples = load_corpus(self._config.grouping_corpus_dir, tag_filter=tag_filter)
        if not examples:
            return upsert_gate(self._config.gate_path, run_id="grouping-no-corpus")

        result_cache, touched = await self._run_all_predictions(examples)

        expected_by_id = {ex.corpus_id: ex.expected_groups for ex in examples}

        def _predict(corpus_id: str, _items: list[dict]) -> dict:
            if corpus_id not in result_cache:
                raise KeyError(f"corpus_id {corpus_id!r} not in result cache")

            return {
                # used by the scorer
                "predicted_groups": result_cache[corpus_id].groups,
                # used for debugging in the MLflow UI (shown in traces output); not part of the scorer input
                "expected_groups": expected_by_id[corpus_id],
            }

        df = _build_dataframe(examples)
        eval_result = mlflow.genai.evaluate(data=df, predict_fn=_predict, scorers=[scorer], model_id=model_id)

        run_id = eval_result.run_id
        grouping_metrics = _aggregate_metrics(eval_result)

        gate = upsert_gate(
            self._config.gate_path,
            run_id=run_id,
            grouping_metrics=grouping_metrics,
        )

        log_eval_run_metadata(
            run_id=run_id,
            harness="grouping",
            gate_passed=gate.passed,
            corpus_dir=self._config.grouping_corpus_dir,
            corpus_examples=examples,
            breakdown_artifact=_build_breakdown(eval_result, examples, result_cache),
            tag_filter=tag_filter,
        )

        sweep_stale_entries(
            self._config.grouping_cache_dir,
            corpus_ids=[ex.corpus_id for ex in examples],
            touched=touched,
        )
        return gate

    @track_eval_usage("grouping")
    @track_eval_latency("grouping")
    async def _run_all_predictions(
        self, examples: list[GroupingCorpusExample]
    ) -> tuple[dict[str, _GroupingCacheEntry], set[Path]]:
        sem = asyncio.Semaphore(self._config.max_concurrent_predictions)
        agent_hash = self._agent.fingerprint()

        async def _run_one(task_idx: int, ex: GroupingCorpusExample) -> tuple[str, _GroupingCacheEntry, Path]:
            set_task_num(task_idx)
            cache_fp = build_cache_fp(self._config.grouping_cache_dir, ex, agent_hash=agent_hash)
            async with sem:
                result, used = await read_or_run(cache_fp, _GroupingCacheEntry, _run_grouping(self._agent, ex))
            return ex.corpus_id, result, used

        triples = await asyncio.gather(*(_run_one(i, ex) for i, ex in enumerate(examples)))
        results = {corpus_id: result for corpus_id, result, _ in triples}
        touched = {used for _, _, used in triples}
        return results, touched


async def _run_grouping(agent: GroupingAgent, example: GroupingCorpusExample) -> _GroupingCacheEntry:
    """Run the grouping agent and return groups as lists of corpus item IDs."""
    anchored = _build_anchored_concepts(example.items)
    concept_type = ConceptType(example.tags.get("concept_type", "decision"))
    groups = await agent.group(anchored, concept_type)
    # item.title holds the corpus item ID (see _build_anchored_concepts).
    # The grouping agent passes title to the LLM as display text and never interprets it,
    # so using it as a carrier for the corpus ID is safe. Slugs like "kafka-choice" are
    # also readable in the breakdown artifact.
    return _GroupingCacheEntry(groups=[[ac.item.title for ac in group.members] for group in groups])


def _build_anchored_concepts(items: list[GroupingCorpusItem]) -> list[AnchoredConcept]:
    """Store the corpus item ID in item.title for recovery after grouping.
    The LLM sees description as the semantic signal; title is used as the display label."""
    result = []
    for item in items:
        model = _EvalConceptModel(quote=item.quote, title=item.id, description=item.description)
        result.append(AnchoredConcept(item=model, quote_anchor=None))
    return result


def _build_dataframe(examples: list[GroupingCorpusExample]) -> pd.DataFrame:
    rows = []
    for ex in examples:
        rows.append(
            {
                "inputs": {
                    "corpus_id": ex.corpus_id,
                    "_items": [{"id": item.id, "description": item.description} for item in ex.items],
                },
                "expectations": {"expected_groups": ex.expected_groups},
                "tags": {f"corpus.{k}": str(v) for k, v in ex.tags.items()},
            }
        )
    return pd.DataFrame(rows)


def _aggregate_metrics(eval_result: EvaluationResult) -> dict[str, float]:
    result: dict[str, float] = {}
    for key in ("exact_match", "group_hit_rate"):
        v = eval_result.metrics.get(f"grouping.{key}/mean")
        if v is not None:
            result[key] = float(v)
    return result


def _build_breakdown(
    eval_result: EvaluationResult,
    examples: list[GroupingCorpusExample],
    result_cache: dict[str, _GroupingCacheEntry],
) -> dict:
    assert eval_result.result_df is not None
    breakdown: dict = {}
    for ex, (_, row) in zip(examples, eval_result.result_df.iterrows(), strict=True):
        scores: dict[str, float | None] = {}
        for key in ("exact_match", "group_hit_rate"):
            v = row.get(f"grouping.{key}/value")
            scores[key] = float(v) if not pd.isna(v) else None
        breakdown[ex.corpus_id] = {
            "tags": ex.tags,
            "scores": scores,
            "expected_groups": ex.expected_groups,
            "predicted_groups": result_cache[ex.corpus_id].groups if ex.corpus_id in result_cache else None,
        }

    return breakdown
