from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np

from seshat.app.platform.observability.usage_tracker import track_eval_usage
from seshat.core.models.enums import ConceptType
from seshat.core.models.nodes import IdentificationResult
from seshat.eval.cache import build_cache_fp, read_or_run, sweep_stale_entries
from seshat.eval.calibration.models import (
    IdentificationSweepPoint,
    IdentificationSweepResult,
    TypePC,
)
from seshat.eval.identification.corpus_loader import load_corpus
from seshat.eval.identification.matcher import match_nodes

if TYPE_CHECKING:
    from pathlib import Path

    from seshat.app.pipeline.extraction.orchestrator import ExtractionOrchestrator
    from seshat.core.config.eval_settings import EvalConfig
    from seshat.eval.models import IdentificationCorpusExample

# corpus_id → (pipeline result, corpus example)
type _Cache = dict[str, tuple[IdentificationResult, IdentificationCorpusExample]]

_DEFAULT_P_TARGET = 0.95


class IdentificationMetaScorer:
    def __init__(
        self,
        orchestrator: ExtractionOrchestrator,
        config: EvalConfig,
        step: float = 0.01,
    ) -> None:
        self._orchestrator = orchestrator
        self._config = config
        self._step = step

    async def sweep_threshold(
        self, p_target: float = _DEFAULT_P_TARGET, ignore_grounding: bool = False
    ) -> IdentificationSweepResult:
        """Load corpus results (file cache or pipeline), then sweep heuristics thresholds [0, 1].

        Auto-approval gate:
        - Grounding enabled: node is approved iff grounding == 1 AND heuristics >= threshold.
        - Grounding disabled: node is approved iff heuristics >= threshold.

        suggested_threshold = argmax coverage subject to precision_approved >= p_target.
        p_target is a business policy (minimum acceptable precision on auto-approved nodes).
        Ties resolve to the lower threshold.
        Falls back to argmax precision when no threshold meets p_target.

        Pass ignore_grounding=True to calibrate as if grounding were disabled — useful for
        comparing thresholds and avoiding grounding costs during development.
        """
        cache = await self._build_cache()
        return self._compute_sweep(cache, p_target=p_target, ignore_grounding=ignore_grounding)

    async def precision_coverage_curve(self, ignore_grounding: bool = False) -> list[IdentificationSweepPoint]:
        """Return the full precision-vs-coverage curve across thresholds [0, 1].

        Use this to inspect the precision/coverage tradeoff and choose an appropriate
        p_target before calling sweep_threshold(p_target=...).

        Pass ignore_grounding=True to see the heuristics-only curve even when grounding
        scores are present in the cache.
        """
        cache = await self._build_cache()
        return self._build_curve(cache, ignore_grounding=ignore_grounding)

    def _build_curve(self, cache: _Cache, ignore_grounding: bool = False) -> list[IdentificationSweepPoint]:
        n_points = round(1 / self._step) + 1
        thresholds = np.linspace(0.0, 1.0, n_points).tolist()
        return [_compute_pc_point(cache, t, ignore_grounding=ignore_grounding) for t in thresholds]

    def _compute_sweep(
        self, cache: _Cache, p_target: float = _DEFAULT_P_TARGET, ignore_grounding: bool = False
    ) -> IdentificationSweepResult:
        points = self._build_curve(cache, ignore_grounding=ignore_grounding)

        # argmax coverage subject to precision_approved >= p_target; ties → lower threshold
        eligible = [p for p in points if p.precision_approved >= p_target]
        if eligible:
            best = max(eligible, key=lambda p: (p.coverage, -p.threshold))
        else:
            # no threshold meets p_target — fall back to argmax precision
            best = max(points, key=lambda p: (p.precision_approved, -p.threshold))

        return IdentificationSweepResult(points=points, suggested_threshold=best.threshold)

    @track_eval_usage("identification")
    async def _build_cache(self) -> _Cache:
        """Run identification pipeline once per corpus example; use file cache when available."""
        examples = load_corpus(self._config.identification_corpus_dir)
        cache: _Cache = {}
        sem = asyncio.Semaphore(self._config.max_concurrent_predictions)
        agent_hash = self._orchestrator._identification_registry.fingerprint()

        async def _run_one(ex: IdentificationCorpusExample) -> tuple[IdentificationResult, Path]:
            cache_fp = build_cache_fp(self._config.identification_cache_dir, ex, agent_hash=agent_hash)
            async with sem:
                result, used, _cached = await read_or_run(
                    cache_fp,
                    IdentificationResult,
                    self._orchestrator._run_identification(ex.transcript, ex.corpus_id, job_id=ex.corpus_id, hints={}),
                )
            cache[ex.corpus_id] = (result, ex)
            return result, used

        pairs = await asyncio.gather(*(_run_one(ex) for ex in examples))
        touched = {used for _, used in pairs}
        sweep_stale_entries(
            self._config.identification_cache_dir,
            corpus_ids=[ex.corpus_id for ex in examples],
            touched=touched,
        )
        return cache


def _compute_pc_point(cache: _Cache, threshold: float, ignore_grounding: bool = False) -> IdentificationSweepPoint:
    """Compute precision-of-approved and coverage at one threshold, globally and per type.

    "Gold" = the expected_nodes from the corpus fixture (ground-truth annotations).
    coverage = TP / gold_count (recall: what fraction of expected nodes were correctly approved).
    precision_approved = TP / (TP + FP) (of the approved nodes, how many were correct).
    """
    per_type_tp: defaultdict[ConceptType, int] = defaultdict(int)
    per_type_fp: defaultdict[ConceptType, int] = defaultdict(int)
    per_type_gold: defaultdict[ConceptType, int] = defaultdict(int)

    for _, (result, ex) in cache.items():
        accepted = _filter_by_threshold(result, threshold, ignore_grounding=ignore_grounding)

        # gold count per type (denominator for coverage / recall)
        for node in ex.expected_nodes:
            per_type_gold[node.type] += 1

        match = match_nodes(ex.transcript, ex.expected_nodes, accepted)

        for ct in ConceptType:
            per_type_tp[ct] += sum(1 for m in match.matched if m.predicted.type == ct)
            per_type_fp[ct] += sum(1 for n in match.spurious if n.type == ct)

    per_type: dict[ConceptType, TypePC] = {}
    total_tp = total_fp = total_gold = 0

    for ct in ConceptType:
        tp = per_type_tp[ct]
        fp = per_type_fp[ct]
        gold = per_type_gold[ct]
        approved = tp + fp

        precision = tp / approved if approved > 0 else 1.0
        coverage = tp / gold if gold > 0 else 0.0
        per_type[ct] = TypePC(precision_approved=precision, coverage=coverage)

        total_tp += tp
        total_fp += fp
        total_gold += gold

    total_approved = total_tp + total_fp
    agg_precision = total_tp / total_approved if total_approved > 0 else 1.0
    agg_coverage = total_tp / total_gold if total_gold > 0 else 0.0

    return IdentificationSweepPoint(
        threshold=round(threshold, 10),
        precision_approved=agg_precision,
        coverage=agg_coverage,
        per_type=per_type,
    )


def _filter_by_threshold(result: IdentificationResult, threshold: float, ignore_grounding: bool = False) -> list:
    """Return nodes that pass the auto-approval gate at the given heuristics threshold.

    When grounding scores are present (and ignore_grounding is False), a node must pass both:
      grounding == 1 AND heuristics >= threshold.
    Otherwise a node passes on heuristics alone:
      heuristics >= threshold.
    """
    filtered = []
    for node in result.nodes:
        breakdown = result.confidence_breakdowns.get(node.id)
        if breakdown is None:
            continue

        if not ignore_grounding and breakdown.grounding_passed is False:
            continue

        if breakdown.heuristics >= threshold:
            filtered.append(node)

    return filtered
