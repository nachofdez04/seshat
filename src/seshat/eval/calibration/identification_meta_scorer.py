from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np

from seshat.eval.calibration.models import IdentificationSweepResult, SweepPoint, TypeMetrics
from seshat.eval.identification.corpus_loader import load_corpus
from seshat.eval.identification.matcher import match_nodes
from seshat.models.enums import ConceptType

if TYPE_CHECKING:
    from seshat.config.settings import ConfidenceWeights, EvalConfig
    from seshat.eval.models import IdentificationCorpusExample
    from seshat.models.nodes import IdentificationResult
    from seshat.pipeline.extraction.orchestrator import ExtractionOrchestrator

# corpus_id → (pipeline result, corpus example)
type _Cache = dict[str, tuple[IdentificationResult, IdentificationCorpusExample]]


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
        self._cache: _Cache | None = None

    async def build_cache(self) -> None:
        """Run identification pipeline once per corpus example; cache IdentificationResult."""
        examples = load_corpus(self._config.identification_corpus_dir)
        cache: _Cache = {}
        sem = asyncio.Semaphore(self._config.max_concurrent_predictions)

        async def _run_one(ex: IdentificationCorpusExample) -> None:
            async with sem:
                result = await self._orchestrator._run_identification(
                    ex.transcript, ex.corpus_id, job_id=ex.corpus_id, hints={}
                )
            cache[ex.corpus_id] = (result, ex)

        await asyncio.gather(*(_run_one(ex) for ex in examples))
        self._cache = cache

    def sweep_threshold(self) -> IdentificationSweepResult:
        """Replay confidence_breakdown.final threshold cutoffs [0, 1] at step intervals."""
        if self._cache is None:
            raise RuntimeError("build_cache() must be called before sweep_threshold()")

        n_points = round(1 / self._step) + 1
        thresholds = np.linspace(0.0, 1.0, n_points).tolist()
        points: list[SweepPoint] = [_compute_metrics(self._cache, t) for t in thresholds]

        # argmax macro_f1; ties → lower threshold (np.argmax returns first occurrence, grid is ascending)
        best_idx = int(np.argmax([p.macro_f1 for p in points]))
        return IdentificationSweepResult(points=points, suggested_threshold=points[best_idx].threshold)

    def fit_weights(self) -> ConfidenceWeights:
        raise NotImplementedError("fit_weights() requires the verification gate to pass first.")


def _compute_metrics(cache: _Cache, threshold: float) -> SweepPoint:
    """Return a SweepPoint with per-type P/R/F1 and macro_f1 for one threshold value."""
    per_type_tp: defaultdict[ConceptType, int] = defaultdict(int)
    per_type_fp: defaultdict[ConceptType, int] = defaultdict(int)
    per_type_fn: defaultdict[ConceptType, int] = defaultdict(int)

    for _, (result, ex) in cache.items():
        accepted = _filter_by_threshold(result, threshold)
        match = match_nodes(ex.transcript, ex.expected_nodes, accepted)
        for ct in ConceptType:
            per_type_tp[ct] += sum(1 for m in match.matched if m.predicted.type == ct)
            per_type_fp[ct] += sum(1 for n in match.spurious if n.type == ct)
            per_type_fn[ct] += sum(1 for n in match.missed if n.type == ct)

    metrics: dict[ConceptType, TypeMetrics] = {}
    f1s: list[float] = []
    for ct in ConceptType:
        tp, fp, fn = per_type_tp[ct], per_type_fp[ct], per_type_fn[ct]
        if tp + fp + fn == 0:
            continue
        p = tp / (tp + fp) if tp + fp > 0 else 0.0
        r = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * p * r / (p + r) if p + r > 0 else 0.0
        metrics[ct] = TypeMetrics(precision=p, recall=r, f1=f1)
        f1s.append(f1)

    macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0
    return SweepPoint(threshold=round(threshold, 10), metrics=metrics, macro_f1=macro_f1)


def _filter_by_threshold(result: IdentificationResult, threshold: float) -> list:
    """Return nodes whose confidence_breakdown.final >= threshold."""
    accepted = []
    for node in result.nodes:
        bd = result.confidence_breakdowns.get(node.id)
        if bd is not None and bd.final >= threshold:
            accepted.append(node)
    return accepted
