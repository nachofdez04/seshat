# Seshat — Tuning Package: IdentificationMetaScorer & RetrievalMetaScorer

**Date:** 2026-05-27
**Status:** Draft

---

## 1. Motivation

Two pipeline parameters currently have no empirical basis:

- `confidence_threshold` (default `0.7`) in `seshat.pipeline.extraction.weighted_scorer` — filters
  predicted `KBNode` objects after the identification stage. `ConfidenceWeights` (`{heuristics: 0.30,
  verification: 0.70}`) and `HeuristicsScorer` sub-weights (`_W_QUOTE=0.3`, `_W_TITLE=0.3`,
  `_W_DESC=0.4`) are all hand-tuned placeholders.

- `retrieval_score_threshold` (default `0.5`) in `EvalConfig` — filters vector similarity results
  before they contribute to recall@5/precision@5 in the retrieval eval. Also marked TODO: calibrate.

Both belong to the same concern: empirical threshold calibration against labeled corpora. The
`tuning/` subpackage provides that path, starting with sweep-based calibration and leaving the door
open for weight optimization.

---

## 2. Scope

**In scope:**
- Rename `MetaScorer` → `IdentificationMetaScorer`; rename `meta_scorer.py` → `identification_meta_scorer.py`
- Add `RetrievalMetaScorer` with cache-then-sweep pattern for `score_threshold`
- Threshold sweep: grid over [0, 1], per-point recall@5/precision@5, `suggested_threshold` at argmax recall@5
- Score cache: single vector store seeding pass, replay without embedding calls for threshold sweeping
- Update all import sites of `MetaScorer` to use `IdentificationMetaScorer`

**Out of scope:**
- `IdentificationMetaScorer.fit_weights()` implementation (prerequisite: verification eval harness)
- `HeuristicsScorer` sub-weight tuning (future)
- `ResolutionMetaScorer` (future)
- MLflow integration for sweep results (callers decide what to do with `RetrievalSweepResult`)

---

## 3. Location

```
src/seshat/eval/
  gate.py
  thresholds.py
  models.py
  identification/
  resolution/
  retrieval/
  tuning/
    __init__.py                    # exports IdentificationMetaScorer, RetrievalMetaScorer,
                                   #         IdentificationSweepResult, RetrievalSweepResult
    identification_meta_scorer.py  # renamed from meta_scorer.py; class IdentificationMetaScorer
    retrieval_meta_scorer.py       # new
    models.py
```

**Rename note:** Any code that currently imports `MetaScorer` from `seshat.eval.tuning` must be
updated to import `IdentificationMetaScorer`. This includes tests, scripts, and any caller that
references the old class name or module path `tuning.meta_scorer`.

---

## 4. Data Models (`tuning/models.py`)

### Existing (unchanged)

```python
class TypeMetrics(BaseModel):       # internal
    precision: float
    recall: float
    f1: float

class SweepPoint(BaseModel):        # internal
    threshold: float
    metrics: dict[ConceptType, TypeMetrics]
    macro_f1: float

class IdentificationSweepResult(BaseModel):       # exported
    points: list[SweepPoint]
    suggested_threshold: float      # argmax(macro_f1); ties → lower threshold
```

### New

```python
class RetrievalSweepPoint(BaseModel):   # internal
    threshold: float
    recall_at_5: float
    precision_at_5: float

class RetrievalSweepResult(BaseModel):  # exported
    points: list[RetrievalSweepPoint]   # sorted by threshold ascending
    suggested_threshold: float          # argmax(recall_at_5); ties → lower threshold
```

`TypeMetrics`, `SweepPoint`, `RetrievalSweepPoint` are internal detail. `__init__.py` exports only
`IdentificationMetaScorer`, `RetrievalMetaScorer`, `IdentificationSweepResult`, `RetrievalSweepResult`.

---

## 5. `IdentificationMetaScorer` Interface (`identification_meta_scorer.py`)

Identical to the former `MetaScorer` — this section documents the rename only.

```python
class IdentificationMetaScorer:
    def __init__(
        self,
        orchestrator: ExtractionOrchestrator,
        config: EvalConfig,
        step: float = 0.01,
    ) -> None: ...

    async def build_cache(self) -> None:
        """Run identification pipeline once per corpus example; store raw IdentificationResult.

        Must be called before sweep_threshold(). Re-run whenever ConfidenceWeights change,
        since changed weights produce different confidence scores.
        """

    def sweep_threshold(self) -> IdentificationSweepResult:
        """Replay threshold cutoffs [0, 1] at self.step intervals against the cached scores.

        For each threshold t, a predicted node is accepted if confidence_breakdown.final >= t.
        Accepted nodes are matched against corpus ground truth using match_nodes(); TP/FP/FN
        are accumulated per ConceptType across all corpus examples.

        Requires build_cache() to have been called first.
        """

    def fit_weights(self) -> ConfidenceWeights:
        """Fit logistic regression on (heuristics_score, verification_score) -> is_tp.

        Prerequisites (raises NotImplementedError if not met):
        - Corpus must contain >= MIN_SAMPLES (50) labeled TP/FP pairs after a pipeline run.
        - Verification eval harness must pass its gate (see §8 of
          2026-05-24-seshat-eval-quality-scoring.md).
        """
```

---

## 6. `RetrievalMetaScorer` Interface (`retrieval_meta_scorer.py`)

```python
class RetrievalMetaScorer:
    def __init__(
        self,
        vector_store: AbstractVectorStore,
        config: EvalConfig,
        step: float = 0.01,
    ) -> None: ...

    async def build_cache(self) -> None:
        """Seed each corpus example's candidates, search with score_threshold=None and
        top_k=len(candidates), store list[SearchResult] keyed by corpus_id. Tears down
        candidates after each example.

        Must be called before sweep_threshold(). Re-run if embeddings or candidates change.
        """

    def sweep_threshold(self) -> RetrievalSweepResult:
        """Replay threshold cutoffs [0, 1] at self.step intervals against the cached scores.

        For each threshold t, filter cached SearchResult list to score >= t, take up to
        top-5, compute recall@5 and precision@5 against expected_relevant_ids.

        Requires build_cache() to have been called first.
        """
```

`build_cache()` is async (calls the vector store). `sweep_threshold()` is sync (pure computation).

---

## 7. Sweep Logic

### Cache construction (`build_cache`)

For each `RetrievalCorpusExample` in `config.retrieval_corpus_dir`:
1. Seed candidate nodes into the vector store (reuse `RetrievalEvalRunner`'s seed/teardown helpers).
2. Call `vector_store.search(query_text, top_k=len(candidates), score_threshold=None)`.
3. Store `(list[SearchResult], expected_relevant_ids)` keyed by `corpus_id`.
4. Tear down candidates.

Results come back sorted by score descending from the store. No re-sorting needed at sweep time.

### Threshold replay (`sweep_threshold`)

For threshold `t` in `linspace(0, 1, round(1/step) + 1)`:
1. For each cached `(corpus_id, results, expected_ids)`:
   a. Filter `results` to `score >= t`; take up to first 5.
   b. TP = len(intersection of returned ids and expected_ids).
   c. Compute per-example recall and precision (see conventions below).
2. Average recall and precision across all examples.
3. Emit a `RetrievalSweepPoint`.

`suggested_threshold` is `argmax(recall_at_5)` over sweep points. Ties broken by taking the lower
threshold (more inclusive).

### Metric conventions

| Scenario | recall_at_5 | precision_at_5 |
|---|---|---|
| expected non-empty, results returned | TP / len(expected) | TP / 5 |
| expected non-empty, no results returned | 0.0 | 0.0 |
| expected empty (negative example), results returned | 1.0 | 0.0 |
| expected empty (negative example), no results returned | 1.0 | 1.0 |

Negative examples are never skipped. A threshold that correctly suppresses all results on a negative
example scores 1.0/1.0; one that floods results scores 1.0/0.0, penalizing aggregate precision.

Precision always uses 5 as the denominator (fixed-k convention), regardless of how many results
actually come back after threshold filtering. This is intentional: it penalizes thresholds that
return fewer than 5 results on positive examples.

### Corpus

`data/eval/corpus/retrieval/` (~15 files). `EvalConfig.retrieval_corpus_dir` is set by the caller.

---

## 8. Testing

### Unit tests (`tests/unit/eval/tuning/test_retrieval_meta_scorer.py`)

Construct the cache directly — inject `dict[str, tuple[list[SearchResult], list[str]]]` with
hand-crafted scores. No vector store, no corpus loader, no embedding calls.

Key cases:
- All results above threshold, all relevant → recall=1.0, precision=1.0
- All results above threshold, none relevant → recall=0.0, precision=0.0
- All results below threshold, expected non-empty → recall=0.0, precision=0.0
- Negative example, results above threshold → recall=1.0, precision=0.0
- Negative example, no results above threshold → recall=1.0, precision=1.0
- Partial overlap (1 of 2 relevant returned) → recall=0.5, precision=TP/5
- `suggested_threshold` is argmax(recall_at_5); ties take lower value
- Multiple examples averaged correctly

### Integration tests (`tests/integration/eval/tuning/`, marked `integration`)

Real corpus loader against `data/eval/corpus/retrieval/`, stubbed vector store returning fixed
`SearchResult` lists. Verifies `build_cache()` → `sweep_threshold()` produces a valid
`RetrievalSweepResult`: correct number of points for given step, `suggested_threshold` in [0, 1],
all metric values in [0, 1].

No embedding calls required.

---

## 9. Future Work

- **`IdentificationMetaScorer.fit_weights()`** — unblock after verification eval harness passes gate.
- **`HeuristicsScorer` sub-weight tuning** — extend identification cache to capture per-signal scores
  (`quote_score`, `title_score`, `desc_score`) for independent weight optimization.
- **`ResolutionMetaScorer`** — threshold sweep for the resolution pass once a labeled resolution
  corpus exists.
- **Sweep visualization** — a `print_sweep_table(result)` helper or matplotlib plot; trivial to add.
- **Per-type threshold** — retrieval sweep could be extended to report per-type recommended thresholds
  if the corpus grows to support per-type analysis.
