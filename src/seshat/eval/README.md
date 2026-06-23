# `seshat.eval` — Evaluation Harnesses

Three eval harnesses that measure quality across the three pipeline passes:
**identification**, **resolution**, and **retrieval**.
All three use [MLflow Evaluate](https://mlflow.org/docs/latest/llms/llm-evaluate/) as the
evaluation framework and write their results to a shared gate file.

## Installation

The eval package requires optional dependencies.  Install them with:

```bash
uv sync --group eval
```

## Architecture

```
eval/
├── identification/       # ExtractionOrchestrator → KBNodes
│   ├── corpus_loader.py    reads YAML fixtures, builds IdentificationCorpusExample
│   ├── matcher.py          greedy bipartite span-overlap matcher
│   ├── scorers.py          precision/recall + field-accuracy feedback
│   └── runner.py           IdentificationEvalRunner
├── resolution/           # ExtractionOrchestrator → relationships
│   ├── corpus_loader.py    reads YAML fixtures, builds KBNodes from slugs
│   ├── scorers.py          exact-triple precision/recall feedback
│   └── runner.py           ResolutionEvalRunner
├── retrieval/            # AbstractVectorStore → top-k search
│   ├── corpus_loader.py    reads YAML fixtures, builds KBNodes
│   ├── scorers.py          recall@5 / precision@5 feedback
│   └── runner.py           RetrievalEvalRunner
├── models.py             # corpus + GateResult Pydantic models
├── gate.py               # read/write/upsert gate file helpers
└── thresholds.py         # pass/fail thresholds (hard-coded, not config)
```

---

## The Gate

Every runner produces a `GateResult` and writes it to the `gate_path` configured in
`EvalConfig` via `upsert_gate`.  `upsert_gate` is additive: running one harness
carries over the other harnesses' metric blocks so you can run them independently
without zeroing out prior results.

`GateResult.passed` computes the overall pass/fail verdict by comparing all present
metric blocks against the thresholds in `thresholds.py`.  A block that is `None`
(not yet run) is skipped — it does not fail the gate.

Thresholds are in code, not config, so that lowering them requires a reviewable code
change.

---

## Harness 1 — Identification (`IdentificationEvalRunner`)

**What it measures:** Can the extraction pipeline find the right nodes in a meeting
transcript and populate their fields correctly?

**How it works:**

1. Loads YAML corpus fixtures from `EvalConfig.identification_corpus_dir`.
2. For each fixture, runs `ExtractionOrchestrator._run_identification` against the
   transcript (async, pre-populated into a cache before MLflow's sync loop runs).
3. Feeds the resulting `KBNode` list through the **greedy bipartite span-overlap matcher**
   in `matcher.py`.  Matching prefers `quote_anchors` spans; falls back to weighted
   title+description fuzzy similarity when no quote anchor is present.
4. The `scorer` function produces per-`ConceptType` precision/recall/F1 feedback, plus
   per-field accuracy scores (fuzzy for `assignee`, `due`, `rationale`, `context`;
   exact for `type`; set-overlap for `alternatives_considered`).
5. Gate metrics: per-type precision and recall (dotted keys like
   `action_item.precision`).  Field-level scores are logged to MLflow as observability
   signals but are **not** gate criteria.

**Corpus fixture format** — one YAML file per meeting scenario:

```yaml
transcript: |
  ...full meeting transcript...
expected:
  action_item:
    - quote: "exact substring from transcript"
      title: "Short label"
      description: "Full description"
      assignee: "Alice"          # optional extra field
      due: "2026-06-01"
  decision:
    - quote: "..."
      title: "..."
      description: "..."
      rationale: "..."
```

**Limitations:**
- Quote matching uses fuzzy partial ratio — a predicted node whose quote is a
  substring of a different sentence can still match if the score exceeds the 0.70
  threshold.
- Field accuracy scores cover only structured fields; semantic quality of `title` and
  `description` is not scored.
- Greedy matching: if two predicted nodes are both good candidates for the same
  expected node, only the higher-scoring one is counted as a true positive.
- Does not test the full ingestion pipeline (no S3, no Postgres, no deduplication).

---

## Harness 2 — Resolution (`ResolutionEvalRunner`)

**What it measures:** Can the pipeline correctly link new nodes to existing KB nodes
via SUPERSEDES, RELATED_TO, or DUPLICATE relationships?

**How it works:**

1. Loads YAML corpus fixtures from `EvalConfig.resolution_corpus_dir`.
2. Converts corpus slugs to fresh UUIDs (deterministic within a run, random across
   runs — fixtures reference nodes by slug, not UUID).
3. For each fixture, calls `ExtractionOrchestrator._run_resolution` with the
   `source_nodes` and a `per_source_targets` map covering all KB nodes in the fixture.
4. The `scorer` computes precision/recall over exact `(source_id, target_id, rel_type)`
   triples.

**Corpus fixture format:**

```yaml
corpus_id: decision-supersedes-001
description: "A revised decision supersedes an earlier one"
source_nodes:
  - id: new-decision
    type: decision
    title: "..."
    description: "..."
    quote: "..."
kb_nodes:
  - id: old-decision
    type: decision
    title: "..."
    description: "..."
    quote: "..."
expected_relations:
  - source: new-decision
    target: old-decision
    rel_type: SUPERSEDES
```

**Limitations:**
- Fixtures set `per_source_targets` to the full KB nodes list for every source node.
  Real runs use a filtered candidate set from the retrieval pass — the eval therefore
  tests resolution quality assuming perfect retrieval, not end-to-end quality.
- Exact triple matching: a RELATED_TO when SUPERSEDES was expected scores zero, even
  if the relationship is directionally correct.
- Does not cover ambiguous cases where multiple valid relationship types exist.

---

## Harness 3 — Retrieval (`RetrievalEvalRunner`)

**What it measures:** Can the vector store surface relevant KB nodes in the top-5
results for a given query node?

**How it works:**

1. Loads YAML corpus fixtures from `EvalConfig.retrieval_corpus_dir`.
2. For each fixture, seeds the `AbstractVectorStore` collection with the candidate
   nodes, issues a `search(title + description, top_k=5)` query, then tears down
   (delete) the candidates — all within a try/finally to avoid leaving stale data.
3. `build_kb_nodes` is called **once** per run: it generates UUIDs via `uuid4()` and
   must not be called twice for the same fixture or the UUID mapping breaks.
4. The `scorer` computes recall@5 and precision@5 against the `expected_relevant_ids`.

**Corpus fixture format:**

```yaml
corpus_id: retrieval-action-001
description: "Should retrieve a related action item"
query_node:
  id: new-action
  type: action_item
  title: "Deploy hotfix"
  description: "Deploy the fix to production"
  quote: "..."
candidate_nodes:
  - id: existing-action
    type: action_item
    title: "Deploy release"
    description: "Deploy the release build"
    quote: "..."
  - id: unrelated-decision
    ...
expected_relevant_ids:
  - existing-action
```

**Limitations:**
- The caller **must** pass a dedicated, empty collection.  Any pre-existing nodes in
  the collection will appear in search results and corrupt scores.
- Seed/teardown is best-effort: exceptions during upsert or delete are suppressed.  A
  failed teardown leaves nodes in the store and will skew subsequent fixtures in the
  same run.
- Evaluates embedding-based retrieval only — does not test re-ranking or any
  post-retrieval filtering.
- Query is constructed as `title + " " + description`; the production pipeline may
  use a different query strategy.
- Gate criterion is recall@5 ≥ 0.70; precision@5 is logged but not gated.
