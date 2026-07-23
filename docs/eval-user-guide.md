# Seshat Eval User Guide

Seshat ships six evaluation harnesses that measure the quality of the AI pipeline
against a labelled ground-truth corpus:

| Harness | What it measures |
|---------|------------------|
| **identification** | Did extraction find the right nodes in a transcript and fill their fields? |
| **resolution** | Did the pipeline infer the correct relationships between nodes? |
| **retrieval** | Does vector search surface the relevant nodes in the top-5? |
| **grouping** | Does the grouping agent cluster extracted items the way a human would? |
| **grounding** | Does the grounding agent tell a supported quote from a hallucinated one? |
| **transcription** | How accurately does the configured transcription provider recognize the corpus audio? |

Every run produces **two** artifacts:

1. **MLflow metrics and traces** — a browsable run at `http://localhost:5000` with
   per-fixture scores and the full LLM traces behind each prediction. This is where you
   *read and debug* results.
2. **`eval_gate.json`** — a machine-readable pass/fail verdict at the project root, used
   by CI and by the API's startup gate.

> **Two caveats before you start.** Every harness calls a **paid API — the four agent
> harnesses hit live LLM endpoints, and retrieval calls the embedding API (plus an LLM in
> keyword/hybrid mode), while transcription calls its configured speech-to-text provider.**
> Results are also **not fully deterministic** across runs, even at
> `temperature=0`; see the [determinism section](../src/seshat/eval/README.md#eval-determinism)
> in the eval package README for why, and how the corpus is designed around it.

This guide is a linear walkthrough: set up, run with defaults, read the results in
MLflow, change the config, understand the gate, clear the cache, and finally calibrate a
threshold. For per-harness internals, matcher details, and the corpus fixture formats,
see [`src/seshat/eval/README.md`](../src/seshat/eval/README.md) and
[`data/eval/README.md`](../data/eval/README.md).

<details>
<summary><strong>Pre-requisites</strong> (expand to set up)</summary>

### 1. Install the eval dependency group

The eval harnesses need optional dependencies that are not installed by default:

```bash
uv sync --group eval
```

### 2. Provide API keys

The harnesses use the same providers as the runtime pipeline. Eval runs **locally**
(not in Docker), so the keys must be in your **`.env`** file — loaded via `python-dotenv`
when the CLI starts. At minimum you need the embedding provider (retrieval) and the
identification/resolution/grounding agent providers, plus the configured transcription
provider. These are the same keys described in the
[UI testing guide pre-requisites](./ui-testing-guide.md); reuse them.

### 3. Start MLflow

Eval is **not** a Docker service — you run it from your shell with `uv run seshat eval`,
and it always logs to an MLflow server. You do not need Docker for this: run MLflow
directly with

```bash
uvx mlflow server --port 5000
```

Or use the stack's container if you prefer: `docker compose up mlflow`. Either way the CLI
expects the server at `http://localhost:5000` — it checks reachability before doing any
work and exits with a clear error if it cannot connect. Open the URL in a browser to
confirm the UI loads.

### 4. Start Postgres

Bring up Postgres for any eval run:

```bash
docker compose up postgres
```

Strictly, not every harness uses it:

| Harness | Needs Postgres? | Why |
|---------|-----------------|-----|
| retrieval | **Yes** | seeds and searches a real pgvector collection |
| identification, resolution | **Yes, at startup** | the shared orchestrator opens a connection pool on launch (a deliberate fail-fast check) — the eval itself does not read from the KB |
| grouping, grounding, transcription | No | run directly against the labelled corpus; no store involved |

Bringing Postgres up for every run keeps the setup uniform, so that is the recommendation.
You do **not** need `localstack`, the API, or the worker: local runs read secrets straight
from environment variables rather than from Secrets Manager.

</details>

---

## 1. Run with defaults and read results in MLflow

Run a single harness by name:

```bash
uv run seshat eval harness identification
```

The six harness names are `identification`, `resolution`, `retrieval`, `grouping`,
`grounding`, and `transcription`. Start with one — every harness spends on a paid API
per run.

To run the whole suite in one command — the usual way to produce a complete gate — pass
`--all` instead of a name:

```bash
uv run seshat eval harness --all
```

`--all` runs every harness whose `EVAL__RUN_<harness>` flag is enabled (all six are `true`
by default). Set a flag to `false` in `.env` to drop that harness from the suite — e.g.
`EVAL__RUN_GROUNDING=false` to skip grounding. These flags gate `--all` only; running a
harness by name always runs it regardless.

When the run finishes, the CLI has written metrics and traces to MLflow and updated
`eval_gate.json`. Now open the MLflow UI to read them.

### Find your run in MLflow

Open `http://localhost:5000`. Each harness logs to its own **experiment**, named
`seshat-eval-<harness>` — so the run above lands in the `seshat-eval-identification`
experiment. Inside it, runs are named `seshat-eval-<harness>-<timestamp>`; the newest is
your latest run.

### What to look at

1. **Metrics** — the run's metrics panel shows the gated scores (e.g.
   `decision.precision`, `action_item.recall`) plus logged-only signals like per-field
   accuracy. These are the same numbers that land in `eval_gate.json`.
2. **Traces (the payoff)** — open the **Traces** tab. There is **one trace row per corpus
   fixture**. Each trace's output is built for diagnosis: it places the **expected** and
   **predicted** results side by side, so a failing fixture is fully explainable from its
   own row without opening the corpus YAML.
   - **Identification** shows `expected_nodes` next to the predicted `nodes` (with verbose
     fields slimmed away, keeping `type / title / description / confidence`).
   - **Resolution** shows expected and predicted relationships as readable slug triples
     alongside the raw UUID relationships the scorer consumes.
   - **Transcription** shows the reference and provider hypothesis together with the
     per-fixture Word Error Rate (WER).

If a fixture scores lower than expected, the trace row tells you whether the model picked
the wrong type, missed a node, or invented one — before you touch any code.

---

## 2. Run with a different configuration

Eval runs the **same pipeline components** the app runs, so the way you change *what gets
evaluated* is by changing their config in `.env` — the model an agent uses, the retrieval
search mode, whether the reranker fires. The two roots split by prefix:

> **`SeshatConfig` vars have no prefix; `EvalConfig` vars need the `EVAL__` prefix.** Both
> load from the same `.env`, with `__` as the nesting delimiter. The knobs that change
> *what a harness produces* (models, search) are **`SeshatConfig`** vars — no prefix. Only
> the eval-harness settings (gate path, per-mode thresholds) take `EVAL__`.

Edit `.env`, then re-run the harness. The useful knobs, per harness:

**Swap the model or provider for an agent harness.** Each agent reads its own block:

```bash
# identification harness (and grouping, which reuses the identification LLM)
EXTRACTION__IDENTIFICATION__PROVIDER=anthropic
EXTRACTION__IDENTIFICATION__MODEL=claude-sonnet-5
EXTRACTION__IDENTIFICATION__TEMPERATURE=0

# grounding harness
EXTRACTION__GROUNDING__MODEL=claude-haiku-4-5-20251001
```

Two things worth knowing: the **grouping** harness runs on the *identification* LLM config
(there is no separate grouping model), and the **identification** harness force-disables
grounding — so `EXTRACTION__GROUNDING__*` affects only the grounding harness, never
identification.

**Toggle identification self-review.** The reflective second pass is a real behavior
change the identification harness picks up:

```bash
EXTRACTION__IDENTIFICATION_SELF_REVIEW__ENABLED=true
```

**Change retrieval search behavior.** The retrieval harness builds the same
`SearchEngine` the app uses, so its mode, reranker, and multi-query knobs all apply.
`RAG__SEARCH_MODE` picks the strategy; each mode has its own score scale, so the pass
threshold is set per mode under `EVAL__` (this one *is* an eval var):

```bash
RAG__SEARCH_MODE=hybrid                           # semantic | keyword | hybrid
RAG__RERANKER__PROVIDER=cohere                    # provider + model both required to enable reranking
RAG__RERANKER__MODEL=rerank-v3.5
EVAL__RETRIEVAL_SCORE_THRESHOLDS__SEMANTIC=0.77
EVAL__RETRIEVAL_SCORE_THRESHOLDS__HYBRID=0.5
```

**Compare transcription providers.** The configured provider owns the gate result; the
language is shared across comparison runs, while each other provider resolves its own
default model and API-key setting:

```bash
TRANSCRIPTION__PROVIDER=assemblyai
TRANSCRIPTION__LANGUAGE=en

uv run seshat eval harness transcription --provider assemblyai --provider openai
```

AssemblyAI and OpenAI are currently supported. Each `--provider` value creates a separate
MLflow child run. Only the provider selected by `TRANSCRIPTION__PROVIDER` updates
`eval_gate.json`; the others are comparison-only runs and report their own
`harness.passed` result without replacing the global gate.

> Changing a **threshold** does not invalidate cached predictions the way a model or
> prompt change does — so after retuning one you often need to clear the cache. See
> [section 4](#4-clear-the-eval-cache).

**Run against a subset of the corpus.** Filter fixtures by any tag with a repeatable
`--tag key=value` flag (this narrows the corpus without touching config):

```bash
uv run seshat eval harness identification --tag tier=happy --tag difficulty=easy
```

Which harnesses `--all` runs is controlled separately by the `EVAL__RUN_<harness>` flags —
see [section 1](#1-run-with-defaults-and-read-results-in-mlflow). For the full variable
reference, see [`configuration.md`](./configuration.md) (the
[`SeshatConfig`](./configuration.md#seshatconfig) blocks for models and RAG, and
[`EvalConfig`](./configuration.md#evalconfig) for the eval-only settings).

---

## 3. The gate file

Every run writes a `GateResult` to **`eval_gate.json`** at the project root (configurable
via `EVAL__GATE_PATH`). It is the machine-readable verdict — a trimmed example:

```json
{
  "run_id": "6bdb0b10ee3343af93a20324815dd3e7",
  "timestamp": "2026-07-14T14:51:57.088272+00:00",
  "identification_metrics": {
    "decision.precision": { "value": 0.964, "passed": true },
    "decision.recall":    { "value": 0.893, "passed": true }
  },
  "retrieval_metrics": {
    "recall_at_5": { "value": 0.742, "passed": true },
    "mrr_at_5":    { "value": 0.944, "passed": true }
  },
  "transcription_metrics": {
    "wer":       { "value": 0.117, "passed": true },
    "wer_macro": { "value": 0.121, "gated": false, "passed": null }
  },
  "validation_hash": "bc7fa1e0330eebbb",
  "passed": true
}
```

Three things to know:

- **The file is additive.** Running one harness updates only its own metric block and
  carries the others over from the previous file. So you can run harnesses independently
  and the gate accumulates results — `passed` is the AND of every block present. A block
  that has never run is `null` and is skipped (it does not fail the gate), but if *no*
  block has ever run, `passed` is `false`.
- **Thresholds live in code, not config.** The pass/fail bars are in
  [`src/seshat/eval/thresholds.py`](../src/seshat/eval/thresholds.py). Lowering a bar
  therefore requires a reviewable code change — you cannot quietly relax the gate through
  an env var.
- **The file is tamper-evident.** Each result carries a `validation_hash` over its
  contents. If you hand-edit a metric, the hash no longer matches and the pipeline refuses
  to read the file. Regenerate it by re-running eval, never by editing.

The API enforces this gate at startup and refuses to boot on a failing (or missing) gate;
`API__SKIP_EVAL_GATE=true` bypasses the check (the Docker stack sets this so the app can
start without a fresh eval run).

---

## 4. Clear the eval cache

To avoid paying for the same LLM calls on every iteration, the harnesses cache each
prediction to disk under `.seshat/eval_cache/<harness>/`. Cache files are keyed on the
corpus example, the agent's prompt fingerprint, and the input hash — so **changing a
prompt or a fixture automatically invalidates** the affected entries on the next run.
For transcription, the key includes provider, model, language, and the audio SHA-256, so
provider comparisons never reuse each other's hypotheses.

What is **not** captured by that key is a **threshold or other config change** (for
example, recalibrating the retrieval semantic threshold, which is baked into cached hybrid
results). After a change like that, cached results are stale and must be cleared manually.

Clear everything, or a single harness:

```bash
# Clear all harness caches
uv run seshat eval clear-cache

# Clear just one harness
uv run seshat eval clear-cache retrieval
```

To clear and immediately re-run in one step, pass `--clear-cache` to the harness command
— it wipes that harness's cache first, then runs fresh:

```bash
uv run seshat eval harness retrieval --clear-cache
```

---

## 5. Calibration, and how it relates to eval

The harnesses **measure** quality at a *fixed* threshold. Calibration does the opposite:
it **finds** the threshold. A component and its harness share the same on-disk prediction
cache (`calibrate retrieval` and `harness retrieval` read and write the same
`.seshat/eval_cache/retrieval/` entries), so a calibration sweep reuses whatever a harness
has already computed — it does not pay for the LLM calls again. `calibrate` accepts the
same `--clear-cache` flag as `harness` when you need a clean sweep.

Two components can be calibrated: **retrieval** (the vector-search score cutoff) and
**identification** (the confidence cutoff that routes a node to auto-approve vs. human
review).

### Worked example: calibrate the retrieval threshold

```bash
uv run seshat eval calibrate retrieval
```

This loads every retrieval fixture (from cache where possible), sweeps the
`score_threshold` across `[0, 1]`, and reports a **suggested threshold** — the value that
maximises macro-F2 across the corpus. Results are logged to MLflow under the
`seshat-eval-retrieval-calibration` experiment.

Close the loop by feeding the suggestion back into eval:

1. Copy the suggested value into `.env`:
   ```bash
   EVAL__RETRIEVAL_SCORE_THRESHOLDS__SEMANTIC=0.81
   ```
2. Re-run the retrieval harness to confirm the gate still passes. Because you changed a
   threshold (which the cache key does not capture — see
   [section 4](#4-clear-the-eval-cache)), clear the stale cache in the same command with
   `--clear-cache`:
   ```bash
   uv run seshat eval harness retrieval --clear-cache
   ```

The identification calibrator works the same way but tunes the auto-approve confidence
cutoff; its `--pc-curve` flag prints the full precision-coverage tradeoff instead of a
single suggestion. For the meta-scorer internals — why F1 is the wrong metric for the
identification cutoff, how macro-F2 is computed for retrieval, and the full flag set — see
the [calibration section of the eval package README](../src/seshat/eval/README.md#calibration-meta-scorers).
