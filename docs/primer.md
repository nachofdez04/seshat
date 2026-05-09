# Seshat — Developer Primer

A narrative introduction to the system. After reading this you should have a working mental model of how Seshat is shaped and why. For a quick-reference decision index see [docs/architecture.md](architecture.md). For an implementation-oriented overview see [docs/seshat-sdd.md](seshat-sdd.md). For full implementation contracts see [docs/superpowers/specs/2026-04-21-seshat-design.md](superpowers/specs/2026-04-21-seshat-design.md).

---

## The Problem

Technical meetings produce decisions that matter — which database to use, which risk to accept, who owns which action. In practice these end up scattered across notes, Slack threads, and memory, with no record of the reasoning behind them.

Seshat ingests a meeting recording (audio or pre-formatted transcript), extracts structured items from it — Architecture Decision Records, risks, agreements, and action items — and writes them to a queryable graph-shaped knowledge base. Relationships between decisions across meetings are tracked explicitly: a later decision can supersede, amend, or conflict with an earlier one.

---

## System Topology

```
Streamlit UI
     │
     ▼
FastAPI  ──────────────► Pipeline Worker
     │                         │
     │                         ▼
     │              ┌──────────────────────┐
     │              │   Storage Layer       │
     │              │  Postgres (ops + KB)  │
     │              │  pgvector             │
     │              │  S3 / LocalStack      │
     └──────────────┘  MLflow               │
                    └──────────────────────┘
```

Five services run in Docker Compose: `api`, `worker`, `streamlit`, `postgres`, `mlflow`, and `localstack` (AWS S3 + Secrets Manager emulation). The `api` and `worker` use the same Docker image with different entrypoints. All pipeline stages run inside the worker.

**Key design principle:** every external dependency — LLM, vector store, KB store, blob store, secrets — is accessed through a factory-created abstract interface. Pipeline stages never import a concrete implementation. This means swapping providers (e.g. Postgres → Neo4j for the KB, asyncio queue → ARQ/Redis) is a config change, not a refactor.

---

## Following a Job End-to-End

### 1. Submission

A user submits a job via the Streamlit UI (Screen 1), which calls `POST /jobs` with a multipart request: the audio or text file plus a `JobSubmissionRequest` JSON document containing the meeting date, participants, and optional per-request config overrides.

Before creating the job the API enforces two rate-limit checks: a per-user hourly cap and a configurable global concurrency cap. Both are simple queries against `ops.jobs` — the Postgres table that is the authoritative source of job state.

If a request includes an `idempotency_key`, the API checks whether a job with that key already exists. If it does and is not `FAILED`, the existing job ID is returned without creating a new job — safe to retry without duplicating work.

### 2. Ingestion and Transcription

The API constructs a `TranscriptDocument` — the pipeline-internal representation — and enqueues it. The job transitions to `TRANSCRIBING`.

For audio input, validation happens at the API boundary in order: size check (streaming, rejects mid-upload if the file exceeds the configured maximum size) then magic byte check (rejects MP3/WAV/M4A imposters). File extension and `Content-Type` are never trusted.

The original file is immediately written to blob storage at `jobs/{date}/{job_id}/raw/input.*`. The transcription stage then downloads it to a system-generated temp path (original filename is never used in any filesystem path), calls `AbstractTranscriptionService.transcribe()`, and writes the plain text output to `raw/transcript.txt`. For pre-formatted text input, the YAML/JSON validator populates `raw_text` directly.

From this point on every stage works from the same `TranscriptDocument` with its `raw_text` field populated.

### 3. Extraction — Two Passes

The job transitions to `EXTRACTING`. This is the most expensive stage and the heart of the system.

**First, the transcript is chunked.** TextTiling (NLTK) detects topic-shift boundaries and produces variable-length, topically coherent chunks. A hard ceiling (`max_chunk_count=50`) prevents cost blowup.

**Pass 1 — fan-out:** for each chunk, four extraction agents run concurrently — one per concept type (`ADR`, `RISK`, `AGREEMENT`, `ACTION_ITEM`). Each agent receives the chunk text, a static system prompt, a small "KB hint" (the most recent same-type nodes already in the knowledge base, to avoid re-extracting known decisions), and any relevant context retrieved from the KB via RAG. Agents return a list of `KBNode` objects with empty relationship lists — relationships are always created in Pass 2.

Before agents run, each one gets a **prompt cache benefit**: system prompts are static per concept type and reused across every job. On Anthropic the `cache_control` header is set explicitly; on OpenAI, prefix caching is automatic.

Outputs from all chunks are merged. Two nodes of the same type are considered duplicates if their titles match exactly (primary criterion) or their embeddings exceed a cosine similarity threshold (fallback). When duplicates are found, the later-in-transcript node survives — it represents the settled outcome of the discussion.

**Pass 2 — RAG + Resolution:** runs once, after the complete merged node list is in memory. This is where relationships are born. For each new node, the RAG service embeds it, runs a vector similarity search against the existing KB, then traverses the graph one hop out from the top matches. Two parallel LLM calls then classify relationships:

- *Same-type resolution* — is this new ADR superseding, amending, or conflicting with an existing one?
- *Cross-type resolution* — does this Risk mitigate an ADR? Does this Agreement support one?

A heuristic validation step follows: malformed relationships (wrong direction, contradictory types) are dropped and logged without failing the job.

### 4. Confidence Scoring

Every node gets a confidence score before the job moves to review. Three signals contribute:

- **Logprobs** (OpenAI only) — token probability of the extracted content
- **Verification agent** — a cheap model from a *different* provider that answers "is this claim well-supported by the source quote?" Using a different provider is required; same-provider verification produces correlated errors.
- **Heuristics** (always active) — spaCy-based rule classifiers measuring quote length, title specificity, and description directness

The formula is a weighted normalised average. Signals that are unavailable (logprobs on Anthropic, verification when not configured) are excluded from both numerator and denominator — the remaining weights redistribute proportionally so the result always lies in [0, 1].

Nodes above `confidence_threshold` get `status=AUTO_APPROVED`. Nodes below it get `status=PENDING_REVIEW` and will need a human decision before they enter the KB.

### 5. Human Review

The job transitions to `AWAITING_REVIEW` and pauses. The Streamlit reviewer screen (Screen 3) shows each pending node alongside its source quote, confidence breakdown, and any resolution candidates — existing KB nodes that the resolution step flagged as potentially affected by an approval decision.

The reviewer approves, rejects, or edits nodes. A bulk-approve threshold rule can be applied first, then per-node decisions override it. Once every pending node has a decision the pipeline resumes.

If `auto_mode=True` is set (operator role only), the `AWAITING_REVIEW` state is skipped entirely — all nodes are auto-approved and the full decision list is logged in MLflow for audit.

### 6. Writing to the Knowledge Base

The job transitions to `WRITING`. Before touching the KB, the full `ExtractionResult` is written to `curated/extraction.json` in blob storage — this is the audit trail and recovery artifact, and it exists even if all nodes were rejected.

Then, for each approved node, a single Postgres transaction writes the KB row (`ops.kb_nodes`) and the vector embedding (pgvector). Both succeed or neither does — there is no partial state. When a new node carries a `SUPERSEDES` or `AMENDS` relationship, the target node's `state` field is updated in the same transaction.

The job reaches `DONE`.

---

## The Knowledge Base Model

The KB is **append-and-state-only**. Once a node is written, its content (`title`, `description`, `source_quote`, `confidence`, `relationships`) is immutable. The only thing that can change on an existing node is its `state` — from `CURRENT` to `AMENDED` or `SUPERSEDED` — when a later meeting's extraction establishes a relationship to it.

This means the full decision history is always preserved in the graph. If a decision was later reversed, the original node still exists in the graph with `state=SUPERSEDED`, linked to the newer node by a `SUPERSEDES` edge. There is no delete, no overwrite.

`CONFLICTS_WITH` relationships are the exception: they do not trigger a state change on either party. Both nodes remain `CURRENT`. A conflict is a signal for human judgment — the reviewer sees it at review time via `resolution_candidates`, and Screen 4 highlights active conflicts in the graph view.

---

## Confidence Scoring and the Release Gate

Confidence scores are only meaningful if they are calibrated. A confidence threshold chosen without measurement is just a number.

`seshat eval` is the calibration tool. It runs the extraction pipeline directly (not through the API worker) against a hand-crafted synthetic corpus of labelled transcripts in `tests/eval/corpus/`. It measures precision and recall per concept type and recall@5 on the retrieval baseline, and writes the results to `data/eval_gate.json`.

The worker reads this file at startup. If it is absent or `passed=false`, the worker refuses to accept jobs. **No real meeting data is processed until `seshat eval` has passed.** The gate can be bypassed via `SESHAT_SKIP_EVAL_GATE=true` but that must be an explicit act.

The same gate enforces the regression rule: any change to an agent system prompt, model, or confidence scoring logic must pass `seshat eval` before it goes to production.

---

## Key Interfaces for Developers

Six abstract base classes define the seams between the pipeline and its infrastructure. You will touch at least one of these on almost any non-trivial change:

| Interface | Where | What it does |
|-----------|-------|-------------|
| `AbstractTaskQueue` | `src/seshat/config/` | Enqueue jobs, get status, cancel. MVP: asyncio. v2: ARQ/Redis. |
| `AbstractTranscriptionService` | `src/seshat/transcription/` | `transcribe(audio_path) → str`. MVP: AssemblyAI. |
| `AbstractKBStore` | `src/seshat/knowledge_store/` | Write/read nodes and relationships, transition state, query by filter. |
| `AbstractVectorStore` | `src/seshat/vector_store/` | Upsert embeddings, similarity search, delete. |
| `AbstractBlobStore` | `src/seshat/blob_store/` | Put/get/exists for artifact paths. |
| `AbstractSecretsProvider` | `src/seshat/secrets/` | `get_secret(key) → str`. Resolved once at startup. |

All methods are async. The pipeline runs in an asyncio context throughout.

---

## Where to Go Next

| Topic | Spec section |
|-------|-------------|
| Audio validation rules and blob path structure | §2 Ingestion & Transcription |
| Full `SeshatConfig` and all config fields | §3 Configuration |
| Two-pass extraction contract, agent registry, prompt injection mitigation | §4 Multi-Agent Extraction |
| RAG retrieval flow, resolution criteria, heuristic validation | §5 RAG + Resolution Layer |
| Postgres schema (`ops.*`), write order, v2 migration paths | §6 Storage Layer |
| API endpoints, job lifecycle, review flow, `ApproveRequest` shape | §8 API Layer & Job Lifecycle |
| MLflow instrumentation, prompt/response artifacts | §9 Observability |
| Labelled corpus format, threshold calibration procedure, release gate | §12 Evaluation Strategy |
