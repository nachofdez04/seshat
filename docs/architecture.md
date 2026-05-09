# Seshat — Architecture Summary

Key design decisions and their rationale. Full detail lives in [docs/superpowers/specs/2026-04-21-seshat-design.md](superpowers/specs/2026-04-21-seshat-design.md).

---

## System Shape

**Pipeline + Async Task Queue (Option B).** `Streamlit UI → FastAPI → Pipeline Worker → Storage Layer`. The pipeline is a sequential set of stages, each independently testable. All external dependencies (LLM, vector store, KB store, secrets) are accessed through factory-created interfaces — pipeline stages never import concrete implementations.

**Queue:** `asyncio` in-memory queue for MVP, behind `AbstractTaskQueue` (ARQ/Redis-compatible contract). The v2 swap to `ARQTaskQueue` is a provider change, not a refactor.

---

## Ingestion & Transcription

- **Two input paths** (audio and pre-formatted text) both normalize to `TranscriptDocument` before the pipeline continues.
- **Audio validation** at the API boundary: size check (streaming, rejects when the file exceeds the configured maximum size before upload completes) → magic byte check. `Content-Type` and file extension are untrusted.
- **Video deferred to v2** — ffmpeg has a known CVE history; audio extraction requires explicit security hardening before it is safe to enable.
- **Diarization deferred to v2** — AssemblyAI is the recommended provider once production audio is available.
- **Blob storage** uses S3 + LocalStack for MVP. Rationale: the thesis exercises a cloud-native path; `AbstractBlobStore` keeps the seam clean. The same Postgres instance used for ops tables also hosts pgvector — eliminating Chroma as a separate service.
- **Blob write timing:** `raw/input.*` and `raw/transcript.txt` written after ingestion; `curated/extraction.json` written at the start of WRITING (before KB writes), so the artifact exists even when all nodes are rejected.

---

## Configuration

- **Single `SeshatConfig` singleton** (pydantic-settings, `env_nested_delimiter="__"`). Only the root inherits from `BaseSettings` to prevent dual env-var resolution paths in nested models.
- **Per-request overrides** are deep-merged onto the singleton into a new object — the singleton is never mutated.
- All provider fields use `StrEnum` with `auto()` — validated at startup.

---

## Multi-Agent Extraction

**Two-pass contract:**

- **Pass 1 — fan-out:** agents run concurrently, one per `ConceptType` per chunk. Agents return `KBNode` objects with `relationships: []`. Any non-empty relationship list is rejected and logged as a hallucination signal. The Action Item agent additionally returns `assignee: str | None` (not a relationship — consumed in Pass 2).
- **Pass 2 — RAG + Resolution:** runs only after the complete merged Pass 1 node list is in memory. Constructs all `KBRelationship` objects, resolves `ASSIGNED_TO` by matching `assignee` against `TranscriptMetadata.participants`.

**Agent registry:** `ConceptType` → registered agent class with its own system prompt. The orchestrator discovers agents at runtime — adding a concept type is adding an agent + registry entry.

**Prompt caching is a first-class requirement** (not an optimisation). OpenAI: automatic prefix caching. Anthropic: explicit `cache_control` headers on the system prompt block.

---

## Chunking & Within-Meeting Deduplication

**Chunking:** TextTiling (NLTK) for MVP. Its suitability for meeting transcripts is unvalidated — if the chunking sanity check (§12) shows systematic mis-segmentation, fall back to `RecursiveCharacterTextSplitter` (LangChain, 500-token windows, 100-token overlap). Diarization-based and semantic chunking remain v2 options.

**Within-meeting deduplication:** two nodes of the same type are merged if their titles exact-match (primary) or their embeddings exceed `merge_similarity_threshold=0.85` (fallback). The surviving node is the one with the highest chunk index — later in the transcript is assumed to reflect the settled outcome. Earlier reversed positions are discarded. No `SUPERSEDES` relationship is created within a single job.

---

## Confidence Scoring

Three signals, weighted and normalised:

```
final = Σ(w_i × s_i) / Σ(w_i)   [unavailable signals excluded from both]
```

| Signal | Availability | Default weight |
|--------|-------------|----------------|
| Logprobs | OpenAI only | 0.50 |
| Verification agent | When configured | 0.35 |
| Heuristics (spaCy) | Always | 0.15 |

**Verification agent must use a different `LLMProvider` than the extraction agent** (enforced by `model_validator`). Same-provider verification produces correlated errors. Weakest valid configuration: Anthropic extraction + no verification agent → heuristics-only; startup warning issued.

---

## RAG + Resolution

**RAG runs after extraction, not before.** Extraction agents receive only a lightweight KB hint (same-type nodes, title + 80-char summary, no embeddings). Full retrieval and relationship resolution happen once all new nodes are extracted.

**Retrieval flow:** embed new node → vector search (top-K=5) → graph traversal (depth=1) on top-K candidates.

**Resolution:** two parallel LLM calls:
- Same-type: classifies each new node as `SUPERSEDES`, `AMENDS`, `CONFLICTS_WITH`, or no relationship against its KB candidates.
- Cross-type: resolves `MITIGATES` (Risk→ADR), `SUPPORTS` (Agreement→ADR), `DEPENDS_ON` (ADR→ADR).

Followed by **heuristic validation** that drops malformed relationships (contradictory types, wrong direction, type schema violations) and logs them without failing the job.

**Tiebreaker (AMENDS vs SUPERSEDES):** prefer `AMENDS` — it is the less destructive classification.

**`CONFLICTS_WITH` does NOT trigger a state transition.** Both nodes remain `NodeState.CURRENT`. Conflict is a graph annotation for human judgment, not an automatic state change.

---

## Node Lifecycle Invariant

The pipeline is **append-and-state-only**. `title`, `description`, `source_quote`, `confidence`, and `relationships` are immutable after creation. The only permitted mutation on an existing node is `update_node_state()` — advancing `state` to `SUPERSEDED` or `AMENDED` when a new node carries a `SUPERSEDES` or `AMENDS` relationship.

**Stale CONFLICTS edges** are never deleted (append-only). They are filtered out at query time: `GET /graph/{node_id}`, impact traversal, and Screen 4 ignore CONFLICTS edges where either party has `state != CURRENT`.

**Vector indexing timing:** vectors are written during the WRITING stage as part of the same Postgres transaction as the KB row — not after extraction. Unreviewed nodes from a job in `AWAITING_REVIEW` are invisible to RAG until the job reaches `DONE`.

**Atomicity:** KB row + vector embedding are a single Postgres transaction. A crash mid-transaction leaves no partial state.

---

## Storage

Three independent factories, no coupling between them:

| Layer | MVP | v2 upgrade |
|-------|-----|-----------|
| KB Store | `PostgresKBStore` (`ops.kb_nodes`, `ops.kb_relationships`) | `Neo4jKBStore` |
| Vector Store | `PGVectorStore` (pgvector + `langchain-postgres`) | Chroma, Qdrant, Weaviate |
| Blob Store | `S3BlobStore` (LocalStack in dev) | AWS S3 in prod |

Both KB and vector stores share the same Postgres instance (different schemas: `ops` vs `store`). Single connection string — no coordination protocol needed.

**Weaviate future:** one `WeaviateStore` class would implement both `AbstractKBStore` and `AbstractVectorStore` — adapter pattern, two interfaces, one instance, no pipeline changes.

---

## Secrets

`AbstractSecretsProvider` with `EnvSecretsProvider` and `AWSSecretsProvider`. Secrets are resolved **once at startup** and cached in-process. If a secret is rotated, the worker must be restarted (acceptable for MVP; v2 adds TTL-based cache). LocalStack emulates AWS Secrets Manager locally.

---

## API & Job Lifecycle

- **Auth:** API key in `X-API-Key` header; bcrypt hashes (cost 12); three roles: `submitter`, `reviewer`, `operator`.
- **Rate limiting:** per-user hourly cap + global concurrency cap — both checked before job creation.
- **Idempotency:** `POST /jobs` deduplication via `UNIQUE` constraint on `idempotency_key`. A failed job with the same key starts a fresh run; an in-progress or completed job returns the existing ID.
- **Review flow:** `AWAITING_REVIEW` pauses the pipeline for human review. `POST /jobs/{id}/approve` accepts bulk-threshold rules (processed first) and per-node decisions (processed second). All-reject is valid — the job transitions to `DONE` with an empty result.
- **Auto-mode:** `operator` role only; all nodes set to `AUTO_APPROVED`; audit trail logged in MLflow.
- **`WRITING` recovery on worker boot:** stranded `WRITING` jobs are detected at startup, marked `FAILED(recoverable=True)`, before the event loop accepts new work.

---

## Prompt Injection Mitigation

1. Structural isolation: user content in `<transcript>` and `<context>` delimiters; `<instructions>` section is the only authoritative block.
2. Output validation: agent responses parsed against `KBNode` schema; non-conforming responses rejected.
3. Source quote verification: `source_quote` must be a substring of `TranscriptDocument.raw_text` (whitespace-normalised); failing nodes rejected before the resolution pass.
4. KB context sanitisation: KB nodes serialised through Pydantic before injection — raw field strings never interpolated into prompt instructions.

---

## Observability

**MLflow 3** as the observability backbone. `mlflow.langchain.autolog()` instruments all LangChain agent calls automatically. Captured: agent identity, usage (tokens, audio seconds), prompt cache hit/miss, latency per stage, confidence distributions, errors and retries.

Prompt/response artifacts are written automatically by autolog and are considered sensitive (may contain transcript excerpts).

---

## Evaluation & Release Gate

**`seshat eval`** is a first-class CLI command that invokes the extraction pipeline directly in-process (bypasses the API worker — breaks the circular dependency with `eval_gate.json`).

**Release gate:** the worker refuses to accept jobs at startup unless `data/eval_gate.json` is present and `passed=true`. Gate conditions (retrieval performance and per-`ConceptType` precision/recall targets) are defined in the design spec (see �12).

**Threshold calibration** requires a minimum of 15 annotated instances per `ConceptType` in the synthetic corpus before the targets are statistically meaningful.

**Regression gate:** any change to agent system prompts, model, or confidence scoring must pass `seshat eval` before promotion.

---

## Deferred to v2 (selected highlights)

| Item | Reason |
|------|--------|
| ARQ/Redis durable queue | In-memory queue sufficient for MVP; `AbstractTaskQueue` interface makes the swap mechanical |
| Neo4j KB store | Meaningful only after KB reaches scale |
| Speaker diarization | Requires production audio samples |
| Video input (ffmpeg) | CVE risk; needs explicit security hardening |
| JWT / SSO authentication | Overkill for ~10-user trusted team |
| `AWAITING_REVIEW` timeout SLA | Requires durable scheduler (ships with ARQ/Redis) |
| Post-approval `PATCH /graph/{node_id}` | Edit at approval time via `edited_content` is sufficient for MVP |
| Reranking (cross-encoder) | Add only if `seshat eval` retrieval baseline shows top-K-only recall@5 < 0.7 |
| Replace verification/resolution agents with ONNX NLI models | Evaluate against LLM baseline using `seshat eval` once MVP data is available |
