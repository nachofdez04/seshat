# Seshat — Design Spec

**Date:** 2026-04-21
**Status:** Approved

## Overview

Seshat is an API-first GenAI application that transcribes technical meeting recordings, extracts structured decisions (ADRs, risks, agreements, action items) using a multi-agent pipeline, and writes them to a graph-shaped knowledge base. It is designed for technical users (staff engineers, data architects, heads of engineering), to help them document architecture decisions, and keep them updated.

---

## 0. Problem, Goals, and Success Criteria

### Problem

Technical council members make architecture decisions, surface risks, and assign actions in meetings. These are currently unrecorded or captured informally — they scatter across notes, Slack threads, and memory. There is no searchable, structured record of why a decision was made, what risks were considered, or what was agreed.

### Goals

- Extract structured decisions (ADRs, risks, agreements, action items) from meeting recordings and write them to a queryable, graph-shaped knowledge base.
- Surface relationships between decisions across meetings (supersession, amendment, conflict, dependency).
- Provide a human review step before any node enters the knowledge base, with confidence scoring to guide reviewer attention.
- Enable seeding the KB from an existing documentation corpus (`seshat init`).

### Non-Goals (MVP)

- Video input (ffmpeg dependency — deferred to v2)
- Speaker diarization (deferred to v2)
- Multi-tenant or SSO authentication (API keys sufficient for MVP user base)
- Production cloud deployment (runs locally against LocalStack only)
- Integration with Notion, Confluence, or Neo4j (v2 upgrade paths)
- Any UI beyond Streamlit
- Post-approval node editing (`PATCH /graph/{node_id}` — deferred to v2)

### Success Criteria

1. `seshat eval` passes the release gate: recall@5 ≥ 0.7 and per-type precision/recall targets met (§12) — no real meeting data is processed until this gate is cleared.
2. A reviewer can process a meeting end-to-end — submit → transcribe → extract → review → KB written — using the Streamlit UI without touching the API directly.
3. The KB is queryable and returns nodes consistent with what was discussed in the source meeting, traceable via `source_quote`.

---

## 1. Architecture

Option B — Pipeline + Async Task Queue (asyncio for MVP, ARQ/Redis for v2).

```
Streamlit UI → FastAPI → Pipeline Worker → Storage Layer
                    ↑                             │
                    └────────── Results ──────────┘
```

- The Streamlit UI communicates exclusively with FastAPI
- The pipeline is a sequential set of stages; each stage is independently testable
- LLM, vector store, transcription, and secrets are accessed through factory-created interfaces — pipeline stages never import concrete implementations for these
- Blob storage and KB storage use concrete classes directly (`S3BlobStore`, `PostgresKBStore`) — both have single MVP implementations and no planned v2 provider swap within the same process
- The queue system connecting `api` to `worker` is `AsyncioTaskQueue` — in-memory, no external dependencies. The v2 upgrade to `ARQTaskQueue` (durable, Redis-backed) is a one-line swap in the worker entrypoint; `AsyncioTaskQueue` exposes the same `enqueue` / `get_status` / `cancel` methods so the call sites do not change.

---

## 2. Ingestion & Transcription

### Meeting Recording Pipeline

Two input paths normalise to a shared `TranscriptDocument` before the pipeline continues:

```
Audio file (.mp3/.wav/.m4a)    → Audio Validator → Transcription Service → TranscriptDocument
Pre-formatted text (YAML/JSON) → Text Validator                          → TranscriptDocument
```

**Audio intake validation:** enforced in order at the API boundary, before any file is buffered to disk or uploaded to S3:

1. **Size check** — the upload is read in chunks; if the running byte count exceeds `TranscriptionConfig.max_file_bytes` (default 500 MB) before the upload completes, the connection is rejected immediately with HTTP 413. The server drains the remaining request body before closing the connection to avoid leaving the client hanging.
2. **Magic byte check** — once the upload is complete, the first 16 bytes are inspected. If the signature does not match an allowed audio format, the file is rejected with HTTP 400. Allowed signatures: MP3 (`ID3` or `\xFF\xFB`), WAV (`RIFF....WAVE`), M4A (`ftyp` box at offset 4 with `M4A ` brand). Do not rely on `Content-Type` or file extension — both are caller-controlled.

> **v2 — video input:** video files (.mp4/.mkv/.webm) require ffmpeg audio extraction before transcription. ffmpeg has a known CVE history and is a significant attack surface. Deferred to v2 with explicit security hardening: magic byte validation, subprocess timeout, and system-generated temp filenames (no original filename used in any filesystem path).

### TranscriptDocument

`TranscriptDocument` is the pipeline-internal representation — constructed by the API layer after accepting a `JobSubmissionRequest` (Section 8) and passed from stage to stage.

```python
class TranscriptDocument(BaseModel):
    id: UUID = Field(default_factory=uuid4)  # auto-generated at construction time
    idempotency_key: str | None      # echoed from JobSubmissionRequest; used for deduplication on POST /jobs
    schema_version: str = "1.0"
    source_type: Literal["audio", "text"]   # "video" deferred to v2 (ffmpeg dependency)
    raw_text: str                    # populated by the transcription stage; or by the text validator for source_type="text"
    metadata: TranscriptMetadata

class TranscriptMetadata(BaseModel):
    meeting_date: date
    participants: list[str] | None = None  # caller-supplied; required for ASSIGNED_TO resolution
    duration: timedelta | None = None
    language: str = "en"
    turns: list[Turn] | None = None   # reserved for diarization (v2)
```

### Transcription

Provider selection happens at startup via the transcription factory reading `TranscriptionConfig.provider`. The pipeline stage calls `transcription_service.transcribe(audio_path) -> str` only.

**Blob download before transcription:** the uploaded audio file is stored in blob storage as `jobs/{meeting_date}/{job_id}/raw/input.*` immediately after ingestion validation. The transcription stage must download it to a system-generated temporary file path before invoking `transcribe()`. The original filename must never be used in any filesystem path. The temporary file is deleted after `transcribe()` returns (success or failure).

```python
class AbstractTranscriptionService(ABC):
    async def transcribe(self, audio_path: str) -> str: ...
    # Returns the plain-text transcript. Diarization output (speaker turns) is reserved for v2.
```

**Diarization:** skipped for MVP. `turns` field reserved for when AssemblyAI (recommended provider) is configured — best-in-class speaker diarization in a single API call.

### Blob Storage

All raw and curated artifacts are stored in blob storage via `S3BlobStore`, backed by LocalStack in development. Folder structure uses meeting date + job ID for human-readable, chronologically browsable paths:

```
seshat-mvp/
  jobs/
    {meeting_date}/         # e.g. 2026-04-22
      {job_id}/
        raw/
          input.mp3         # original uploaded file (audio or text, original extension; video deferred to v2)
          transcript.txt    # raw transcription output (plain text)
        curated/
          extraction.json   # full ExtractionResult (all nodes + relationships)
  init/
    {job_id}/               # UUID generated at seshat init time
      source/               # copy of the markdown files fed in (data lineage)
      curated/
        extraction.json
```

Artifacts are written at two points per path:

**Regular job (`jobs/`):**
1. **After ingestion** — `raw/input.*` (original file) and `raw/transcript.txt` (normalised plain text output). For `source_type="text"`, `input.*` is the uploaded YAML/JSON file and `transcript.txt` is the `content` field extracted by the validator.
2. **At the start of WRITING** — `curated/extraction.json` is written unconditionally at the beginning of the WRITING stage, before any KB writes. It contains the full `ExtractionResult` including all nodes with their final `status` values (`AUTO_APPROVED`, `PENDING_REVIEW`, or `REJECTED`). This means the artifact is always present after a job completes, including the all-reject case — which is precisely when a complete audit trail matters most.

**Init pipeline (`init/`):**
1. **After corpus load (step 1)** — `init/{job_id}/source/` is written immediately after all markdown files are loaded, before any LLM calls. Present even for runs the user later aborts; skipped only for `--dry-run`.
2. **After extraction (step 4)** — `init/{job_id}/curated/extraction.json` is written immediately after extraction completes, before the summary is printed and before the user is prompted. Present even for runs the user subsequently rejects — the audit trail is established as soon as extraction completes, which is precisely when it matters most.

This provides a full recovery path (reprocess from raw transcript without re-transcribing) and an audit trail independent of the KB store. Per-node `.md` files are not written — the KB is Postgres-backed and the `extraction.json` is the complete audit artifact.

> **Scope note:** production hardening (private bucket policy, SSE-S3 encryption, IAM service identities, lifecycle rules) is out of scope — this system runs locally against LocalStack only. The bucket structure above is an `S3BlobStore` implementation detail. **Pre-production checklist:** if this system is ever pointed at real AWS S3 (not LocalStack), SSE-KMS encryption and a private bucket policy blocking public access must be enabled before any real meeting content is stored. This is not optional — raw transcripts and extracted decisions are sensitive by default.

> **Why S3 + LocalStack for MVP (not local filesystem):** the master's programme includes a cloud module, and the thesis intentionally exercises a cloud-native persistence path. S3 is the MVP blob store, and LocalStack runs the AWS APIs locally so the same code paths execute in dev and (hypothetically) prod — no dual "local FS vs S3" branch to maintain. The same rationale applies to using AWS Secrets Manager (via LocalStack) instead of falling back to `EnvSecretsResolver` for all local development.

### Text Input Schema

Pre-formatted text input must conform to a defined YAML/JSON schema with required fields: `date`, `content`. `participants` is optional — include it when known to enable `ASSIGNED_TO` relationship extraction. The validator rejects non-conforming input at the boundary.

### Init Pipeline (KB Seeding)

A separate CLI-only path for seeding the KB and vector store from an existing documentation corpus — not API-accessible, not surfaced in the Streamlit UI.

```
Document corpus (markdown files)
      │
      ▼
AbstractDocumentLoader
      │
      └── MarkdownDocumentLoader   # MVP (NotionLoader, ConfluenceLoader — v2)
      │
      ▼
Extraction Pipeline (same agents as meeting pipeline)
      │
      ▼
Init Summary (stdout)
      │
  [user confirms]
      │
      ▼
KB Store + Vector Store   (nodes written as AUTO_APPROVED, ingestion_source=INIT)
```

**Command:**
```
seshat init --source ./docs/           # full run
seshat init --source ./docs/ --dry-run # scope estimate only — no LLM calls, nothing written
seshat init --source ./docs/ --force   # run even if the KB is already populated (see step 0)
```

**Flow:**
0. **Populated-KB guard:** queries `ops.kb_nodes` for any existing rows. If any are found, the command aborts with a clear error: `"KB already contains N nodes. Use --force to run anyway, or --dry-run to inspect what would be written."` This prevents accidental re-seeding of an already-populated KB. `--force` skips this check and proceeds; `--dry-run` also skips it (a dry run never writes anything so the risk is zero). `max_concurrent_init_runs` is checked here too — if an `ops.init_runs` entry with `status=running` already exists, the command aborts regardless of `--force`.
1. Discovers and loads all markdown files under `--source` recursively. **Blob write:** `init/{job_id}/source/` — a verbatim copy of every loaded file — is written to blob storage immediately after load, before any LLM calls. This mirrors the `raw/input.*` write in the regular pipeline: data lineage is established as soon as the input is known, independently of whether extraction succeeds or the user aborts. Written even when `--dry-run` is not set; skipped for `--dry-run` (nothing is written on a dry run).
2. Chunks the corpus and prints a pre-flight scope estimate to stdout — **no LLM calls yet**:
   - File count and estimated chunk count
   - Estimated input tokens (chunks × 4 agents × avg prompt size) vs `ExtractionConfig.max_total_input_tokens`
   - Estimated output tokens vs `ExtractionConfig.max_total_output_tokens`
3. If `--dry-run`: exits here. Nothing written, no LLM calls made.
4. Runs the same multi-agent extraction pipeline used for meeting recordings — no special agents or prompts. **Blob write:** `init/{job_id}/curated/extraction.json` — the full `ExtractionResult` — is written immediately after extraction completes, before the summary is printed and before the user is prompted. This mirrors the `curated/extraction.json` write at the start of WRITING in the regular pipeline: the audit trail is always present from the moment extraction completes, regardless of whether the user subsequently approves or aborts. If the user types `N`, the `ExtractionResult` remains in blob storage for inspection.
5. Prints an extraction summary to stdout before writing anything:
   - Total nodes extracted per `ConceptType`
   - Confidence distribution (mean, min, max per type)
   - A sample of extracted titles (up to 5 per type)
6. Prompts `Approve and write to KB? [y/N]`. On rejection, no KB or vector store writes are made; blob artifacts from steps 1 and 4 are retained.
7. On approval, all nodes are written as `AUTO_APPROVED` with `ingestion_source=INIT`.

**Rollback and recovery:** if `seshat init` crashes mid-write, re-running it is safe — each node write is a single Postgres transaction (KB row + vector embedding), so no partial state can be left behind. The init `job_id` is recorded in `init_runs`; re-running queries `init_runs` to detect the previous incomplete run and resumes from where it stopped. Resume is defined as **path (a) — skip, not upsert**: on detecting an incomplete `init_runs` entry, query `ops.kb_nodes WHERE job_id = X` to load already-written nodes, skip re-extraction for documents whose nodes are already present, and continue from the first document whose nodes are absent. This is consistent with the Node Lifecycle Invariant (append-only) and does not require upsert semantics. A full undo of a completed init is `DELETE FROM ops.kb_nodes WHERE job_id = X` (cascades to `ops.kb_relationships`) followed by deleting the corresponding pgvector embeddings and clearing the `init_runs` row.

**Re-running on a completed KB:** if `seshat init` is run again against the same `--source` path and the previous `init_runs` entry has `status=done`, the command treats it as a fresh run — it generates a new `job_id`, runs extraction from scratch, and writes any new nodes it produces. Previously written nodes (from the earlier init run) are not touched — the Node Lifecycle Invariant (append-only) applies. This means duplicate nodes may be written if the source corpus has not changed. Operators should use `--dry-run` to inspect what would be written before running a second init on an already-seeded KB.

**Document loader factory:**

```python
class DocumentLoaderProvider(StrEnum):
    MARKDOWN = auto()
    # NOTION = auto()     # v2
    # CONFLUENCE = auto() # v2

class AbstractDocumentLoader(ABC):
    async def load(self, source: str) -> list[Document]: ...
    # Returns a list of LangChain Documents (text + metadata) to feed into the extraction pipeline.
    # metadata carries at minimum {"source": filepath} for MarkdownDocumentLoader; v2 loaders
    # (Notion, Confluence) may include page title, URL, last-modified, etc.
    # Async for consistency with the rest of the pipeline and v2 network-backed loaders;
    # MarkdownDocumentLoader wraps its file I/O in asyncio.to_thread().

class DocumentLoaderConfig(BaseModel):
    provider: DocumentLoaderProvider = DocumentLoaderProvider.MARKDOWN
    source_path: str = "./init-docs"
```

`DocumentLoaderConfig` is added to `SeshatConfig` as an optional field — only required when running `seshat init`.

---

## 3. Configuration

Single `SeshatConfig` singleton loaded at startup via pydantic-settings. Per-request overrides are applied via a recursive deep-merge onto the base singleton (see `get_request_settings` below) — the singleton is never mutated.

> **Config pattern:** only the root `SeshatConfig` inherits from `BaseSettings` — it owns env var resolution. All nested configs (`ExtractionConfig`, `RAGConfig`, etc.) are plain `BaseModel`. Nested fields are still fully configurable from the environment via `env_nested_delimiter="__"` (e.g. `EXTRACTION__CONFIDENCE_THRESHOLD=0.8`) — pydantic-settings resolves them through the root, not independently. This prevents dual resolution paths where a nested `BaseSettings` could silently read env vars on its own.

```python
class SeshatConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_nested_delimiter="__"   # e.g. EXTRACTION__LLM__PROVIDER=anthropic
    )
    transcription: TranscriptionConfig
    vector_store: VectorStoreConfig
    vector_index: VectorIndexConfig
    kb_store: KBStoreConfig
    blob_store: BlobStoreConfig
    extraction: ExtractionConfig
    rag: RAGConfig
    secrets: SecretsConfig
    observability: ObservabilityConfig
    document_loader: DocumentLoaderConfig | None = None  # only required for seshat init
    max_jobs_per_user_per_hour: int = 10                  # per-user job submission rate limit
    max_concurrent_jobs: int = 1                          # global cap on jobs in TRANSCRIBING, EXTRACTING, or WRITING simultaneously; protects against LLM cost blowup
    max_concurrent_init_runs: int = 1                     # cap on simultaneous seshat init runs; init is typically the most expensive pipeline operation
```

All provider fields use `StrEnum` with `auto()` — values are lowercased member names, validated at startup:

```python
class LLMProvider(StrEnum):
    OPENAI = auto()
    ANTHROPIC = auto()

class TranscriptionProvider(StrEnum):
    ASSEMBLYAI = auto()
    OPENAI = auto()
    DEEPGRAM = auto()

class VectorStoreProvider(StrEnum):
    PGVECTOR = auto()
    # CHROMA = auto()    # v2
    # QDRANT = auto()    # v2
    # WEAVIATE = auto()  # v2

class EmbeddingProvider(StrEnum):
    OPENAI = auto()
    ANTHROPIC = auto()
    COHERE = auto()
    FASTEMBED = auto()  # local ONNX-based inference, no API cost; evaluate during RAG implementation

class SecretsProvider(StrEnum):
    ENV = auto()
    AWS = auto()
    # AZURE = auto()   # v2
    # VAULT = auto()   # v2
```

### Per-Request Overrides

`get_request_settings` deep-merges an optional `SeshatConfigOverride` onto the base singleton and returns a new `SeshatConfig` — the singleton is never mutated.

**Contract:**
- When `overrides is None`, return the base singleton unchanged.
- Only fields explicitly set in `overrides` replace base values. Unset fields retain their base value at every depth — a caller sending `{"extraction": {"auto_mode": True}}` must not revert `extraction.confidence_threshold` to its default, and `{"extraction": {"verification": {"model": "…"}}}` must not wipe `extraction.verification.provider`.
- The merge recurses through nested config objects to any depth; leaf values (scalars, enums, lists) are replaced wholesale.

```python
class SeshatConfigOverride(BaseModel):
    transcription: TranscriptionConfig | None = None
    extraction: ExtractionConfig | None = None
    rag: RAGConfig | None = None

def get_request_settings(overrides: SeshatConfigOverride | None) -> SeshatConfig: ...
```

### ExtractionConfig

```python
class LLMConfig(BaseModel):
    provider: LLMProvider = LLMProvider.ANTHROPIC
    model: str = "claude-sonnet-4-6"
    temperature: float = 0.0

class VerificationConfig(BaseModel):
    provider: LLMProvider      # must differ from ExtractionConfig.llm.provider — enforced by model_validator at startup
    model: str                 # e.g. "gpt-4o-mini" or "claude-haiku-4-5-20251001"

class ExtractionConfig(BaseModel):
    llm: LLMConfig = LLMConfig()
    concept_types: list[ConceptType] = list(ConceptType)
    confidence_threshold: float = 0.7
    per_type_thresholds: dict[ConceptType, float] | None = None  # overrides confidence_threshold per type; None = use global default for all types
    auto_mode: bool = False
    max_chunk_count: int = 50                       # hard ceiling; prevents O(agents × chunks) cost blowup
    max_output_tokens: int = 2048                   # output (generation) tokens per agent call
    max_total_input_tokens: int = 2_000_000         # aggregate input token cap across all agent calls in the extraction stage
    max_total_output_tokens: int = 400_000          # aggregate output token cap across all agent calls in the extraction stage
    max_transcript_chunk_tokens: int = 8000         # per-chunk input ceiling; see prompt budget note below
    max_hint_nodes: int = 20                        # most recent same-type KB nodes included in extraction-time hint
    max_hint_tokens: int = 1000                     # hard token cap on the hint; oldest nodes dropped first if exceeded
    merge_similarity_threshold: float = 0.85        # cosine similarity floor for within-meeting deduplication fallback; see Chunking
    max_retries: int = 3                            # per-call retry attempts on transient errors (API timeout, HTTP 429)
    verification: VerificationConfig | None = None  # None = heuristics-only scoring; see Confidence Scoring
    confidence_weights: ConfidenceWeights = ConfidenceWeights()
    result_cache_enabled: bool = False              # in-memory extraction result cache keyed on hash(chunk_text + concept_type + model + prompt_hash); auto-set True by seshat eval regardless of config; False for production to avoid stale results across jobs

    # model_validator enforces: verification.provider != llm.provider (startup error if violated)
    # and issues a UserWarning when llm.provider=ANTHROPIC and verification=None (heuristics-only)

class ConfidenceWeights(BaseModel):
    logprobs: float = 0.5       # weight when provider supports logprobs (OpenAI); ignored otherwise
    verification: float = 0.35  # weight when verification agent is configured; ignored otherwise
    heuristics: float = 0.15    # always active

    # Unavailable signals are excluded from both numerator and denominator — weights redistribute
    # proportionally rather than collapsing to zero. Examples:
    #   logprobs + verification + heuristics → (0.5*lp + 0.35*vf + 0.15*h) / 1.0
    #   verification + heuristics only       → (0.35*vf + 0.15*h) / 0.5
    #   heuristics only                      → h / 1.0

# These are the baseline values. Adjust only after running seshat eval on the labelled corpus
# and confirming the calibration — the formula and weights are fixed until eval data justifies a change.
```

> **Prompt token budget:** each agent call assembles a prompt from four components. The combined total must not exceed the model's context window:
>
> ```
> system_prompt         ~500t   (static per ConceptType; cached after first call)
> kb_hint               ≤1000t  (ExtractionConfig.max_hint_tokens; recency-scoped, oldest dropped first)
> retrieved_context     ≤4000t  (RAGConfig.max_context_tokens)
> transcript_chunk      ≤8000t  (ExtractionConfig.max_transcript_chunk_tokens)
> output_schema         ~200t   (structured output schema injected into prompt)
> ─────────────────────────────
> total input           ≤13700t  well within claude-sonnet-4-6's 200k context window
> ```
>
> The practical ceiling is cost, not the model context limit. `max_transcript_chunk_tokens=8000` keeps per-call input tokens manageable; combined with `max_chunk_count=50` and `max_output_tokens=2048` this bounds per-call token usage. The aggregate caps `max_total_input_tokens` and `max_total_output_tokens` enforce a ceiling across all agent calls in the extraction stage — at the defaults (2M input / 400k output), worst case (50 chunks × 4 agents × 13,700 input + 2,048 output ≈ 2.74M input / 410k output) will trip the caps on a max-length transcript. This is intentional: the defaults are a conservative ceiling, not an expected operating point. Calibrate both caps after running `seshat eval` on representative transcripts.

### RAGConfig

```python
class RAGConfig(BaseModel):
    enabled: bool = True
    top_k: int = 5        # candidates retained from vector search (fed to graph traversal)
    max_context_tokens: int = 4000
    traversal_max_depth: int = 1                            # direct neighbours only for MVP
    traversal_rel_types: list[RelationshipType] | None = None  # None = all relationship types
```

Metadata filters for retrieval are **not** config — they are passed per-job in the request payload.

> **Traversal risk:** unbounded `traversal_max_depth` or large graphs can inflate retrieved context beyond `max_context_tokens`. The assembler truncates at `max_context_tokens` — but a high depth combined with a dense graph will silently drop nodes from the end of the context window. Keep `traversal_max_depth=1` for MVP; increase only after measuring context token usage on real data.

> **Truncation ordering:** before serialising retrieved nodes into the context window, the assembler estimates each node's token cost using `len(title + description + source_quote) / 4` and greedily includes nodes in order (see below) until `max_context_tokens` is reached. Nodes that would exceed the budget are pre-empted — not serialised at all. The count of pre-empted nodes is logged to MLflow before serialisation begins (alongside the count of any nodes dropped after-the-fact for other reasons). Ordering: `meeting_date DESC NULLS LAST` (most recent first; init-sourced nodes with `meeting_date=None` sort last). Within the same `meeting_date`, nodes that appear in `resolution_candidates` for the current job rank above unrelated nodes — these are the nodes the resolution agent is most likely to need for accurate resolution.

### VectorStoreConfig

```python
class VectorStoreConfig(BaseModel):
    provider: VectorStoreProvider = VectorStoreProvider.PGVECTOR
    connection_secret_key: str = "postgres_url"   # key looked up in SecretsProvider at startup
    # Other providers (Chroma, Qdrant, Weaviate): add provider-specific config fields when implementing
```

### VectorIndexConfig

Embedding and collection settings are split into a separate config so they can be tuned independently of the provider selection.

```python
class VectorIndexConfig(BaseModel):
    collection: str = "seshat-docs"
    embedding_provider: EmbeddingProvider = EmbeddingProvider.OPENAI
    embedding_model: str = "text-embedding-3-small"
    max_indexing_tokens: int = 500_000   # aggregate token cap across all embedding calls in the RAG stage
```

### KBStoreConfig

```python
class KBStoreConfig(BaseModel):
    schema_name: str = "ops"             # PostgreSQL schema that owns kb_nodes and kb_relationships
    pool_min_size: int = 2
    pool_max_size: int = 10
    connection_secret_key: str = "postgres_url"   # shared with VectorStoreConfig; same Postgres instance
```

### BlobStoreConfig

```python
class BlobStoreConfig(BaseModel):
    bucket: str = "seshat-mvp"
    region: str = "eu-west-1"
    endpoint_url: str | None = None   # set to LocalStack URL in dev; None = real AWS
```

### TranscriptionConfig

```python
class TranscriptionConfig(BaseModel):
    provider: TranscriptionProvider = TranscriptionProvider.ASSEMBLYAI
    model: str | None = None            # None = provider default (e.g. AssemblyAI "best" tier)
    language: str = "en"                # matches TranscriptMetadata.language default
    max_file_bytes: int = 500 * 1024 * 1024  # 500 MB; enforced by streaming size check before upload completes (HTTP 413)
    max_audio_seconds: int = 7200       # job transitions to FAILED if duration.total_seconds() exceeds this cap (default: 2h)
    max_retries: int = 3                # per-call retry attempts on transient errors (API timeout, HTTP 429)
```

### ObservabilityConfig

```python
class ObservabilityConfig(BaseModel):
    mlflow_tracking_uri: str = "http://mlflow:5000"
    mlflow_experiment_name: str = "seshat"
    # experiment_id is resolved from experiment_name at startup via the MLflow client —
    # cached in-process and used to build deep links (see Section 9).
```

---

## 4. Multi-Agent Extraction

```
TranscriptDocument
              │
              ▼
        Orchestrator
  (chunk → dispatch → merge → score → set status)
              │
    ┌─────────┼──────────┬─────────────┬──────────────┐
    ▼         ▼          ▼             ▼              ▼
ADR Agent  Risk Agent  Agreement   Action Item   [custom via
(+ADR      (+Risk      Agent        Agent         registry]
 hints)     hints)    (+Agreement  (+ActionItem
                        hints)       hints)
    │         │          │             │              │
    └─────────┴──────────┴─────────────┴──────────────┘
                              │
                              ▼
                    ExtractionResult (new nodes)
                              │
                              ▼
                    RAG + Resolution (Orchestrator)
                              │
                              ▼
                    ExtractionResult (nodes + relationships as top-level list)
```

> **Note:** the Action Item agent additionally returns `assignee: str | None` — see Two-Pass Extraction Contract below.

### Two-Pass Extraction Contract

**Ordering invariant:** all agent calls in the fan-out phase must complete and their outputs merged before the RAG + Resolution pass begins. All `KBRelationship` objects — without exception — are created in Pass 2.

- **Pass 1 — Fan-out:** agents run concurrently, one per `ConceptType` per chunk. Agents return `KBNode` objects — no relationships. `KBNode` has no `relationships` field; the model itself enforces this. The Action Item agent is the sole exception in output schema: it includes an additional field `assignee: str | None = None` (the participant name it identifies as owner). This field is not a `KBRelationship` — it is a named extraction output that Pass 2 resolves.
- **Pass 2 — RAG + Resolution:** runs only after the complete merged Pass 1 node list is in memory. Constructs all relationships, including `ASSIGNED_TO` (by matching `assignee` against `TranscriptMetadata.participants` — exact match first, then case-insensitive prefix). The `assignee` field is consumed here and does not appear on the final `KBNode` or in the KB store.

  **`participants=None` fallback:** when `TranscriptMetadata.participants` is `None`, `ASSIGNED_TO` resolution is skipped for all action items — no `ASSIGNED_TO` relationships are created. The `assignee` value is discarded. The action item node is written without an assignee relationship; it is not rejected or downgraded. This is logged as a warning per affected node so the reviewer is aware the assignee could not be resolved.

Cross-chunk assignment (e.g. "as we agreed earlier, you handle this") is handled correctly under this contract: the agent records the assignee name it can see; resolution runs once against the full participant list after all chunks are processed.

### Agent Registry

Each `ConceptType` maps to a registered agent class with its own system prompt. Adding a new concept type = register a new agent + add the type to `ExtractionConfig.concept_types`. The orchestrator discovers agents from the registry at runtime.

### Prompt Caching

Agent system prompts are static per `ConceptType` and reused across every job. Prompt caching is a **first-class design requirement**, not an optimisation. Full caching strategy (Anthropic `cache_control` headers, OpenAI automatic prefix caching, LLM wrapper ownership, cold-start behavior, and MLflow observability) is defined in [docs/superpowers/specs/2026-04-27-prompt-interaction-design.md §2.3](2026-04-27-prompt-interaction-design.md).

### Core Enums

```python
class ConceptType(StrEnum):
    ADR = auto()
    RISK = auto()
    AGREEMENT = auto()
    ACTION_ITEM = auto()

class RelationshipType(StrEnum):
    MITIGATES = auto()       # Risk → ADR
    SUPPORTS = auto()        # Agreement → ADR
    CONFLICTS_WITH = auto()  # X → X (same ConceptType only; any type)
    DEPENDS_ON = auto()      # ADR → ADR
    SUPERSEDES = auto()      # ADR/Agreement → ADR/Agreement (fully replaces)
    AMENDS = auto()          # ADR/Agreement → ADR/Agreement (partial update or clarification)
    ASSIGNED_TO = auto()     # Action Item → participant
```

### KBNode (graph-shaped from day one)

```python
class KBNode(BaseModel):
    id: UUID = Field(default_factory=uuid4)  # auto-generated at construction time
    schema_version: str = "1.0"
    type: ConceptType
    title: str
    description: str
    confidence: float
    source_quote: str        # exact transcript excerpt (grounding)
    status: NodeStatus
    state: NodeState = NodeState.CURRENT
    metadata: NodeMetadata

class KBRelationship(BaseModel):
    source_id: str
    target_id: str
    rel_type: RelationshipType
    job_id: str           # which job created this relationship; duplicates source_id → KBNode.metadata.job_id
    created_at: datetime  # genuinely non-derivable; not available via the source node

    # Implementation note: `job_id` is intentionally duplicated here for query convenience —
    # the alternative is a join through source_id → ops.kb_nodes.metadata->>'job_id'.
    # If duplication becomes a maintenance concern, drop job_id and use the join instead.

class NodeStatus(StrEnum):
    AUTO_APPROVED = auto()
    PENDING_REVIEW = auto()
    REJECTED = auto()

class NodeState(StrEnum):
    CURRENT = auto()      # no changes since creation
    AMENDED = auto()      # a later node amends it; still relevant
    SUPERSEDED = auto()   # replaced by a later node; no longer active

class ApprovalMethod(StrEnum):
    INDIVIDUAL = auto()   # per-node decision in ApproveRequest.decisions
    BULK = auto()         # matched an ApproveRequest.approve_above_threshold rule
    AUTO = auto()         # auto_mode=True; no human review
    THRESHOLD = auto()    # confidence >= threshold at extraction time; no human review, no auto_mode flag

class IngestionSource(StrEnum):
    JOB = auto()    # extracted from a meeting recording via the normal pipeline
    INIT = auto()   # seeded from a document corpus via seshat init

class NodeMetadata(BaseModel):
    job_id: str                               # UUID4; same namespace for both JOB and INIT ingestion (refs ops.jobs.job_id or ops.init_runs.job_id)
    meeting_date: date | None = None          # None when ingestion_source=INIT
    participants: list[str] | None = None     # best-effort; None when unknown or ingestion_source=INIT
    ingestion_source: IngestionSource = IngestionSource.JOB
    team: str | None = None
    project: str | None = None
    domain: str | None = None
    approved_by: str | None = None
    approved_at: datetime | None = None
    approval_method: ApprovalMethod | None = None
    corrected_by: str | None = None    # set when a reviewer provides edited_content in NodeDecision
    corrected_at: datetime | None = None   # set to the same timestamp as approved_at (corrections only happen at approval time in v1)
```

```python
class ResolutionCandidate(BaseModel):
    node_id: str                      # existing KB node flagged by the resolution step
    rel_type: RelationshipType        # SUPERSEDES, AMENDS, or CONFLICTS
    candidate_title: str              # title of the existing KB node (read-only; for display)
    target_node_confidence: float     # echoed from KBNode.confidence of the target node

class ExtractionResult(BaseModel):
    job_id: str
    nodes: list[KBNode]
    relationships: list[KBRelationship]           # all relationships produced by Pass 2; written to KB separately from nodes
    confidence_breakdowns: dict[str, UUID]         # node.id → breakdown
    resolution_candidates: dict[str, list[ResolutionCandidate]]  # node.id → candidates flagged for that node
```

`ExtractionResult` is the output of the extraction + resolution pass and the payload returned by `GET /jobs/{id}/results`. Relationships are a top-level list rather than embedded on nodes — `KBNode` has no `relationships` field. The writing stage iterates `ExtractionResult.relationships` and calls `write_relationship()` for each after all nodes are written. `resolution_candidates` surfaces the SUPERSEDES/AMENDS/CONFLICTS_WITH candidates identified by the resolution step for each new node — the reviewer UI uses these to show the reviewer what existing KB nodes may be affected by an approval decision.

> **Vector store indexing:** one vector per `KBNode` — the embedding is generated from `title` + `description` + `source_quote` after extraction. `NodeMetadata` travels with the vector for runtime metadata filtering (applied as `>=` comparisons for `min_confidence`, equality for all other fields). The `confidence`, `node_type`, and `node_state` fields are duplicated from `KBNode` intentionally — the vector store needs them for filtering without a round-trip to the KB store. **Invariant:** `NodeMetadata.confidence`, `NodeMetadata.node_type`, and `NodeMetadata.node_state` must always equal the corresponding fields on `KBNode` — they are written together in the same transaction and neither is ever updated independently.
>
> **Indexing timing:** vectors are written to the vector store during the `WRITING` stage, as part of the same Postgres transaction as the KB row (§4, Node Lifecycle Invariant). They are **not** written after extraction. Nodes from a job in `AWAITING_REVIEW` are not yet retrievable via RAG — invisible to concurrent jobs until the reviewing job reaches `DONE`. This is intentional: unreviewed nodes should not influence future extractions.

### Confidence Scoring

Confidence is derived from three signals, weighted and normalised:

```
final = sum(w_i * s_i  for each available signal i)
      / sum(w_i         for each available signal i)
```

Weights come from `ExtractionConfig.confidence_weights`. Unavailable signals (logprobs when using Anthropic; verification when `verification=None`) are excluded from both numerator and denominator — their weight redistributes proportionally. `final` always lies in [0, 1] regardless of which signals are active.

1. **Logprobs** — used when provider supports it (OpenAI). Derives confidence from token probability of extracted content.
2. **Verification agent** — a separate lightweight agent (cheap model: `gpt-4o-mini`, `claude-haiku`) that receives the extraction and source quote and answers a binary "is this well-supported?" question. Must use a different `LLMProvider` than the extraction agent — same-provider verification produces correlated errors (enforced by `model_validator`). Example pairing: extraction on `anthropic`, verification on `openai`.
3. **Heuristics** — always active. Formula:

```
heuristics_score = (
    0.4 * clamp(len(source_quote) / 200, 0.0, 1.0)    # quote length: saturates at 200 chars
  + 0.4 * title_specificity(title)                      # 1.0 if specific, 0.5 if generic, 0.0 if empty
  + 0.2 * directness(description)                       # 1.0 if direct, 0.5 if passive/vague, 0.0 if absent
)
```

`title_specificity` and `directness` are rule-based classifiers — no model calls. Both are implemented using spaCy's dependency parser and NER (no LLM required). The formula, sub-scores, and weights are fixed — implementation must match them exactly, as `seshat eval` calibrates against this contract.

**`title_specificity` scoring:**
- **1.0 (specific)** — title identifies a named component or technology (spaCy NER labels `ORG`/`PRODUCT`, or CamelCase/hyphenated token pattern) **and** contains a qualifier phrase (preposition or subordinating conjunction: "for", "when", "instead of", "via", "using").
- **0.5 (generic)** — title has a named component without a qualifier, or a qualifier without a named component.
- **0.0 (empty)** — title is absent, whitespace-only, or solely punctuation.

**`directness` scoring:**
- **1.0 (direct)** — description has an active-voice main verb with a direct object (spaCy `dobj`/`obj` dependency arc on an active subject) and no hedging tokens ("should", "might", "could", "may", "possibly", "consider").
- **0.5 (passive/vague)** — description is present but uses passive voice (spaCy `auxpass`/`nsubjpass`), contains hedging tokens, or has a verb with no object.
- **0.0 (absent)** — description is empty or whitespace-only.

**Active signals per configuration:**

| `llm.provider` | `verification` | Active signals | Effective weights (default) |
|---|---|---|---|
| `openai` | configured (anthropic) | logprobs + verification + heuristics | 0.50 / 0.35 / 0.15 |
| `openai` | `None` | logprobs + heuristics | 0.77 / 0.23 |
| `anthropic` | configured (openai) | verification + heuristics | 0.70 / 0.30 |
| `anthropic` | `None` | **heuristics only** | 1.0 — weakest configuration; startup warning issued |

The `anthropic` + `verification=None` combination is the weakest: no logprobs, no verification agent. Valid but must be used with awareness — the `confidence_threshold` calibrated against heuristics-only scores is not comparable to one calibrated with verification enabled.

```python
class ConfidenceBreakdown(BaseModel):
    logprobs: float | None = None        # None when provider does not support logprobs (e.g. Anthropic)
    verification: float | None = None    # None when verification agent not configured
    heuristics: float                    # always present
    final: float                         # normalised weighted sum per formula above; echoes KBNode.confidence
```

### Prompt Injection Mitigation

Transcript text and retrieved KB context are untrusted inputs injected into agent prompts. Full security model (structural isolation, output validation, source quote verification, context sanitisation, second-order risk, and per-agent coverage table) is defined in [docs/superpowers/specs/2026-04-27-prompt-interaction-design.md §3](2026-04-27-prompt-interaction-design.md).

### Chunking

**MVP:** TextTiling (NLTK implementation). Detects topic-shift boundaries in the transcript and produces variable-length, topically coherent chunks. Requires no model calls. `max_chunk_count` in `ExtractionConfig` is a hard ceiling to prevent O(agents × chunks) cost blowup. TextTiling tuning parameters (`w`, `k`) are implementation details.

> **Hypothesis, not guarantee:** TextTiling was designed for expository prose. Its suitability for meeting transcripts — which contain interruptions, backtracking, and mid-sentence topic drift — is unvalidated. The chunking sanity check in §12 must be completed before extraction metrics are interpreted. If the sanity check reveals systematic boundary errors, replace TextTiling with **`RecursiveCharacterTextSplitter`** (LangChain, `langchain-text-splitters`) as an immediate fallback — 500-token windows, 100-token overlap, markdown-aware separator hierarchy (`\n\n`, `\n`, ` `, `""`). No model calls, predictable boundaries, no dependency on transcript structure. Diarization and semantic chunking remain v2 options once production audio is available.

Agents run per chunk. Results are deduplicated and merged before returning. The merge criterion is:

1. **Primary — title exact-match + type equality:** two nodes of the same `ConceptType` with identical normalised titles (lowercased, whitespace-collapsed) are unconditionally the same concept. No embedding call needed.
2. **Fallback — cosine similarity:** two nodes of the same `ConceptType` whose titles are not exact-match but whose embeddings (title + description) exceed `ExtractionConfig.merge_similarity_threshold` are considered the same concept. Uses the same embedding model as `RAGConfig.embedding_provider` — no additional model needed.

Two nodes that satisfy either criterion are merged: the final settled position is kept — "settled" means the node with the highest chunk index (later in the transcript is assumed to reflect the settled discussion outcome; chunk start-token position as tiebreaker if ambiguous). The earlier node is discarded. No `SUPERSEDES` relationship is created within a single job — `SUPERSEDES` is reserved for cross-meeting evolution only.

> **Asymmetric failure modes:** a threshold too high leaves duplicate nodes for the same decision, which the resolution pass may then flag as spurious CONFLICTS. A threshold too low silently collapses distinct decisions about different components into one node. The eval corpus must include both near-duplicate pairs (should merge) and near-distinct pairs (should not merge) to validate the threshold — see §12, Labelled Corpus.

> **Trade-off:** within-meeting deduplication prioritises a clean final KB over preserving debate history. The reversal is intentionally discarded — only the final settled position survives. The `source_quote` on the surviving node must reflect the final position, not the earlier reversed one. The count of within-meeting merges per job is logged to MLflow as a quality signal: a job with many merges may indicate a contentious or poorly-transcribed meeting worth manual review.

**v2 upgrade path (if `RecursiveCharacterTextSplitter` fallback is also insufficient):**
- **Diarization-based splitting:** AssemblyAI speaker diarization splits on speaker turn boundaries. Highest coherence for multi-speaker meetings; requires production audio samples to validate quality.
- **Semantic chunking:** embedding-based boundary detection (e.g. LangChain `SemanticChunker`). Best quality but adds a pre-extraction embedding pass — justify against the eval corpus before adopting.

### Status Assignment

- `confidence >= threshold` → `status=AUTO_APPROVED`, `approval_method=THRESHOLD`, `approved_by="system"`, `approved_at=<extraction timestamp>`
- `confidence < threshold` → `status=PENDING_REVIEW`; `approval_method`, `approved_by`, `approved_at` remain `None` until `POST /jobs/{id}/approve`
- `auto_mode=True` → all nodes `status=AUTO_APPROVED`, `approval_method=AUTO`, `approved_by=<submitting user_id>`, `approved_at=<job submission timestamp>`, regardless of confidence

### Node Lifecycle Invariant

The pipeline is **append-and-state-only** — extracted content is never modified after creation. A node's `title`, `description`, `source_quote`, and `confidence` are immutable once written. The only permitted mutation is `update_node_state()` on existing nodes when a `SUPERSEDES` or `AMENDS` relationship is established by the resolution step. If a later meeting revisits an existing decision, the resolution step creates a new node and expresses the relationship via `SUPERSEDES`, `AMENDS`, or `CONFLICTS_WITH` — it never overwrites the original. This preserves the full decision history in the graph.

Consequences:
- **State transitions:** when a new node carries a `SUPERSEDES` or `AMENDS` relationship, the pipeline calls `update_node_state()` on the target node — advancing its `state` to `SUPERSEDED` or `AMENDED` respectively. This is the only mutation the pipeline applies to an existing node. A `CONFLICTS_WITH` relationship does **not** trigger a state transition — both nodes remain `NodeState.CURRENT`. `CONFLICTS_WITH` is a graph-level annotation only; reviewers discover active conflicts via the graph query UI (Screen 4 highlights them) and via `resolution_candidates` surfaced at review time (Screen 3). This is intentional: a conflict between two `CURRENT` nodes is a signal for human judgment, not an automatic state change.

  **Stale CONFLICTS edges:** when one party in a `CONFLICTS_WITH` pair is later superseded (its `state` advances to `SUPERSEDED`), the edge is not deleted — the Node Lifecycle Invariant is append-only. Instead, `GET /graph/{node_id}`, `GET /graph/{node_id}/impact`, and Screen 4 filter out `CONFLICTS_WITH` relationships where either party has `state != CURRENT`. The stale edge remains in `ops.kb_relationships` for historical audit but is invisible on all normal query paths. A future "show historical graph" view can expose it explicitly.
- **Store sync and recovery:** the KB store (`PostgresKBStore`) and the vector store (pgvector) share the same Postgres instance. Each node write is a single database transaction: `write_node()` inserts the KB row and `upsert()` inserts the vector embedding atomically. A crash mid-transaction leaves no partial state — Postgres MVCC ensures either both writes are committed or neither is. `NodeState` transitions (`update_node_state()`) are also single-row updates within a transaction.
- **Concurrent pipeline runs:** `max_concurrent_jobs=1` (default) prevents two jobs from running the pipeline simultaneously, which eliminates concurrent `update_node_state()` races at MVP scale. If `max_concurrent_jobs` is raised, `update_node_state()` transitions remain safe — setting `state=SUPERSEDED` twice on the same node is idempotent — but resolution quality may degrade because both jobs read the KB before either writes to it.
- **Human corrections:** edits happen at review time via `ApproveRequest.decisions[].edited_content`. When `edited_content` is non-null, the node's `title` and `description` are updated with the human-supplied values, and `NodeMetadata.corrected_by` / `corrected_at` are set alongside `approved_by` / `approved_at`. `corrected_by` is the single field distinguishing human-corrected content from unmodified LLM output — queries on the KB can filter on it to assess how often auto-extraction required correction. A post-approval `PATCH /graph/{node_id}` is deferred to v2.

---

## 5. RAG + Resolution Layer

RAG runs **after** extraction, not before. Extraction agents receive a lightweight same-type KB hint at prompt time (see below); the full retrieval and relationship resolution pass happens once all new nodes are extracted.

```
ExtractionResult (new nodes + relationships from Pass 2)
      │
      ▼
 RAG Service — per new node:
 1. Embed node (title + description + source_quote)
 2. Vector search → top-K candidate KB nodes (same type)
 3. Graph traversal on top-K nodes (KB Store)
      │
      ▼
 Resolution — two parallel orchestrator calls:
 ┌─────────────────────────┐
 ▼                         ▼
Same-type resolution    Cross-type resolution
(SUPERSEDES/AMENDS/     (MITIGATES/SUPPORTS/
 CONFLICTS per type)     DEPENDS_ON)
 └─────────────────────────┘
              │
              ▼
     Heuristic validation (merge + drop invalid rels)
              │
              ▼
 ExtractionResult (new nodes + full relationship set)
```

### Extraction-time KB hint

Before extraction, each agent receives a flat list of **same-type** existing KB nodes — title, date, and a one-line summary (the node's `description` truncated to 80 characters). No embeddings, no retrieval. This gives agents enough anchor to avoid re-extracting already-known decisions without adding retrieval complexity to the extraction step. Full relationship resolution is deferred to the RAG + resolution pass.

The hint is scoped by recency: only the `max_hint_nodes` most recent same-type nodes are included (ordered by `meeting_date DESC`, `INIT`-sourced nodes last). This keeps the hint useful — agents need awareness of recent decisions, not the full KB history — and keeps token usage bounded as the KB grows.

A hard token cap (`max_hint_tokens`) is enforced after assembly: if the serialised hint exceeds the cap, the oldest nodes are dropped until it fits. Hint token count is logged to MLflow per agent call so growth is visible over time. When hint tokens consistently approach `max_hint_tokens`, that is the signal to invest in semantic filtering (v2).

### Retrieval Flow

- **Embedding target:** each new `KBNode` is embedded from `title + description + source_quote` — node-to-node comparison is homogeneous and avoids the semantic distance problem of comparing raw transcript chunks against distilled KB summaries. **Known limitation:** `text-embedding-3-small` is a general-purpose semantic model — it conflates semantic similarity with logical coupling. Two ADRs on the same topic but independent may score highly similar; a Risk and an ADR with a `MITIGATES` relationship may score dissimilar. This is the primary reason the retrieval baseline must be measured before real use. If recall@5 < 0.7, switching the embedding model (e.g. a domain-specific or fine-tuned encoder) is the first tuning lever, before increasing `top_k`.
- **Vector search:** semantic similarity via embedding model (pgvector for MVP), per new node against same-type KB nodes. Returns top-K candidates — no reranker in MVP (see Decisions Deferred).
- **Graph traversal:** structural retrieval from KB Store — direct neighbours of top-K candidates (both inbound and outbound edges, depth=1). For MVP (Postgres): SQL join on `ops.kb_relationships`. For Neo4j: Cypher query. Same interface.

### Resolution

Resolution is two parallel LLM calls to the orchestrator — both receive the same context (all new nodes + their KB candidates) and run concurrently:

- **Call 1 — same-type resolution:** for each concept type, classifies each new node against its KB candidates as `SUPERSEDES`, `AMENDS`, `CONFLICTS_WITH`, or no relationship. The resolution agent prompt must include the following operational criteria — without them the agent will guess and produce inconsistent history:

  - **`SUPERSEDES`**: the new node renders the prior decision actionable-irrelevant — the old decision would no longer be followed. The prior node's `NodeState` transitions to `SUPERSEDED`.
  - **`AMENDS`**: the new node narrows, extends, conditionally qualifies, or corrects a detail of the prior decision while leaving it broadly active. The prior node's `NodeState` transitions to `AMENDED`.
  - **`CONFLICTS_WITH`**: both decisions are currently active but mutually incompatible. Neither node's state changes — see Node Lifecycle Invariant (Section 4).
  - **No relationship**: the new node covers the same topic but is independently valid (e.g. a separate decision about a different component). No state change.
  - **Tiebreaker (AMENDS vs SUPERSEDES):** when the relationship is ambiguous between the two, prefer `AMENDS` — it is the less destructive classification. The eval corpus must include a labelled borderline example to validate the agent applies this tiebreaker correctly.

  The eval corpus must include at least 2 labelled examples per relationship type (SUPERSEDES, AMENDS, CONFLICTS, no-relationship) to validate that the agent applies these criteria correctly.
- **Call 2 — cross-type resolution:** across all new nodes, resolves `MITIGATES`, `SUPPORTS`, and `DEPENDS_ON`. The relationship schema constrains which pairings are evaluated — no N×N comparison across all types:

| Relationship | Source → Target |
|---|---|
| `MITIGATES` | Risk → ADR |
| `SUPPORTS` | Agreement → ADR |
| `DEPENDS_ON` | ADR → ADR |

Once both calls return, a **heuristic validation step** merges the outputs and rejects malformed relationships before the result is finalised:

- A node cannot both `SUPERSEDES` and `CONFLICTS_WITH` with the same target
- `SUPERSEDES` and `AMENDS` are mutually exclusive on the same (source, target) pair
- Relationship direction must match the schema (e.g. a Risk cannot `DEPENDS_ON` an ADR)
- A new node cannot relate to a target node of a different `ConceptType` unless the schema permits it

Validation failures are logged and the offending relationship is dropped — they do not fail the job.

**Effect on `resolution_candidates`:** heuristic validation operates on the `KBRelationship` list only — it does not prune `resolution_candidates`. A candidate that was flagged by the resolution step but whose resulting relationship was dropped by validation will still appear in `resolution_candidates` in the `ExtractionResult`. This is intentional: `resolution_candidates` is a reviewer signal ("the resolution agent thought this existing node might be affected"), not a guarantee that a relationship was written. The reviewer sees the candidate, assesses it, and the absence of a written relationship is the correct outcome if validation dropped it.

### Retrieval Quality Baseline

The `top_k=5` default must be justified against a measured baseline before MVP ships — not tuned by intuition.

**Baseline approach:**
1. Seed a test KB from the eval corpus (`tests/eval/corpus/`) using `seshat init`.
2. For each labelled transcript in the eval corpus, run extraction to produce new nodes, then embed them and search against the seeded KB.
3. For each new node with a known ground-truth match in the KB, measure **recall@5**: fraction of known matches appearing in the top-5 retrieved candidates.
4. If recall@5 is below 0.7 with default settings, tune `top_k` upward or adjust the embedding model before locking defaults.
5. Also measure **precision@5** alongside recall@5. High recall with low precision@5 (suggested floor: ≥ 0.6) means resolution agents receive many irrelevant candidates — noisy resolution, not just incomplete retrieval. Low precision@5 with acceptable recall@5 is the signal to invest in reranking (see Decisions Deferred).

`seshat eval` runs this retrieval baseline alongside extraction evaluation — both are part of the same MLflow eval run, linked to the same experiment.

---

## 6. Storage Layer

### Storage Classes

`PostgresKBStore` and `S3BlobStore` are concrete classes — no abstract base. `AbstractVectorStore` is retained because LangChain already manages provider abstraction there and the interface is thin.

```
PostgresKBStore              AbstractVectorStore          S3BlobStore
                                   │                      (LocalStack in dev,
                             PGVectorStore                 AWS S3 in prod)
                             # ChromaVectorStore — v2
                             # QdrantVectorStore — v2
                             # WeaviateVectorStore — v2
```

**Abstraction layer decisions — summary:**

| Component | Pattern | Rationale |
|---|---|---|
| `PostgresKBStore` | Concrete class | Single MVP implementation; v2 adds `Neo4jKBStore` and a shared `KBStore` protocol *at that point* — no speculative interface today |
| `S3BlobStore` | Concrete class | Single MVP implementation (LocalStack / real S3); no v2 provider swap planned within the same process |
| `AbstractVectorStore` | Abstract + implementations | LangChain already owns provider abstraction here; interface is thin and multiple v2 providers (Chroma, Qdrant, Weaviate) are realistic |
| `AbstractTranscriptionService` | Abstract + implementations | Three providers are already enumerated (AssemblyAI, OpenAI, Deepgram); factory-swappable at startup |
| `AbstractDocumentLoader` | Abstract + implementations | v2 loaders (Notion, Confluence) are network-backed and behaviourally different from `MarkdownDocumentLoader` |
| `AbstractSecretsResolver` | Abstract + implementations | Two providers in MVP (ENV, AWS); v2 adds Azure and Vault — startup factory swap |
| `AsyncioTaskQueue` | Concrete class, duck-typed swap | One queue for MVP; the v2 `ARQTaskQueue` exposes the same three methods (`enqueue / get_status / cancel`) — no formal protocol needed, the swap is a one-line change at the worker entrypoint |

### Shared Filter and Result Types

```python
class NodeFilter(BaseModel):
    # All fields optional. Filters AND together; None = no constraint on that field.
    node_type: ConceptType | None = None
    team: str | None = None
    project: str | None = None
    domain: str | None = None
    ingestion_source: IngestionSource | None = None
    min_confidence: float | None = None   # applied as confidence >= min_confidence
    state: NodeState | None = None
    meeting_date_from: date | None = None   # inclusive lower bound on NodeMetadata.meeting_date
    meeting_date_to: date | None = None     # inclusive upper bound on NodeMetadata.meeting_date
    # AbstractVectorStore.search() ignores state, meeting_date_from, meeting_date_to —
    # these fields are only applied by PostgresKBStore.query() (WHERE clauses on ops.kb_nodes).
```

```python
class SearchResult(BaseModel):
    node_id: str
    score: float   # provider-native similarity score; higher = more similar
```

Both stores accept the same `NodeFilter` type for runtime filtering so filter semantics stay identical whether a request hits the KB store (for graph queries) or the vector store (for similarity search with metadata narrowing).

### Interfaces

All methods are async — the pipeline runs in an asyncio context. `PostgresKBStore` uses an async-native Postgres client (`asyncpg` via `langchain-postgres`). The async decision must be made before implementation — retrofitting sync-to-async is expensive.

**`PostgresKBStore`** exposes: write a node (plain `INSERT`, returns its UUID as `str`; relationships are always written separately); write a relationship (both source and target UUIDs required); transition a node's state (the only pipeline-legal mutation on an existing node); retrieve a node by ID; retrieve a node's neighbours filtered by relationship type(s) and direction (`inbound`, `outbound`, or `both`); query nodes by `NodeFilter`.

**`AbstractVectorStore`** exposes: upsert a node embedding (node ID + text + metadata); similarity search returning ranked `SearchResult` objects (query text, top-K count, optional `NodeFilter`); delete a node embedding by ID.

**`S3BlobStore`** exposes: put an artifact at a path key; get an artifact by key; check whether a key exists.

```python
class S3BlobStore:
    async def put(self, key: str, data: bytes) -> None: ...
    async def get(self, key: str) -> bytes: ...
    async def exists(self, key: str) -> bool: ...
```

### Write Order and Consistency

`PostgresKBStore` and `PGVectorStore` share the same Postgres instance. Each node write (KB row + vector embedding) is a single database transaction — no coordination protocol needed. If the transaction fails, neither store is written; the job transitions to `FAILED` with `recoverable=True` and the full pipeline can be retried.

`S3BlobStore` artifact writes (`curated/extraction.json`) happen at the start of WRITING, before KB transactions begin, and are non-fatal — if the blob write fails, the job continues and nodes are still written to Postgres. The raw transcript is already in blob storage for reprocessing regardless.

### MVP: PostgresKBStore

`PostgresKBStore` needs no config object — it resolves the connection string from secrets at startup (`seshat/postgres_url`), shared with the pgvector store.

KB nodes and relationships are stored in the `ops` schema alongside the operational tables. Two tables, managed by Alembic (same migration path as `ops.jobs`, `ops.api_keys`, and `ops.init_runs`):

**`ops.kb_nodes`** — one row per `KBNode`. Columns: `node_id` (PK), `schema_version`, `job_id`, `type` (ConceptType), `title`, `description`, `confidence`, `source_quote`, `status` (NodeStatus), `state` (NodeState, default `current`), `metadata` (JSONB), `created_at` (TIMESTAMPTZ).

**`ops.kb_relationships`** — one row per `KBRelationship`. Columns: `source_id` (FK → kb_nodes), `target_id` (FK → kb_nodes), `rel_type` (RelationshipType), `job_id` (UUID4, which job created this relationship), `created_at` (TIMESTAMPTZ). Composite PK on `(source_id, target_id, rel_type)`. Index on `target_id` for inbound traversal.

`get_neighbours()` joins on `ops.kb_relationships`. `direction="both"` returns inbound and outbound edges (used by `GET /graph/{node_id}` and RAG graph traversal); `direction="inbound"` filters to edges where `target_id = node_id` (used by impact traversal); `direction="outbound"` filters to edges where `source_id = node_id`. `query()` applies `NodeFilter` fields as SQL predicates on `ops.kb_nodes`.

**Schema migration:** Alembic manages all `ops` schema migrations — the same tool used for `ops.jobs`, `ops.api_keys`, and `ops.init_runs`. The `schema_version` field on `KBNode` is retained for application-level compatibility checks at read time.

**v2 path:** introduce `Neo4jKBStore` with the same method signatures as `PostgresKBStore` and extract a shared `KBStore` protocol at that point. Migration exports `ops.kb_nodes` and `ops.kb_relationships` rows into Neo4j nodes and edges — structured rows are easier to migrate than parsed YAML frontmatter files. The vector store remains pgvector when this migration happens — only the KB layer migrates.

### Future: Weaviate Adapter

When Weaviate is introduced, a single `WeaviateStore` class satisfies **both** the KB store and vector store interfaces — adapter pattern, one class, two roles. Both call sites receive the same instance. No changes to pipeline stages.

---

## 7. Secrets Layer

```python
class AbstractSecretsResolver(ABC):
    def get_secret(self, key: str) -> str: ...   # synchronous; cached in-process after first call

# Implementations: EnvSecretsResolver, AWSSecretsResolver
# (AzureSecretsResolver, VaultSecretsResolver — v2)
```

**Call frequency:** secrets are resolved **once at startup**, not per-agent invocation. The factory resolves all required secrets (LLM API keys, transcription API key) during worker initialisation and caches them in-process for the lifetime of the worker. The interface is synchronous — `AWSSecretsResolver` calls the boto3 `secretsmanager` client directly; the cache on `AbstractSecretsResolver` ensures the blocking HTTP call is made at most once per key per process lifetime.

```python
class SecretsConfig(BaseModel):
    provider: SecretsProvider = SecretsProvider.AWS
    # ENV — ignored when provider=AWS
    region: str = "eu-west-1"
    secret_path_prefix: str = "seshat"
    endpoint_url: str | None = None   # set to LocalStack URL in dev; None = real AWS
```

The secrets factory reads `SecretsConfig.provider` and returns the appropriate implementation. API keys stored as `SecretStr` in config are resolved through this layer at runtime — never hardcoded.

**Rotation:** secrets are resolved once at startup and cached in-process. If a secret is rotated (e.g. an LLM API key is compromised), the worker must be restarted to pick up the new value. For MVP this is acceptable — document it as a known operational procedure. For v2, implement a TTL-based cache in `AbstractSecretsResolver` so rotation takes effect within a configurable window without a full restart.

LocalStack emulates AWS Secrets Manager locally (`SERVICES=secretsmanager,s3`).

> **Future hardening:** once provider requirements are stable, replace the flat `SecretsConfig` with a Pydantic v2 discriminated union (`EnvSecretsConfig | AWSSecretsConfig` on the `provider` discriminator; Azure and Vault variants added when needed). Each provider gets its own model with only the fields that make sense for it — invalid combinations become impossible at startup. The same pattern can be applied to `LLMConfig` and `TranscriptionConfig` for the same reason.

---

## 8. API Layer & Job Lifecycle

### Authentication and Authorization

All API endpoints require authentication via an **API key** passed in the `X-API-Key` header.

**Key storage:** keys are stored as bcrypt hashes (cost factor 12) — plaintext keys are never persisted. FastAPI validates each request via a `Depends` function using constant-time bcrypt comparison against stored hashes, then extracts the role.

### Postgres Schema

All operational and vector state lives in a single Postgres database (`seshat`). Two schemas:

- **`ops`** — operational tables owned and migrated by Seshat
- **`store`** — pgvector tables created and managed by `langchain-postgres` (`langchain_pg_collection`, `langchain_pg_embedding`)

One role (`seshat`) with read/write on both schemas. Connection string stored in Secrets Manager under key `seshat/postgres_url` and resolved at startup via `SecretsProvider`.

Three operational tables in the `ops` schema, all managed by Alembic:

**`ops.api_keys`** — one row per issued API key. Fields: `key_hash` (PK, bcrypt), `user_id`, `role` (`submitter | reviewer | operator`), `created_at`, `last_used_at`.

**`ops.jobs`** — authoritative job state. Fields: `job_id` (PK, UUID4), `user_id` (FK → api_keys.user_id), `status` (mirrors `JobStatus`), `idempotency_key` (UNIQUE nullable), `source_type`, `created_at`, `updated_at`, `error_payload` (JSONB, null until FAILED), `mlflow_run_id` (null while PENDING). Index on `(user_id, created_at)` for the per-user rate-limit query.

**`ops.init_runs`** — coordination for `seshat init`. Fields: `job_id` (PK, UUID4), `status` (`running | done | failed`), `source_path`, `created_at`, `updated_at`.

`ops.jobs` is the authoritative source for API job lifecycle state. Idempotency key deduplication on `POST /jobs` is a single `SELECT` against the `UNIQUE` constraint on `idempotency_key`. `ops.init_runs` serves the same coordination role for `seshat init` runs. KB nodes and relationships live in `ops.kb_nodes` and `ops.kb_relationships` — see §6, MVP: PostgresKBStore. The `store` schema is entirely managed by `langchain-postgres` — Seshat code never writes DDL against it directly.

**Roles:**

| Role | Allowed actions |
|------|----------------|
| `submitter` | `POST /jobs`, `GET /jobs/{id}`, `GET /jobs/{id}/results`, `GET /graph`, `GET /graph/{node_id}`, `GET /graph/{node_id}/impact` |
| `reviewer` | All `submitter` actions + `POST /jobs/{id}/approve` |
| `operator` | All actions + `auto_mode=True` per-request override |

Key provisioning is a CLI command (`seshat create-api-key --user <id> --role <role>`) that prints the plaintext key once and stores the hash.

> **JWT (deferred to v2):** JWT with an external IdP (e.g. Azure AD) is the natural upgrade when the user base grows or SSO is needed. See Deferred Decisions.

### Endpoints

```
POST /jobs                   Submit a new job (audio file or text); see Job Submission below
GET  /jobs/{id}              Job status + per-stage progress; see Job Progress Contract below
GET  /jobs/{id}/results      ExtractionResult (nodes + relationships); available from AWAITING_REVIEW onwards (also DONE); returns HTTP 409 if job is not yet in a reviewable or completed state
POST /jobs/{id}/approve      Submit node review decisions
POST /jobs/{id}/retry        Retry a FAILED job (operator only)
GET  /graph                  Query KB nodes with filters
GET  /graph/{node_id}        Node + neighbours
GET  /graph/{node_id}/impact Traversal from node; see Impact Traversal below
GET  /health                 Liveness + readiness; returns {status: "ok"|"degraded"|"down", components: {postgres: str, mlflow: str, localstack: str, worker: str}}
```

`GET /graph` filter fields from `NodeFilter` are passed as query params: `?node_type=adr&team=platform&min_confidence=0.7&ingestion_source=job`. All fields are optional and AND-combined. Enum fields accept the lowercased `StrEnum` value.

### Job Submission

`POST /jobs` is a `multipart/form-data` request with two parts:
- **`file`** — the audio or text payload (required for all `source_type` values; the text variant carries the caller's pre-formatted YAML/JSON file)
- **`request`** — a `JobSubmissionRequest` JSON document:

```python
class JobSubmissionRequest(BaseModel):
    source_type: Literal["audio", "text"]   # "video" deferred to v2 (ffmpeg dependency)
    metadata: TranscriptMetadata                  # meeting_date, participants, etc.
    idempotency_key: str | None = None
    overrides: SeshatConfigOverride | None = None  # per-request config, deep-merged onto SeshatConfig
    retrieval_filters: NodeFilter | None = None    # runtime RAG retrieval scope; not config (see Section 3)
```

The API constructs a `TranscriptDocument` from this request (generating `id`; `raw_text` populated by the transcription stage for audio, or by the text validator for `source_type="text"`) and enqueues the job.

**Rate limiting:** `POST /jobs` enforces two checks before creating a job:

1. **Per-user hourly cap:** counts the user's jobs submitted in the last hour using `ops.jobs` (sliding window: current UTC time − 3600 seconds, indexed on `user_id, created_at`). If the count meets or exceeds `max_jobs_per_user_per_hour` (default: 10), the request is rejected with HTTP 429.
2. **Global concurrency cap:** counts jobs system-wide in `TRANSCRIBING`, `EXTRACTING`, or `WRITING` state. If the count meets or exceeds `max_concurrent_jobs` (default: 1), the request is rejected with HTTP 429 and a message indicating a job is already in progress. This prevents LLM cost blowup from simultaneous pipeline runs at MVP scale.

Both checks run before the job is created. Violations are logged with `user_id` and timestamp.

### Job Status Model

```python
class JobStatus(StrEnum):
    PENDING = auto()
    TRANSCRIBING = auto()
    EXTRACTING = auto()
    AWAITING_REVIEW = auto()   # pipeline pauses here for human review
    WRITING = auto()
    DONE = auto()
    FAILED = auto()
```

`AWAITING_REVIEW` is skipped entirely when `auto_mode=True`.

### Job Progress Contract

`GET /jobs/{id}` returns the following shape for all job states:

```python
class JobResponse(BaseModel):
    job_id: str
    status: JobStatus
    idempotency_key: str | None         # echoed from the original JobSubmissionRequest; None when not provided
    current_stage: JobStatus | None     # None when status is DONE or FAILED
    stage_progress: str | None          # human-readable; None when no meaningful message available
    elapsed_seconds: float
    error: ErrorPayload | None          # populated when status=FAILED
    mlflow_run_id: str | None = None    # set when the worker starts the first pipeline stage (TRANSCRIBING); NULL while PENDING
```

**`stage_progress` examples per stage:**
- `TRANSCRIBING` — `"Transcribing audio"` (AssemblyAI does not expose per-call progress in the polling API)
- `EXTRACTING` — `"Extracting: {n}/{total} agents complete"`. `n` counts completed individual agent calls (each chunk × concept_type pair is one call). `total` is `len(concept_types) × chunk_count`, computed once before dispatching the fan-out — e.g. 4 concept types × 12 chunks = 48 total; progress increments per completed call.
- `AWAITING_REVIEW` — `"{n} nodes pending review"`
- `WRITING` — `"Writing nodes to KB"`

> **v2 — Server-Sent Events:** a `GET /jobs/{id}/stream` SSE endpoint is the natural upgrade for lower-latency progress. Deferred until the pipeline stage model is stable and the Streamlit UI is mature enough to warrant it.

**`auto_mode` authorization and audit:** setting `auto_mode=True` requires the `operator` role — the `Depends` function rejects it for `submitter` and `reviewer`. Every job run with `auto_mode=True` is logged in MLflow with: requesting `user_id`, timestamp, job ID, and the full list of nodes that were auto-approved without human review. Auto-approved nodes have `approval_method=ApprovalMethod.AUTO`, `approved_by=user_id`, and `approved_at` set to the job submission timestamp.

### FAILED State and Recovery

`FAILED` carries a structured error payload:

```python
class CallType(StrEnum):
    LLM_INPUT = auto()       # units = tokens
    LLM_OUTPUT = auto()      # units = tokens
    EMBEDDING = auto()       # units = tokens
    TRANSCRIPTION = auto()   # units = audio_seconds

class UsageRecord(BaseModel):
    call_type: CallType
    units: float             # tokens for LLM/embedding; audio_seconds for transcription
    # Implementation note: `units` conflates two dimensions (tokens and seconds) in a single field.
    # Consider splitting into `tokens: int | None` and `audio_seconds: float | None` at implementation
    # time if the mixed type causes confusion in the MLflow logging or cost display code.

class ErrorPayload(BaseModel):
    stage: JobStatus                              # which stage failed
    reason: str                                   # human-readable description
    recoverable: bool                             # if True, POST /jobs/{id}/retry is available
    usage: dict[JobStatus, list[UsageRecord]]     # stage → [records] for this attempt
```

**Usage tracking:** usage is tracked per stage and per attempt as a list of `UsageRecord` — one record per `CallType` active in that stage. `call_type` determines what `units` means (tokens for LLM/embedding, audio seconds for transcription). Accumulated usage across all retries is reported in MLflow. Cost estimation from unit counts is informational only (shown in the MLflow UI) — enforcement uses token/duration caps, not dollar amounts, so no price table needs to be maintained.

**Per-stage usage caps:** each pipeline stage enforces aggregate usage limits in its config:
- `ExtractionConfig.max_total_input_tokens` / `max_total_output_tokens` — checked **after each individual agent call**. Before dispatching the next call, the worker sums cumulative tokens for the stage and aborts if either cap is exceeded. This prevents a job overshooting by more than one agent call's worth of tokens.
- `RAGConfig.max_embedding_tokens` — checked after each embedding call in the RAG stage. Up to 200 embedding calls per job (50 chunks × 4 concept types); the cap covers all embedding calls in the stage and is tracked via `CallType.EMBEDDING` in `UsageRecord`.
- `TranscriptionConfig.max_audio_seconds` — checked before invoking the transcription service. If `duration.total_seconds() > max_audio_seconds`, the job transitions to `FAILED` immediately with `recoverable=False` (audio duration is a property of the input — retrying will not help).

If a cap is exceeded, the job transitions to `FAILED` with `recoverable=True`. The `POST /jobs/{id}/retry` endpoint takes no body — to raise a cap, update the config and retry. The UI surfaces a retry button when `recoverable=True`.

**Automatic retry policy:** transient errors (API timeout, HTTP 429 rate limit) are retried automatically before the stage transitions to `FAILED`. Per-stage policy:
- **Max attempts:** 3 (configurable as `TranscriptionConfig.max_retries` and `ExtractionConfig.max_retries`)
- **Backoff:** exponential with jitter — base 2s, multiplier 2×, max 60s. When a `Retry-After` header is present, it sets the minimum delay floor; jitter is applied on top: `delay = max(computed_backoff, retry_after_seconds) * uniform(0.8, 1.2)`
- **Scope:** per-call retry within the stage. A stage that exhausts retries on any single call transitions to `FAILED` with `recoverable=True`

**Recoverable failures:**
- Transcription transient error (API timeout, rate limit) — retried automatically; if exhausted, raw input file in blob store; user re-submits via `POST /jobs` with original `idempotency_key`
- Extraction failure (LLM timeout, rate limit) — retried automatically per agent call; if exhausted, raw transcript in blob store; user re-submits via `POST /jobs` with original `idempotency_key`
- Usage cap exceeded — operator raises cap in config, then retries via `POST /jobs/{id}/retry` (preserves job ID) or user re-submits via `POST /jobs`
- Transaction failure during node write — Postgres atomicity ensures neither KB row nor vector embedding is written; retry re-runs the full job cleanly

**Fatal failures:**
- Malformed input file (corrupt audio, invalid video) — nothing written, must re-submit
- Text input schema validation failure — rejected at boundary before processing
- Authentication / authorization error — not a pipeline failure

**Worker boot recovery:** On startup, the worker queries the `jobs` table for all jobs in `WRITING` state before accepting new work. A job stranded in `WRITING` indicates a crash mid-write. Because KB and vector writes are a single Postgres transaction, a crash leaves no partial state — the transaction was either committed or rolled back. Recovery is simple: mark the stranded job `FAILED` with `recoverable=True` so the user can re-submit via `POST /jobs` with the original `idempotency_key`, or an operator can retry via `POST /jobs/{id}/retry`. The recovery step runs synchronously at startup before the worker accepts new jobs.

### Job Idempotency

`POST /jobs` accepts an optional `idempotency_key: str`. Deduplication is a `SELECT` against `jobs.idempotency_key` — the `UNIQUE` constraint on that column guarantees atomicity. `source_type` is not validated on deduplication — a key match on any source type is treated as the same job. Behaviour:

- If no existing job has that key → create and return a new job ID as normal.
- If a job with that key exists and is **not** `FAILED` → return the existing job ID immediately (no new job created).
- If a job with that key exists and **is** `FAILED` → create a new job as normal (full re-run from scratch). The raw blob store artifacts from the failed job are retained and share the same path prefix, so a v2 resume-from-checkpoint implementation can locate them via the existing `idempotency_key` without a schema change.

> **v2 — stage-level resume:** the pipeline stages map directly to durable artifacts already written to blob storage: `raw/transcript.txt` after `TRANSCRIBING`, `curated/extraction.json` after `EXTRACTING`. A stage-aware retry can detect which artifacts exist and skip the stages whose output is already present, rather than re-running the full pipeline from scratch. The implementation shape: on retry, the worker checks each stage's artifact key in blob storage before running that stage — if the artifact exists and passes a schema integrity check, the stage is skipped and its output is loaded directly. This requires no schema changes to `JobSubmissionRequest`, `TranscriptDocument`, or the blob path structure — those are already stable. Add this once the pipeline stage model is validated on real data and the cost of full re-runs is measurable.

`idempotency_key` is stored on `TranscriptDocument` and included in `GET /jobs/{id}` responses.

**Retry path division of labour:** there are two ways to retry a failed job, with distinct semantics:

| Path | Role | Job ID | When to use |
|---|---|---|---|
| `POST /jobs` with original `idempotency_key` | Any submitter | New job ID | Normal client retry — the Streamlit retry button uses this path |
| `POST /jobs/{id}/retry` | Operator only | Same job ID | Operator-initiated recovery where job ID continuity matters (e.g. audit trail, config change before re-run) |

The Streamlit UI retry button always uses `POST /jobs` with the original `idempotency_key` — it creates a new job ID and is available to all roles. `POST /jobs/{id}/retry` is an operator tool not surfaced in the UI. A `submitter` whose job fails retries via re-submission; they do not need operator access.

### Review Flow

`PENDING_REVIEW` nodes have no MVP SLA — a job stays in `AWAITING_REVIEW` until a reviewer acts on it. Timeout-based auto-rejection is deferred to v2 (see Decisions Deferred).

`POST /jobs/{id}/approve` accepts an `ApproveRequest`:

```python
class KBNodeEdit(BaseModel):
    title: str
    description: str

class NodeDecision(BaseModel):
    node_id: str
    action: Literal["approve", "reject"]
    edited_content: KBNodeEdit | None = None   # optional: user may edit before approving
    reason: str | None = None                  # optional: human explanation for rejection or override

class BulkApproveRule(BaseModel):
    threshold: float                           # approve all PENDING_REVIEW nodes with confidence >= threshold
    exclude: list[str] | None = None           # skip these node_ids even if confidence >= threshold

class ApproveRequest(BaseModel):
    approve_above_threshold: BulkApproveRule | None = None
    decisions: list[NodeDecision] | None = None
```

**Processing order:**
1. `approve_above_threshold` runs first — approves all `PENDING_REVIEW` nodes with `confidence >= threshold`, skipping any `exclude`d IDs. Sets `approval_method=ApprovalMethod.BULK` on affected nodes.
2. `decisions` runs second — per-node overrides. Can approve, reject, or edit any node regardless of whether the bulk rule already touched it. Sets `approval_method=ApprovalMethod.INDIVIDUAL` on affected nodes.

In both cases `approved_by` is set to the requesting user's `user_id` and `approved_at` to the current UTC timestamp.

Once all `pending_review` nodes have a decision, the pipeline resumes to `WRITING`. If all nodes were rejected (zero approved nodes), the pipeline still transitions to `WRITING` — the writing stage writes zero nodes and the job reaches `DONE` with an empty result. This is valid: a reviewer may legitimately reject all extracted nodes if the meeting produced no recordable decisions. The `DONE` response body will contain an empty `nodes` list; no special terminal state is introduced.

### Streamlit UI Scope

The Streamlit app is the primary interface for human operators. Four screens, defined at the functional level — pixel-level design is out of scope for this spec.

**Screen 1 — Job submission**
- File upload field (audio: `.mp3`, `.wav`, `.m4a`; text: `.yaml`, `.json`)
- Metadata form: `meeting_date` (required), `participants` (optional, comma-separated), `language` (default `en`)
- Optional: per-request overrides (`confidence_threshold`, `auto_mode` — operator role only)
- Submit button → polls `GET /jobs/{id}` and transitions to Screen 2

**Screen 2 — Job status**
- Pipeline stage progress bar: `PENDING → TRANSCRIBING → EXTRACTING → AWAITING_REVIEW → WRITING → DONE`
- `stage_progress` text (e.g. `"Extracting: 3/8 agents complete"`)
- Elapsed time
- "View in MLflow" deep-link button (visible once `mlflow_run_id` is set on `JobResponse`)
- On `FAILED`: error reason + retry button (visible when `recoverable=True`)
- On `AWAITING_REVIEW`: "Review nodes" button → transitions to Screen 3

**Screen 3 — Node review**

This is the primary correctness gate for the knowledge base. For each `PENDING_REVIEW` node the UI must surface:

- Node title, type (`ConceptType`), and description
- `source_quote` inline (the exact transcript excerpt grounding this node)
- `ConfidenceBreakdown`: final score + per-component breakdown (logprobs / verification / heuristics)
- relationships from `ExtractionResult.relationships` where `source_id == node.id` (read-only), with a hyperlink to each target node in Screen 4
- `resolution_candidates` from `ExtractionResult`: existing KB nodes flagged as SUPERSEDES / AMENDS / CONFLICTS candidates — shown prominently so the reviewer can assess downstream impact before approving. Each candidate entry shows `candidate_title`, `rel_type`, and `target_node_confidence`, so reviewers can distinguish a high-confidence `CONFLICTS_WITH` target from a speculative one
- Per-node action: approve / reject / edit (opens an inline form pre-filled with current `title` + `description`)
- Bulk approve rule: threshold slider + optional exclude list → maps to `BulkApproveRule` in `ApproveRequest`
- Submit decisions button → `POST /jobs/{id}/approve`

**Screen 4 — Knowledge base query**
- Filter panel: `ConceptType`, `team`, `project`, `domain`, `NodeState`, date range
- Node list: title, type, confidence, state (`CURRENT` / `AMENDED` / `SUPERSEDED`)
- Node detail panel: full node + its relationships from `ops.kb_relationships` (source node + type + target node for each); CONFLICTS relationships highlighted; stale CONFLICTS edges (where either party has `state != CURRENT`) are filtered out and not displayed
- "View in MLflow" deep-link from node metadata

### Impact Traversal

```
GET /graph/{node_id}/impact?depth=2&rel_types=supersedes,depends_on&min_confidence=0.0
```

Answers "what would break if this node changed?" — inbound BFS from `node_id`, following relationship edges backwards (source → seed). Returns `list[KBNode]` with each node annotated with `traversal_depth: int`. Query parameters:

- `depth` — BFS traversal depth (default 2, max 3; unbounded traversal on a dense KB is a cost risk)
- `rel_types` — comma-separated `RelationshipType` values to follow during traversal; default: all types
- `min_confidence` — filters traversed nodes by `KBNode.confidence >= min_confidence`; default 0.0

Role requirement: `submitter` (read-only, same as `GET /graph/{node_id}`).

`get_neighbours()` is called with `direction="inbound"` at each BFS level. `traversal_depth` on each returned node is the hop count from the seed node. Example: if ADR-A `DEPENDS_ON` ADR-B, traversing inbound from ADR-B returns ADR-A at depth=1 — the node that would break if ADR-B changes.

### Reviewer Data Contract

`GET /jobs/{id}/results` returns `ExtractionResult`. The Streamlit review screen is the only consumer of this endpoint for MVP — the data contract between the API and the UI is:

| Field | Source | Used by |
|---|---|---|
| `nodes` | `ExtractionResult.nodes` | All node fields in Screen 3 |
| `confidence_breakdowns[node_id]` | `ConfidenceBreakdown` per node | Confidence breakdown display |
| `resolution_candidates[node_id]` | Resolution step output | SUPERSEDES/AMENDS/CONFLICTS_WITH impact preview |
| relationships for `node.id` | `KBRelationship` rows from `ExtractionResult` | Relationship list + hyperlinks |
| `nodes[].source_quote` | `KBNode.source_quote` | Inline source quote |

---

## 9. Observability

**MLflow 3** as the observability backbone — built on OpenTelemetry, self-hosted, Docker-native.

LangChain is the LLM orchestration framework. `mlflow.langchain.autolog()` instruments all agent calls automatically — no manual trace wiring required. Agents implement the MLflow Responses Agent interface for native compatibility with `mlflow.genai.evaluate()`.

Captured per job/agent run:
- Agent identity, model, prompt version
- Usage per agent call (`UsageRecord` list per stage); token counts logged as-is — cost estimation at display time is informational only (no price table in the pipeline)
- Prompt cache hit/miss per agent call: for Anthropic, `cache_read_input_tokens` and `cache_creation_input_tokens` from the response; for OpenAI, `cached_tokens` from the usage object. Logged as MLflow metrics so cache effectiveness is measurable. The first job after worker startup will have zero cache hits (cold start) — this is expected and should not trigger alerts.
- Latency per pipeline stage
- Confidence scores and node status distribution
- Errors and retries

**Prompt and response content:** `mlflow.langchain.autolog()` captures full prompt inputs and model response outputs as MLflow artifacts automatically — no manual logging required. This data is essential for forensic investigation of prompt injection or hallucination incidents and must be treated as sensitive (may contain transcript excerpts and extracted decisions).

> **MVP caveat:** MLflow runs locally and its artifact store has no access controls. The API itself is multi-user (roles, per-user rate limiting, up to ~10 users) but MLflow is not — all users share the same local MLflow instance with no visibility separation. This is acceptable for a small trusted team on a local deployment. If the deployment is ever exposed beyond localhost or the user base grows beyond the initial trusted group, the MLflow artifact store must be access-controlled before use — prompt/response artifacts may contain sensitive transcript content.
>
> **v2 hardening:** separate the prompt/response artifact store from operational metrics with access controls (e.g. a restricted S3 prefix or a separate MLflow experiment with role-based visibility).

**Streamlit integration:**
- MVP: "View in MLflow" button deep-links to `{mlflow_tracking_uri}/#/experiments/{experiment_id}/runs/{run_id}`. `mlflow_run_id` comes from `JobResponse`; `experiment_id` is resolved once at startup from `ObservabilityConfig.mlflow_experiment_name` via the MLflow client and cached in-process — not exposed on the API.
- v2: MLflow Python client renders usage, estimated cost, latency, and confidence distributions as native Plotly charts in the Streamlit app

---

## 10. Docker Compose Topology

```yaml
services:
  api:          # FastAPI — uvicorn entrypoint
  worker:       # Pipeline worker — same image, different command (asyncio task queue); restart: unless-stopped
  streamlit:    # Streamlit UI
  postgres:     # Postgres 16 + pgvector extension — ops schema (api_keys, jobs, init_runs, kb_nodes, kb_relationships) + store schema (langchain-postgres vector tables); persistent volume
  mlflow:       # Observability (SQLite backend — MLflow-internal, separate from ops DB)
  localstack:   # AWS SM + Blob Storage (S3BlobStore) emulation (SERVICES=secretsmanager,s3); bucket: seshat-mvp
```

- `api` and `worker` share the same Docker image, different `command` entrypoint
- All inter-service communication via Docker service names
- Single `.env` file drives all config via `env_nested_delimiter="__"`
- `postgres` and `mlflow` mount named volumes for persistence
- `worker` sets `restart: unless-stopped` — automatically restarts on crash without operator intervention
- Postgres connection string stored in LocalStack Secrets Manager under key `seshat/postgres_url`; resolved at startup via `AWSSecretsResolver`

> **v2 — multiple worker replicas:** horizontal scaling requires a durable queue (ARQ/Redis). With the MVP asyncio queue, multiple replicas would race on the same in-memory task list. Upgrade path: swap `AsyncioTaskQueue` for `ARQTaskQueue` and set `replicas: N` in Compose.

### Deployment and Rollback

**Image versioning:** images are tagged with the git commit SHA at build time:
```
seshat-api:abc1234
seshat-worker:abc1234
```
No formal release process — for a local MVP, the SHA is sufficient to identify and reproduce any build.

**In-flight jobs during deploy:** the MVP asyncio task queue is in-memory — in-flight jobs are lost on worker restart. Before deploying:
1. Check for active jobs in `TRANSCRIBING`, `EXTRACTING`, or `WRITING` state.
2. Wait for them to complete or fail, or accept the loss — `recoverable=True` on most failure modes means users can retry via `POST /jobs/{id}/retry`.
3. Jobs in `AWAITING_REVIEW` are safe to deploy through — the worker is idle for those jobs and no in-memory state is held.

Jobs stranded in `WRITING` are handled automatically — the worker boot recovery procedure (see Section 8, Worker boot recovery) detects them on startup, cleans the partially-written KB state, and marks them `FAILED` with `recoverable=True`.

**Rollback:** redeploy the previous image tag:
```bash
docker compose up -d --no-deps api worker
```
with the previous SHA tag set in `.env` or the Compose file. KB data and MLflow artifacts are on named volumes and are unaffected by image rollbacks.

---

## 11. Project Structure

```
src/
  seshat/
    api/            # FastAPI routers, job lifecycle, Depends
    pipeline/       # Stages: ingestion, transcription, extraction, rag, writing
    agents/         # Agent registry, base agent, specialised agents
    knowledge_store/  # PostgresKBStore
    vector_store/     # AbstractVectorStore + implementations
    blob_store/       # S3BlobStore
    transcription/    # AbstractTranscriptionService + implementations (AssemblyAI, OpenAI, Deepgram)
    document_loader/  # AbstractDocumentLoader + implementations (used by seshat init)
    config/           # Settings, StrEnums
    secrets/          # AbstractSecretsResolver + implementations
    models/           # Shared Pydantic models (KBNode, ExtractionResult, etc.)
tests/
  unit/
  integration/
scripts/            # operational scripts (e.g. migrate_kb_schema.py); not part of the application package
data/               # gitignored — local KB, MLflow artifacts
  eval_gate.json    # written by seshat eval; read by worker at startup; must be present on the same filesystem as the worker; gitignored
```

---

## 12. Evaluation Strategy

Evaluation is an MVP requirement — the `confidence_threshold=0.7` already in production config must have a calibration basis before any real data is processed.

### Labelled Corpus

A small set of **hand-crafted synthetic transcripts** with manually annotated expected extractions. Synthetic transcripts are preferred over real recordings: the correct extraction is fully controlled and unambiguous.

**Corpus creation scope:** the minimum corpus (10–15 transcripts, ≥15 instances per type, adversarial cases, and merge test pairs) must be completed before the extraction pipeline is considered testable end-to-end. It is built by the developer as part of the Operations / Evaluation tier — it is not generated by the pipeline and not a byproduct of implementation. Treat it as a first-class deliverable with its own task in the implementation plan.

- **Size:** enough transcripts to reach **at least 15 annotated instances per `ConceptType`** across the corpus (normal + adversarial combined). At ~3 instances per type per transcript, this means roughly 5 transcripts per type — achievable with 10–15 total transcripts if they are written to cover all types. Do not start threshold calibration until this minimum is met; below 15 instances per type, a single mis-annotated example shifts precision or recall by 7% or more, making the targets meaningless. Adversarial transcripts count toward the instance total if they contain annotated ground-truth instances of the relevant type.
- **Location:** `tests/eval/corpus/` — versioned with the codebase
- **Format:** one YAML file per transcript: `raw_text` + `expected_nodes: list[KBNode]` + `expected_relationships: list[KBRelationship]`
- **Adversarial transcripts** (add before first real-data run): at minimum one transcript with confident-sounding but unsupported claims, one with ambiguous pronouncements that could be misclassified across `ConceptType`s, and one with injected instruction-like text in the transcript body. These must be hand-crafted and annotated. OOD testing (non-technical jargon, highly ambiguous transcripts) deferred to v2 when real data is available.
- **Merge test cases** (required before release gate): at least 3 pairs of nodes that **should** merge (same concept, paraphrased titles — e.g. `"PostgreSQL as DB"` and `"We chose Postgres"`) and 3 pairs that **should not** merge (same topic, distinct scope — e.g. `"PostgreSQL for operational DB"` and `"PostgreSQL for analytics pipeline"`). These must be embedded in transcripts where both nodes are extracted by the pipeline, then verified to merge or not-merge correctly under `merge_similarity_threshold=0.85`. The release gate is not passed until all 6 pairs produce the correct merge decision.

### Precision / Recall Targets

> **Statistical caveat:** these targets are directional signals, not statistically validated thresholds. At the minimum corpus size (15 instances per type), a single instance shifts precision or recall by ~7%. Treat the targets as a floor for catching gross failures — a model that ignores a concept type entirely, or inverts confidence scores — not as a precise calibration. Widen the corpus with real or higher-volume synthetic data before treating these numbers as validated gates.

Targets are per `ConceptType` — extraction difficulty varies:

| ConceptType | Precision target | Recall target | Notes |
|-------------|-----------------|---------------|-------|
| `ADR` | ≥ 0.80 | ≥ 0.75 | High-stakes; false positives costly |
| `RISK` | ≥ 0.75 | ≥ 0.80 | Recall-biased — missed risks worse than false positives |
| `AGREEMENT` | ≥ 0.75 | ≥ 0.75 | Moderate difficulty |
| `ACTION_ITEM` | ≥ 0.85 | ≥ 0.85 | Simpler extraction; higher bar |

Relationship extraction is evaluated separately — high node precision/recall does not imply correct relationships:

| RelationshipType | Precision target | Recall target | Notes |
|-----------------|-----------------|---------------|-------|
| `ASSIGNED_TO` | ≥ 0.90 | ≥ 0.85 | Wrong assignee is worse than a missed action item |
| `CONFLICTS_WITH` | ≥ 0.75 | ≥ 0.75 | Missed conflicts accumulate silently in the KB |
| `SUPERSEDES` | ≥ 0.80 | ≥ 0.75 | Wrong supersession corrupts decision history |
| `MITIGATES` | ≥ 0.75 | ≥ 0.70 | Cross-type; harder to extract |

The eval corpus format is extended to include `expected_relationships: list[KBRelationship]` alongside `expected_nodes` — both are required fields in the YAML corpus files.

### Threshold Calibration

> **Pre-condition — chunking sanity check:** before running the threshold sweep, print the TextTiling chunk boundaries for each eval transcript and manually verify that boundaries land at genuine topic shifts. Record the result (pass/fail per transcript). If two or more transcripts show systematic mis-segmentation, switch to fixed-size overlapping chunks (see §4, Chunking) and re-verify before proceeding. Extraction metrics are meaningless if the chunks are incoherent — do not skip this step.

1. Run the extraction pipeline over the labelled corpus across a sweep of `confidence_threshold` values (0.5 → 0.9 in 0.05 steps).
2. Plot the precision-recall curve **per `ConceptType`** — a single global curve conflates types with opposing biases (RISK is recall-biased, ADR is precision-biased) and will sacrifice one for the other.
3. Select per-type optimal thresholds from the curves. If a single global value can satisfy all per-type targets simultaneously, use it — otherwise document the per-type trade-off and choose a value with an explicit justification.
4. The default `confidence_threshold=0.7` is the starting point; calibration may revise it. If per-type thresholds are warranted, add them via `ExtractionConfig.per_type_thresholds` — `None` means use the global default for all types.

### Evaluation Entrypoint

`seshat eval` is a first-class CLI command alongside `seshat init`:

```
seshat eval [--threshold 0.7] [--model claude-sonnet-4-6]
```

**Bootstrap note:** `seshat eval` does not check or require `data/eval_gate.json` and does not submit jobs through the API worker — it invokes the extraction pipeline directly in-process. This breaks the circular dependency: the gate file can't exist before eval runs, and eval doesn't need it to run.

Internally wraps `mlflow.genai.evaluate()`: the extraction pipeline is the model under test, the labelled corpus is the eval dataset, and custom scorers compute precision/recall per `ConceptType`. Each eval run is a versioned MLflow experiment — runs can be compared in the MLflow UI to detect regressions from prompt or model changes.

On completion, `seshat eval` writes `data/eval_gate.json` with the following structure:

```json
{
  "passed": true,
  "run_id": "<mlflow_run_id>",
  "timestamp": "<iso8601>",
  "retrieval_recall_at_5": 0.82,
  "precision_recall_by_type": {
    "adr":         {"precision": 0.83, "recall": 0.78},
    "risk":        {"precision": 0.76, "recall": 0.81},
    "agreement":   {"precision": 0.77, "recall": 0.76},
    "action_item": {"precision": 0.88, "recall": 0.86}
  }
}
```

`passed` is `true` only when both release gate conditions are met (see Release Gate below). The file is gitignored — it is a local runtime artifact and must be regenerated per environment.

> **Implementation note:** the exact `mlflow.genai.evaluate()` scorer API and `ChatAgent` interface should be verified against MLflow 3 docs at implementation time — the API is recent and may evolve.

### Release Gate

No real meeting recordings may be processed until `seshat eval` has been run and **both** conditions are met:
1. `recall@5 >= 0.7` on the retrieval baseline
2. Per-`ConceptType` precision/recall targets met (see Precision/Recall Targets above)

**Enforcement:** on startup, the worker reads `data/eval_gate.json`. If the file is absent or `passed=false`, the worker refuses to accept jobs and logs a clear error directing the operator to run `seshat eval`. This is a startup check, not a per-request check — it runs once before the event loop begins accepting work. The check can be bypassed by setting `SESHAT_SKIP_EVAL_GATE=true` in the environment, but this must be an explicit act; the default is enforced.

### Regression Gate

Any change to an agent system prompt, model, or confidence scoring logic must be run through `seshat eval` before promotion. A change that improves one `ConceptType` at the cost of another is a regression — visible in MLflow as a metric degradation vs the baseline run. Re-running `seshat eval` after the change updates `data/eval_gate.json`; a failing run sets `passed=false` and the worker will refuse to start until a passing run is recorded.

---

## Decisions

### Resolved

| Decision | Resolution |
|----------|-----------|
| Queue system | **Python `asyncio` task queue for MVP** — zero new infra. Durability limitation: in-flight jobs lost on worker crash; acceptable for MVP. `AsyncioTaskQueue` is a concrete class; no abstract base or Protocol — no formal interface needed for a single-swap path. Duck-typed contract: `async def enqueue(fn, *args, **kwargs) -> str`, `async def get_status(job_id: str) -> JobStatus`, `async def cancel(job_id: str) -> bool`. The v2 swap to `ARQTaskQueue` (same three method signatures) is a one-line change at the worker entrypoint. |
| LLM orchestration framework | **LangChain.** Rationale: `SemanticChunker` dependency arrives in v2 anyway; MLflow Responses Agent integrates natively with LangChain via `mlflow.langchain.autolog()` — cross-provider verification and prompt caching both supported. LiteLLM dropped. |
| MLflow ↔ LLM framework wiring | **`mlflow.langchain.autolog()`** — instruments all LangChain agent calls automatically. Exact API shape should be verified against MLflow 3 docs at implementation time. |
| Vector store | **pgvector** (via `langchain-postgres`) — shares the Postgres service already required for operational tables; eliminates Chroma as a separate service. Chroma and Qdrant remain available as alternative `VectorStoreProvider` values. |
| Operational database | **Postgres 16** (`ops` schema) — single service for `api_keys`, `jobs`, `init_runs`, `kb_nodes`, `kb_relationships`. SQLite dropped; Postgres is cloud-native and maps directly to AWS RDS / Azure Database for PostgreSQL in production. |

### Deferred to v2

| Decision | Reason deferred |
|----------|----------------|
| RAG in production (vector search scope, graph traversal depth) | Requires real KB data to tune |
| KB model validation (relationship integrity, duplicate node detection, circular reference guards) | Requires real extraction data to understand failure modes |
| Streamlit pipeline graph view — Airflow-style DAG visualisation of job stage execution (status, duration, retries per stage) using Plotly or `streamlit-agraph` | Requires stable pipeline stage model before the view is meaningful |
| Weaviate migration | Meaningful only after KB reaches scale |
| Speaker diarization (AssemblyAI) | Requires production audio samples to validate quality |
| MLflow Plotly integration in Streamlit | Once pipeline is stable |
| JWT authentication with external IdP | Overkill for MVP user base (~10 users); Postgres + API keys is sufficient; upgrade when SSO or multi-tenant access is needed |
| Replace verification agent with ONNX NLI model | Textual entailment (does source quote support extracted claim?) is well-served by discriminative models (DeBERTa, MiniLM-NLI) via `onnxruntime` — no PyTorch needed. Evaluate once MVP eval data shows whether LLM accuracy is actually needed. |
| Replace same-type resolution with ONNX NLI model + embedding similarity | Node-to-node paraphrase/conflict detection is a constrained classification problem. Reuse retrieval-time embeddings for cosine similarity pre-filtering; ONNX NLI model for classification. Evaluate against LLM baseline using `seshat eval`. |
| `AWAITING_REVIEW` timeout SLA (auto-reject stale pending nodes) | Requires a durable scheduled task (ARQ/Redis). The MVP asyncio queue is in-memory and does not survive worker restarts, so a 72h-ahead timer cannot be reliably fired. Ships together with the ARQ/Redis queue swap. |
| Job state push notifications (webhook callbacks + SSE stream) | MVP has no inbound-HTTP consumer — Streamlit polls `GET /jobs/{id}`. Add `callback_url` + `POST` fan-out when an external consumer (CI, Teams bot, etc.) materialises, and `GET /jobs/{id}/stream` (SSE) when the UI matures enough to want low-latency progress. |
| Post-approval node correction API (`PATCH /graph/{node_id}`) | Reviewers edit at approval time via `ApproveRequest.decisions[].edited_content`; already-approved nodes require a direct Postgres update + vector re-embed on MVP. The endpoint earns its keep once the KB moves to Notion/Neo4j. Ships with `NodeMetadata.last_edited_by` / `last_edited_at` at that point. |
| Reranking (cross-encoder pass after vector search) | Skipped in MVP — vector search returns top-K directly and feeds graph traversal. Adopt a reranker (e.g. Cohere `rerank-v3.5`) only if the retrieval baseline (`seshat eval`) shows top-K-only recall@5 is insufficient. At that point, add `rerank_model: str | None` to `RAGConfig` and a second-stage `top_n` cut. |
