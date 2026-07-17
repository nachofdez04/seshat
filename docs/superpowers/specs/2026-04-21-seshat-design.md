# Seshat — Design Spec

**Date:** 2026-04-21
**Status:** Approved

## Overview

Seshat is an API-first GenAI application that transcribes technical meeting recordings, extracts structured decisions, risks, open questions, and action items using a multi-agent pipeline, and writes them to a graph-shaped knowledge base. It is designed for technical users (staff engineers, data architects, heads of engineering), to help them document architecture decisions, and keep them updated.

---

## 0. Problem, Goals, and Success Criteria

### Problem

Technical council members make architecture decisions, surface risks, and assign actions in meetings. These are currently unrecorded or captured informally — they scatter across notes, Slack threads, and memory. There is no searchable, structured record of why a decision was made, what risks were considered, or what was agreed.

### Goals

- Extract structured decisions, risks, open questions, and action items from meeting recordings and write them to a queryable, graph-shaped knowledge base.
- Surface relationships between decisions across meetings (supersession, amendment, conflict, dependency).
- Provide a human review step before any node enters the knowledge base, with confidence scoring to guide reviewer attention.

### Non-Goals (MVP)

- Video input (ffmpeg dependency — deferred to v2)
- Speaker diarization (deferred to v2)
- Multi-tenant or SSO authentication (API keys sufficient for MVP user base)
- Production cloud deployment (runs locally against LocalStack only)
- Integration with Notion, Confluence, or Neo4j (v2 upgrade paths)
- Any UI beyond Streamlit
- Post-approval node editing (`PATCH /graph/{node_id}` — deferred to v2)
- Standalone post-approval node creation (`POST /graph/nodes` — operator+ role, in scope for MVP)
- KB seeding from an existing documentation corpus (`seshat init` — deferred to v2; config stubs and `IngestionSource.INIT` / `CreationSource.INIT` enums are defined but no CLI command, document loader, or `ops.init_runs` table exists)

### Success Criteria

1. `seshat eval` passes the release gate: recall@5 ≥ 0.7 and per-type precision/recall targets met (§12) — no real meeting data is processed until this gate is cleared.
2. A reviewer can process a meeting end-to-end — submit → transcribe → extract → review → KB written — using the Streamlit UI without touching the API directly.
3. The KB is queryable and returns nodes consistent with what was discussed in the source meeting, traceable via `quote_anchors`.

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
3. **Duration check** — after magic bytes are confirmed, audio duration is extracted via mutagen. If `duration > TranscriptionConfig.max_audio_seconds`, the file is rejected with HTTP 400.

**`AudioValidator` methods:**
- `check_size(actual_bytes, max_bytes)` — enforces the size limit (step 1 above)
- `validate_magic_bytes(data, alleged_ext)` — returns the inferred extension; raises `AudioValidationError` if the signature does not match any allowed format (step 2 above)
- `get_duration_seconds(audio_bytes)` — extracts audio duration via mutagen; raises `AudioValidationError` if the bytes are unparseable
- `check_duration(actual_seconds, max_seconds)` — enforces the duration limit (step 3 above)

> **v2 — video input:** video files (.mp4/.mkv/.webm) require ffmpeg audio extraction before transcription. ffmpeg has a known CVE history and is a significant attack surface. Deferred to v2 with explicit security hardening: magic byte validation, subprocess timeout, and system-generated temp filenames (no original filename used in any filesystem path).

### TranscriptDocument

`TranscriptDocument` is the pipeline-internal representation — constructed by the API layer after accepting a `JobSubmissionRequest` (Section 8) and passed from stage to stage.

```python
class TranscriptDocument(BaseModel):
    id: UUID = Field(default_factory=uuid4)  # auto-generated at construction time
    idempotency_key: str | None      # echoed from JobSubmissionRequest; used for deduplication on POST /jobs
    schema_version: str = "1.0"
    source_type: Literal["audio", "text"]   # "video" deferred to v2 (ffmpeg dependency)
    blob_key: str                    # blob storage key where the transcript text is stored; populated after the transcription stage
    metadata: TranscriptMetadata

class TranscriptMetadata(BaseModel):
    meeting_date: date
    participants: list[str] | None = None  # caller-supplied; used for action item assignee resolution
    duration: timedelta | None = None
    language: str = "en"
```

### Transcription

Provider selection happens at startup via the transcription factory reading `TranscriptionConfig.provider`. The pipeline stage calls `transcriber.transcribe(audio_bytes, extension) -> str` only.

**No temporary files:** the uploaded audio file is stored in blob storage as `jobs/{meeting_date}/{job_id}/raw/input.*` immediately after ingestion validation. The transcription stage downloads the raw bytes from blob storage; the orchestrator passes `audio_bytes` and the inferred `extension` directly to `transcriber.transcribe(audio_bytes, extension)` — no temporary files are created.

```python
class AbstractTranscriber(ABC):
    async def transcribe(self, audio_bytes: bytes, extension: str) -> str:
        """Transcribe raw audio bytes and return plain text."""
    # Returns the plain-text transcript. Diarization output (speaker turns) is reserved for v2.
```

**Diarization:** skipped for MVP. AssemblyAI is the recommended provider for v2 — best-in-class speaker diarization in a single API call.

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
```

Artifacts are written at two points per job:

1. **After ingestion** — `raw/input.*` (original file) and `raw/transcript.txt` (normalised plain text output). For `source_type="text"`, `input.*` is the uploaded YAML/JSON file and `transcript.txt` is the `content` field extracted by the validator.
2. **At the start of WRITING** — `curated/extraction.json` is written unconditionally at the beginning of the WRITING stage, before any KB writes. It contains the full `ExtractionResult` including all nodes with their final `status` values (`APPROVED`, `PENDING_REVIEW`, or `REJECTED`). This means the artifact is always present after a job completes, including the all-reject case — which is precisely when a complete audit trail matters most.

> **Deferred:** the `init/` blob prefix (for `seshat init` corpus seeding) is not implemented. See Non-Goals.

This provides a full recovery path (reprocess from raw transcript without re-transcribing) and an audit trail independent of the KB store. Per-node `.md` files are not written — the KB is Postgres-backed and the `extraction.json` is the complete audit artifact.

> **Scope note:** production hardening (private bucket policy, SSE-S3 encryption, IAM service identities, lifecycle rules) is out of scope — this system runs locally against LocalStack only. The bucket structure above is an `S3BlobStore` implementation detail. **Pre-production checklist:** if this system is ever pointed at real AWS S3 (not LocalStack), SSE-KMS encryption and a private bucket policy blocking public access must be enabled before any real meeting content is stored. This is not optional — raw transcripts and extracted decisions are sensitive by default.

> **Why S3 + LocalStack for MVP (not local filesystem):** the master's programme includes a cloud module, and the thesis intentionally exercises a cloud-native persistence path. S3 is the MVP blob store, and LocalStack runs the AWS APIs locally so the same code paths execute in dev and (hypothetically) prod — no dual "local FS vs S3" branch to maintain. The same rationale applies to using AWS Secrets Manager (via LocalStack) instead of falling back to `EnvSecretsResolver` for all local development.

### Text Input Schema

Pre-formatted text input must conform to a defined YAML/JSON schema. `ParsedTextInput` is a Pydantic `BaseModel` (not a dataclass) with fields: `meeting_date` (alias `date`, required), `content` (required), `participants` (optional list of strings). Missing or invalid fields raise `TextValidationError`. The validator rejects non-conforming input at the boundary.

### Init Pipeline (KB Seeding) — Deferred to v2

> **Not implemented.** The `seshat init` CLI command, `AbstractDocumentLoader`/`MarkdownDocumentLoader`, and `ops.init_runs` DB table do not exist. The following config stubs are defined but unused:
> - `DocumentLoaderProvider` enum (`MARKDOWN`) and `DocumentLoaderConfig` in `src/seshat/core/config/settings.py`
> - `SeshatConfig.document_loader: DocumentLoaderConfig | None` and `SeshatConfig.max_concurrent_init_runs: int`
> - `IngestionSource.INIT` and `CreationSource.INIT` enum values
>
> The design below is the intended v2 specification.

---

## 3. Configuration

Single `SeshatConfig` singleton loaded at startup via pydantic-settings. Per-request overrides are applied via a recursive deep-merge onto the base singleton (see `get_request_settings` below) — the singleton is never mutated.

> **Config pattern:** only the root `SeshatConfig` inherits from `BaseSettings` — it owns env var resolution. All nested configs (`ExtractionConfig`, `RAGConfig`, etc.) are plain `BaseModel`. Nested fields are still fully configurable from the environment via `env_nested_delimiter="__"` (e.g. `EXTRACTION__CONFIDENCE_THRESHOLD=0.8`) — pydantic-settings resolves them through the root, not independently. This prevents dual resolution paths where a nested `BaseSettings` could silently read env vars on its own.

```python
class LoggingConfig(BaseConfig):
    level: str = "INFO"                # root log level
    noisy_loggers: dict[str, str] = {  # per-logger overrides for verbose third-party libraries
        "aiobotocore": "WARNING", "botocore": "WARNING", "httpx": "WARNING",
        "langchain": "WARNING", "langchain_core": "WARNING", "langchain_aws": "WARNING",
        "langchain_openai": "WARNING", "mlflow": "WARNING",
        "urllib3.connectionpool": "ERROR",
    }

class APIConfig(BaseConfig):
    max_jobs_per_user_per_hour: int = 10     # per-user job submission rate limit
    max_concurrent_jobs: int = 1             # global cap on TRANSCRIBING/IDENTIFYING/RESOLVING/WRITING jobs
    eval_gate_path: Path = PROJECT_ROOT / "eval_gate.json"
    skip_eval_gate: bool = False             # bypass eval gate at startup; never use in production
    skip_llm_ping: bool = False              # bypass LLM connectivity check at startup
    root_api_key_secret_key: str = "root-api-key"  # Secrets key for the root admin API key

class SeshatConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_nested_delimiter="__",  # e.g. EXTRACTION__IDENTIFICATION__MODEL=claude-sonnet-4-6
        extra="ignore",
    )
    logging: LoggingConfig = LoggingConfig()
    transcription: TranscriptionConfig = TranscriptionConfig()
    vector_store: VectorStoreConfig = VectorStoreConfig()
    vector_index: VectorIndexConfig = VectorIndexConfig()
    kb_store: KBStoreConfig = KBStoreConfig()
    ops_store: OpsStoreConfig = OpsStoreConfig()
    blob_store: BlobStoreConfig = BlobStoreConfig()
    extraction: ExtractionConfig = ExtractionConfig()
    rag: RAGConfig = RAGConfig()
    secrets: SecretsConfig = SecretsConfig()
    observability: ObservabilityConfig = ObservabilityConfig()
    api: APIConfig = APIConfig()
    document_loader: DocumentLoaderConfig | None = None  # only required for seshat init
    max_concurrent_init_runs: int = 1                    # cap on simultaneous seshat init runs
```

All provider fields use `StrEnum` with `auto()` — values are lowercased member names, validated at startup:

```python
class LLMProvider(StrEnum):
    OPENAI = auto()
    ANTHROPIC = auto()
    AZURE_OPENAI = auto()
    BEDROCK_CONVERSE = auto()

class TranscriptionProvider(StrEnum):
    ASSEMBLYAI = auto()
    OPENAI = auto()
    DEEPGRAM = auto()

class VectorStoreProvider(StrEnum):
    PGVECTOR = auto()
    # CHROMA = auto()    # v2
    # QDRANT = auto()    # v2
    # WEAVIATE = auto()  # v2

class SearchMode(StrEnum):
    SEMANTIC = auto()  # cosine-similarity ANN via pgvector
    KEYWORD = auto()   # PostgreSQL full-text search (GIN tsvector / ts_rank_cd)
    HYBRID = auto()    # RRF fusion of SEMANTIC + KEYWORD legs
    AGENT = auto()     # reserved; raises NotImplementedError in SearchEngine

class RerankerProvider(StrEnum):
    COHERE = auto()
    VOYAGE = auto()

class EmbeddingProvider(StrEnum):
    OPENAI = auto()
    AZURE_OPENAI = auto()
    COHERE = auto()   # not yet handled by the factory — implementation deferred

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

The base LLM config class is `_LLMConfig` (private). Three concrete subclasses cover the three LLM roles:

```python
class _LLMConfig(BaseModel):
    provider: LLMProvider
    model: str
    temperature: float = 0.0
    max_retries: int = 3                  # per-call retry attempts on transient errors (API timeout, HTTP 429)
    timeout_seconds: float = 300.0        # per-request HTTP timeout in seconds
    max_concurrent_calls: int = 5         # maximum number of simultaneous LLM calls
    max_output_tokens: int | None = None  # per-call generation token cap; None = no limit
    api_key_secret_key: str | None = None # Secrets key for the LLM API key; defaults to '<provider>_api_key' if not set

class IdentificationLLMConfig(_LLMConfig):
    provider: LLMProvider = LLMProvider.ANTHROPIC
    model: str = "claude-sonnet-4-6"

class GroundingLLMConfig(_LLMConfig):
    provider: LLMProvider = LLMProvider.OPENAI   # must differ from identification.provider — enforced by model_validator at startup
    model: str = "gpt-5.4-nano"
    use_full_transcript: bool = True      # When False, grounding uses only the extracted quote

class ResolutionLLMConfig(_LLMConfig):
    provider: LLMProvider = LLMProvider.ANTHROPIC
    model: str = "claude-sonnet-4-6"
    max_concurrent_calls: int = 10        # per-agent concurrency cap
    max_global_calls: int = 30            # global cap across all resolution agents combined

class ReflectiveLLMConfig(BaseModel):
    enabled: bool = False
    llm: _LLMConfig | None = None  # None = falls back to the stage's primary LLM

class ExtractionConfig(BaseModel):
    concept_types: list[ConceptType] = list(ConceptType)
    identification: IdentificationLLMConfig = IdentificationLLMConfig()
    identification_self_review: ReflectiveLLMConfig = ReflectiveLLMConfig()
    resolution: ResolutionLLMConfig = ResolutionLLMConfig()
    resolution_self_review: ReflectiveLLMConfig = ReflectiveLLMConfig()
    grouped_identification_types: set[ConceptType] = {ConceptType.DECISION}  # types passed through the grouping step; toggleable without code deploy
    grounding: GroundingLLMConfig | None = None  # None = heuristics-only scoring; see Confidence Scoring
    confidence_threshold: float | None = 0.7
    per_type_thresholds: dict[ConceptType, float] | None = None  # overrides confidence_threshold per type; None = use global default for all types
    auto_mode: bool = False
    max_total_input_tokens: int = 2_000_000         # aggregate input token cap across all agent calls in the extraction stage
    max_total_output_tokens: int = 500_000          # aggregate output token cap across all agent calls in the extraction stage
    max_total_embedding_tokens: int = 10_000_000    # aggregate embedding token cap across all RAG calls
    max_hint_nodes: int = 20                        # most recent same-type KB nodes included in extraction-time hint
    max_hint_tokens: int = 1000                     # hard token cap on the hint; oldest nodes dropped first if exceeded
    identification_timeout_seconds: float | None = None  # wall-clock cap on the full identification pass; None = no cap
    resolution_timeout_seconds: float | None = None      # wall-clock cap on the full resolution pass; None = no cap

    # model_validator enforces: grounding.provider != identification.provider (startup error if violated)
    # and logs a warning when grounding=None (heuristics-only)
    # model_validator enforces: confidence_threshold=None is incompatible with auto_mode=True

# Note: there is no result_cache_enabled field in ExtractionConfig.
# Every eval run makes full LLM calls. Intermediate results are cached via read_or_run()
# in src/seshat/eval/cache.py (keyed on corpus file hash) to avoid re-running the same
# example during iteration — this is eval-internal and not exposed in ExtractionConfig.

```

> **Prompt token budget (MVP — no chunking):** the transcript is passed in full; each agent call assembles:
>
> ```
> system_prompt         ~500t   (static per ConceptType; cached after first call)
> kb_hint               ≤1000t  (ExtractionConfig.max_hint_tokens; recency-scoped, oldest dropped first)
> full_transcript       ≤model_context_limit (no per-chunk ceiling in MVP)
> output_schema         ~200t   (structured output schema injected into prompt)
> ─────────────────────────────
> total input           bounded only by model context window (200k for claude-sonnet-4-6)
> ```
>
> Chunking (TextTiling / RecursiveCharacterTextSplitter) is deferred — the MVP passes the full transcript to each agent. `max_total_input_tokens` and `max_total_output_tokens` are aggregate caps across all agent calls for cost control. When chunking is introduced in a future iteration, `max_chunk_count` and `max_transcript_chunk_tokens` will be added to `ExtractionConfig`.

### RAGConfig

```python
class MultiQueryConfig(BaseModel):
    llm: _LLMConfig
    num_variants: int = 3  # 1–10 query variants generated per search

class RerankerConfig(BaseModel):
    provider: RerankerProvider
    model: str
    top_n: int | None = None          # truncate to top-N after reranking; None = keep all
    max_retries: int = 3
    timeout_seconds: float | None = None
    api_key_secret_key: str | None = None  # defaults to '<provider>_api_key' if not set

class RAGConfig(BaseModel):
    enabled: bool = True
    top_k: int = 5        # candidates retained from vector search (fed to graph traversal)
    min_similarity_score: float = 0.5  # minimum similarity score [0, 1]; only applied to the SEMANTIC dense leg — forwarded as score_threshold to AbstractVectorStore.search(); set by RetrievalMetaScorer (argmax macro-F2)
    max_context_tokens: int = 4000
    traversal_max_depth: int = 1                            # direct neighbours only for MVP
    traversal_rel_types: list[RelationshipType] | None = None  # None = all relationship types
    max_concurrent_retrievals: int = 20                     # maximum number of simultaneous RAG retrieval calls
    search_mode: SearchMode = SearchMode.SEMANTIC           # SEMANTIC | KEYWORD | HYBRID | AGENT (reserved)
    keyword_extraction_llm: _LLMConfig | None = None        # required for KEYWORD and HYBRID modes; None disables sparse leg
    multi_query: MultiQueryConfig | None = None             # when set, generates num_variants query variants and fuses with RRF
    reranker: RerankerConfig | None = None                  # when set, reranks retrieval results after vector search
```

`keyword_extraction_llm` controls whether the sparse leg is active. When `None`, any call to `KEYWORD` or `HYBRID` mode logs a warning and returns no results from the sparse leg — `HYBRID` degrades to pure semantic. When set, the LLM extracts 3–6 discriminating keywords from the query before sparse search (see **Sparse leg design** below).

Metadata filters for retrieval are **not** config — they are passed per-job in the request payload.

> **Traversal risk:** unbounded `traversal_max_depth` or large graphs can inflate retrieved context beyond `max_context_tokens`. The assembler truncates at `max_context_tokens` — but a high depth combined with a dense graph will silently drop nodes from the end of the context window. Keep `traversal_max_depth=1` for MVP; increase only after measuring context token usage on real data.

> **Truncation ordering:** before serialising retrieved nodes into the context window, the assembler estimates each node's token cost using `len(title + description) / 4` and greedily includes nodes in order (see below) until `max_context_tokens` is reached. Nodes that would exceed the budget are pre-empted — not serialised at all. The count of pre-empted nodes is logged to MLflow before serialisation begins (alongside the count of any nodes dropped after-the-fact for other reasons). Ordering: `meeting_date DESC NULLS LAST` (most recent first; init-sourced nodes with `meeting_date=None` sort last).

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
    api_key_secret_key: str | None = None  # Secrets key for the embedding API key; defaults to '<provider>_api_key' if not set
    max_indexing_tokens: int = 500_000   # aggregate token cap across all embedding calls in the RAG stage
```

### KBStoreConfig

```python
class KBStoreConfig(BaseModel):
    schema_name: str = "knowledge_base"  # PostgreSQL schema that owns kb_nodes and kb_relationships
    pool_min_size: int = 2
    pool_max_size: int = 10
    connection_secret_key: str = "postgres_url"   # shared with VectorStoreConfig; same Postgres instance

class OpsStoreConfig(BaseModel):
    schema_name: str = "ops"             # PostgreSQL schema that owns jobs, api_keys
    pool_min_size: int = 2
    pool_max_size: int = 10
    connection_secret_key: str = "postgres_url"
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
    timeout_seconds: float | None = None     # per-request HTTP timeout in seconds; None means no limit
    api_key_secret_key: str | None = None    # Secrets key for the transcription API key; defaults to '<provider>_api_key'
```

### ObservabilityConfig

```python
class ObservabilityConfig(BaseModel):
    mlflow_tracking_uri: str = "http://mlflow:5000"
    mlflow_experiment_name: str = "seshat"
    # experiment_id is NOT a config field — it is resolved at startup by setup_mlflow()
    # and returned as a str for the caller to cache in-process.
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
Decision   Risk Agent  Open Ques.  Action Item   [custom via
Agent      (+Risk      Agent        Agent         registry]
(+Decision  hints)    (+OpenQues.  (+ActionItem
  hints)                hints)       hints)
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

- **Pass 1 — Fan-out:** agents run concurrently, one per `ConceptType`. Each agent returns a list of `AnchoredConcept[M]` (individual items with `QuoteAnchor` grounding) or `ConceptGroup[M]` (for types with `grouped_extraction=True`, e.g. DECISION). These are transient models — not `KBNode`. A `PendingNodeBuilder` converts agent output into `_PendingNode` objects, which carry heuristic scores, verification scores, and status bookkeeping. The Action Item agent schema includes an additional field `assignee: str | None` — a named extraction output that Pass 2 resolves. When `identification_self_review.enabled` is `True`, each agent is wrapped in `ReflectiveIdentificationAgent` — see §4 Reflective Agents.
- **Intermediate — Scoring and status assignment:** `_PendingNode` objects are scored (heuristics + optional verification → `ConfidenceBreakdown`), status-assigned (APPROVED / PENDING_REVIEW), then built into `KBNode` via `_PendingNode.build()`. Within-meeting deduplication runs on `_PendingNode` before build.
- **Pass 2 — RAG + Resolution:** runs only after the complete merged Pass 1 node list is in memory. Constructs all relationships. The `assignee` field from the Action Item agent is resolved against `TranscriptMetadata.participants` (exact match first, then case-insensitive prefix) and stored in `KBNode.metadata.concept_fields`; it does not become a `KBRelationship`.

**Grouping agent:** for concept types in `grouped_extraction_types` (default: `{DECISION}`), a second LLM call (`GroupingAgent`) clusters the flat list of `AnchoredConcept` items into thematic `ConceptGroup` objects before `PendingNodeBuilder` processes them. Each group becomes one `_PendingNode` (and eventually one `KBNode`). If the grouping LLM exhausts retries, each item falls back to a singleton group — no items are lost. `GroupingAgent` takes `(llm, config: LLMConfig)` — callers pass `extraction_config.llm`, not the full `ExtractionConfig`.

  **`participants=None` fallback:** when `TranscriptMetadata.participants` is `None`, assignee resolution is skipped for all action items — the `assignee` value is discarded. The action item node is written without an assignee; it is not rejected or downgraded. This is logged as a warning per affected node so the reviewer is aware the assignee could not be resolved.

Cross-chunk assignment (e.g. "as we agreed earlier, you handle this") is handled correctly under this contract: the agent records the assignee name it can see; resolution runs once against the full participant list after all chunks are processed.

### Agent Registry

Each `ConceptType` maps to a registered agent class with its own system prompt. Adding a new concept type = register a new agent + add the type to `ExtractionConfig.concept_types`. The orchestrator discovers agents from the registry at runtime.

### Reflective Agents

Both the identification and same-type resolution families support an optional **reflective mode** that wraps each shallow agent with a second LLM pass. Enabled via `ReflectiveLLMConfig` fields on `ExtractionConfig`:

- `identification_self_review.enabled` — wraps all identification agents in `ReflectiveIdentificationAgent`. The review LLM is `identification_self_review.llm`; falls back to the primary identification LLM when `None`.
- `resolution_self_review.enabled` — wraps all same-type resolution agents in `ReflectiveResolutionAgent`. Cross-type agents are never wrapped. The review LLM is `resolution_self_review.llm`; falls back to the primary resolution LLM when `None`.

Both flags default to `False`. When disabled, shallow agents are instantiated unchanged.

**`ReflectiveIdentificationAgent`** adds an **extract → validate → filter** pass. After extraction, a single validation call checks each item for logical compliance (does it satisfy the extraction rules?) and semantic compliance (does the description match the quote?). Items that fail are discarded. On any validation failure (retries exhausted, count mismatch), all extracted nodes are returned as-is.

**`ReflectiveResolutionAgent`** adds a **competing-hypothesis tiebreaker** for ambiguous same-type entries. The inner agent signals uncertainty via an optional `alt_rel_type` field — populated only when two relationship types are genuinely competing for a pair. Only contested entries are sent to a tiebreaker call that adjudicates between the two candidates; uncontested entries bypass it entirely. On any tiebreaker failure, the original `rel_type` is kept. Cross-type agents are not wrapped — no quality gap observed in eval.

Both agents use the **subclass-and-delegate** (proxy) pattern: they inherit from the same base as the shallow agents, hold an `inner` instance, and delegate all abstract properties to it. The orchestrator and registries require no changes to support either mode.

Full design details: [docs/superpowers/specs/2026-06-17-reflective-agents.md](2026-06-17-reflective-agents.md).

### Prompt Caching

Agent system prompts are static per `ConceptType` and reused across every job. Prompt caching is a **first-class design requirement**, not an optimisation. Full caching strategy (Anthropic `cache_control` headers, OpenAI automatic prefix caching, LLM wrapper ownership, cold-start behavior, and MLflow observability) is defined in [docs/superpowers/specs/2026-04-27-prompt-interaction-design.md §2.3](2026-04-27-prompt-interaction-design.md).

### Core Enums

```python
class ConceptType(StrEnum):
    DECISION = auto()
    RISK = auto()
    ACTION_ITEM = auto()
    OPEN_QUESTION = auto()

class RelationshipType(StrEnum):
    MITIGATES = auto()       # Risk → Decision
    BLOCKS = auto()          # Risk → Decision | Risk → OpenQuestion (active blocker)
    CONFLICTS_WITH = auto()  # Decision → Decision (same ConceptType only)
    DEPENDS_ON = auto()      # Decision → Decision
    SUPERSEDES = auto()      # Decision → Decision (fully replaces)
    AMENDS = auto()          # Decision → Decision (partial update or clarification)
    RESOLVES = auto()        # Decision → OpenQuestion (closes a deferred question)
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
    quote_anchors: list[QuoteAnchor] = []  # anchored positions of source quotes within the transcript blob
    status: NodeStatus
    state: NodeState = NodeState.CURRENT
    metadata: NodeMetadata

class KBRelationship(BaseModel):
    rel_id: UUID = Field(default_factory=uuid4)  # surrogate PK; used by delete_relationship / get_relationship
    source_id: UUID
    target_id: UUID
    rel_type: RelationshipType
    job_id: str           # which job created this relationship; duplicates source_id → KBNode.metadata.job_id
    source: RelationshipSource = RelationshipSource.PIPELINE
    created_at: datetime  # genuinely non-derivable; not available via the source node

    # Implementation note: `job_id` is intentionally duplicated here for query convenience —
    # the alternative is a join through source_id → ops.kb_nodes.metadata->>'job_id'.
    # If duplication becomes a maintenance concern, drop job_id and use the join instead.

class NodeStatus(StrEnum):
    APPROVED = auto()
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
    MANUAL = auto()       # node created directly by an operator at review time

class IngestionSource(StrEnum):
    PIPELINE = auto()  # extracted from a meeting recording via the normal pipeline
    INIT = auto()      # seeded from a document corpus via seshat init
    MANUAL = auto()    # created directly by an operator (review time or future standalone endpoint)

class RelationshipSource(StrEnum):
    PIPELINE = auto()  # created by the resolution pipeline
    MANUAL = auto()    # created via POST /graph/relationships by an operator
    INIT = auto()      # created during seshat init seeding

class NodeMetadata(BaseModel):
    job_id: str                               # UUID4; same namespace for both JOB and INIT ingestion (refs ops.jobs.job_id or ops.init_runs.job_id)
    meeting_date: date | None = None          # None when ingestion_source=INIT
    participants: list[str] | None = None     # best-effort; None when unknown or ingestion_source=INIT
    ingestion_source: IngestionSource = IngestionSource.PIPELINE
    team: str | None = None
    project: str | None = None
    domain: str | None = None
    approved_by: str | None = None
    approved_at: datetime | None = None
    approval_method: ApprovalMethod | None = None
    pending_reason: str | None = None         # human-readable reason why the node is PENDING_REVIEW (e.g. "below confidence threshold")
    corrected_by: str | None = None    # set when a reviewer provides edited_content in NodeDecision
    corrected_at: datetime | None = None   # set to the same timestamp as approved_at (corrections only happen at approval time in v1)
    correction_reason: str | None = None
    confidence_breakdown: ConfidenceBreakdown | None = None  # populated during extraction; echoes ExtractionResult.confidence_breakdowns[node.id]
    concept_fields: dict[str, Any] | None = None             # type-specific extracted fields not in ConceptModel base (e.g. assignee, due, type)
```

```python
class ExtractionResult(BaseModel):
    job_id: str
    nodes: list[KBNode]
    relationships: list[KBRelationship] = []
    confidence_breakdowns: dict[str, ConfidenceBreakdown] = {}  # str(node.id) → breakdown

class ResolutionResult(BaseModel):
    job_id: str
    relationships: list[KBRelationship]
    failed_sources: list[FailedResolutionSource]            # nodes whose resolution failed (retries exhausted)

class FailedResolutionSource(BaseModel):
    node_id: UUID
    concept_type: ConceptType
```

`ExtractionResult` is the output of the extraction pass and the payload returned by `GET /jobs/{id}/results` while a job is in `AWAITING_REVIEW`. Relationships are produced in a separate `ResolutionResult` after node approval (see §5).

> **Vector store indexing:** one vector per `KBNode` — the embedding is generated from `title` + `description` after extraction. `NodeMetadata` travels with the vector for runtime metadata filtering (applied as `>=` comparisons for `min_confidence`, equality for all other fields). The `confidence`, `node_type`, and `node_state` fields are duplicated from `KBNode` intentionally — the vector store needs them for filtering without a round-trip to the KB store. **Invariant:** `NodeMetadata.confidence`, `NodeMetadata.node_type`, and `NodeMetadata.node_state` must always equal the corresponding fields on `KBNode` — they are written together in the same transaction and neither is ever updated independently.
>
> **Indexing timing:** vectors are written to the vector store during the `WRITING` stage, as part of the same Postgres transaction as the KB row (§4, Node Lifecycle Invariant). They are **not** written after extraction. Nodes from a job in `AWAITING_REVIEW` are not yet retrievable via RAG — invisible to concurrent jobs until the reviewing job reaches `DONE`. This is intentional: unreviewed nodes should not influence future extractions.

### Confidence Scoring

**Heuristics** is the sole continuous confidence signal; `KBNode.confidence` equals the heuristics score directly.

**Grounding** is a hard binary gate, not a blended signal. When `ExtractionConfig.grounding` is configured, a separate lightweight agent independently judges each extracted node. A node that fails (`grounding_passed=False`) is rejected outright regardless of its heuristics score. When grounding is disabled or retries are exhausted, `grounding_passed` is `None` and the decision falls through to the heuristics threshold alone.

1. **Grounding agent** — a separate lightweight agent (cheap model: `gpt-5.4-nano` via OpenAI, or another low-cost provider) that receives the extraction and source quote and answers a binary "is this well-supported?" question. Must use a different `LLMProvider` than the extraction agent — same-provider grounding produces correlated errors (enforced by `model_validator`). Example pairing: extraction on `anthropic`, grounding on `openai`.
2. **Heuristics** — always active. Formula:

```
heuristics_score = (
    0.3 * quote_length_signal(quote_anchors)           # word-count based; saturates at 35 words
  + 0.3 * title_specificity(title)                     # composite: NE presence + qualifier + word count
  + 0.4 * directness(description)                      # multiplicative penalty for hedging/passive/future tense
)
```

`title_specificity` and `directness` are rule-based classifiers — no model calls. Both are implemented using spaCy's dependency parser and NER (no LLM required). The formula, sub-scores, and weights are fixed — implementation must match them exactly, as `seshat eval` calibrates against this contract.

**`quote_length_signal` scoring:**
- Continuous signal in [0, 1]; based on word count of the source quote, saturating at 35 words.

**`title_specificity` scoring (continuous composite):**
```
title_specificity = 0.45 * has_named_entity + 0.35 * has_qualifier + 0.20 * word_count_signal
```
- `has_named_entity` (0 or 1) — spaCy NER labels `ORG`/`PRODUCT`, CamelCase tokens, or domain tech-term lexicon match.
- `has_qualifier` (0 or 1) — prepositional phrase or adverbial clause on the root verb.
- `word_count_signal` — continuous in [0, 1], saturates at 8 words.

**`directness` scoring (multiplicative penalty):**
- Starts at 1.0; each penalty multiplied in:
  - × 0.5 if hedging tokens present ("should", "might", "could", "may", "would", etc.)
  - × 0.75 if passive voice detected (spaCy `auxpass`/`nsubjpass`)
  - × 0.75 if future tense detected ("will", "shall")
  - × 0.75 if no direct object or complement on the root verb

> **Note:** the original spec defined bucketed (0.0 / 0.5 / 1.0) title_specificity and directness scoring. The implementation uses continuous composite / multiplicative formulas which are more nuanced and better suited to calibration. The `seshat eval` gate calibrates against the implemented formula, not the original buckets.

**Auto-approval policy:**

| `grounding_passed` | heuristics vs threshold | Outcome |
|---|---|---|
| `None` (disabled or retries exhausted) | ≥ threshold | APPROVED (auto or threshold) |
| `True` | ≥ threshold | APPROVED (auto or threshold) |
| `False` | any | REJECTED |
| any | < threshold | REJECTED (auto-mode) or PENDING_REVIEW (manual mode) |

When `grounding=None`, no grounding agent runs and startup issues a warning. The heuristics threshold calibrated in a heuristics-only run is not directly comparable to one calibrated with grounding enabled — recalibrate when toggling grounding.

```python
class ConfidenceBreakdown(BaseModel):
    grounding_enabled: bool        # True when a GroundingLLMConfig is present on ExtractionConfig
    grounding_passed: bool | None  # None when grounding is disabled or retries exhausted
    heuristics: float              # always present; echoes KBNode.confidence
```

### Prompt Injection Mitigation

Transcript text and retrieved KB context are untrusted inputs injected into agent prompts. Full security model (structural isolation, output validation, source quote verification, context sanitisation, second-order risk, and per-agent coverage table) is defined in [docs/superpowers/specs/2026-04-27-prompt-interaction-design.md §3](2026-04-27-prompt-interaction-design.md).

### Chunking

**MVP — no chunking:** the full transcript is passed to each agent in a single call. The model's context window (200k tokens for `claude-sonnet-4-6`) is large enough for typical meeting transcripts. `max_total_input_tokens` and `max_total_output_tokens` enforce aggregate cost caps across all agent calls.

**Deduplication (within-meeting):** after all agents complete, `_PendingNode` objects are deduplicated before being built into `KBNode`. The merge criterion is title exact-match + type equality: two `_PendingNode` objects of the same `ConceptType` with identical normalised titles (lowercased, whitespace-collapsed) are the same concept — the later one (higher position in agent output order) is kept, the earlier discarded. No `SUPERSEDES` relationship is created within a single job — `SUPERSEDES` is reserved for cross-meeting evolution only.

**v2 — chunking (when needed):** if transcripts regularly exceed model context limits or per-call cost becomes a concern, introduce TextTiling (NLTK) as the first chunking strategy. Per-chunk agent calls produce multiple `_PendingNode` objects for the same concept; cross-chunk deduplication must be extended with the cosine-similarity fallback at that point. If TextTiling boundaries are poor (meeting-transcript prose is not well-served by topic-shift detection), fall back to `RecursiveCharacterTextSplitter` (LangChain, 500-token windows, 100-token overlap). Semantic chunking and diarization-based splitting remain v2+ options requiring production audio to validate.

### Status Assignment

- `confidence >= threshold` → `status=APPROVED`, `approval_method=THRESHOLD`, `approved_by="system"`, `approved_at=<extraction timestamp>`
- `confidence < threshold` → `status=PENDING_REVIEW`; `approval_method`, `approved_by`, `approved_at` remain `None` until `POST /jobs/{id}/approve`
- `auto_mode=True` → all nodes `status=APPROVED`, `approval_method=AUTO`, `approved_by=<submitting user_id>`, `approved_at=<job submission timestamp>`, regardless of confidence

### Node Lifecycle Invariant

The pipeline is **append-and-state-only** — extracted content is never modified after creation. A node's `title`, `description`, `quote_anchors`, and `confidence` are immutable once written. The only permitted mutation is `update_node_state()` on existing nodes when a `SUPERSEDES` or `AMENDS` relationship is established by the resolution step. If a later meeting revisits an existing decision, the resolution step creates a new node and expresses the relationship via `SUPERSEDES`, `AMENDS`, or `CONFLICTS_WITH` — it never overwrites the original. This preserves the full decision history in the graph.

Consequences:
- **State transitions:** when a new node carries a `SUPERSEDES` or `AMENDS` relationship, the pipeline calls `update_node_state()` on the target node — advancing its `state` to `SUPERSEDED` or `AMENDED` respectively. This is the only mutation the pipeline applies to an existing node. A `CONFLICTS_WITH` relationship does **not** trigger a state transition — both nodes remain `NodeState.CURRENT`. `CONFLICTS_WITH` is a graph-level annotation only; reviewers discover active conflicts via the graph query UI (Screen 4 highlights them). This is intentional: a conflict between two `CURRENT` nodes is a signal for human judgment, not an automatic state change.

  **Stale CONFLICTS edges:** when one party in a `CONFLICTS_WITH` pair is later superseded (its `state` advances to `SUPERSEDED`), the edge is not deleted — the Node Lifecycle Invariant is append-only. Instead, `GET /graph/{node_id}`, `GET /graph/{node_id}/impact`, and Screen 4 filter out `CONFLICTS_WITH` relationships where either party has `state != CURRENT`. The stale edge remains in `ops.kb_relationships` for historical audit but is invisible on all normal query paths. A future "show historical graph" view can expose it explicitly.
- **Store sync and recovery:** the KB store (`PostgresKBStore`) and the vector store (pgvector) share the same Postgres instance. Each node write is a single database transaction: `write_node()` inserts the KB row and `upsert()` inserts the vector embedding atomically. A crash mid-transaction leaves no partial state — Postgres MVCC ensures either both writes are committed or neither is. `NodeState` transitions (`update_node_state()`) are also single-row updates within a transaction.
- **Concurrent pipeline runs:** `api.max_concurrent_jobs=1` (default) prevents two jobs from running the pipeline simultaneously, which eliminates concurrent `update_node_state()` races at MVP scale. If `api.max_concurrent_jobs` is raised, `update_node_state()` transitions remain safe — setting `state=SUPERSEDED` twice on the same node is idempotent — but resolution quality may degrade because both jobs read the KB before either writes to it.
- **Human corrections:** edits happen at review time via `ApproveRequest.decisions[].edited_content`. When `edited_content` is non-null, the node's `title` and `description` are updated with the human-supplied values, and `NodeMetadata.corrected_by` / `corrected_at` are set alongside `approved_by` / `approved_at`. `corrected_by` is the single field distinguishing human-corrected content from unmodified LLM output — queries on the KB can filter on it to assess how often auto-extraction required correction. A post-approval `PATCH /graph/{node_id}` is deferred to v2.

#### Soft-Delete / Archival (Deferred)

`NodeStatus` currently has three values: `APPROVED`, `PENDING_REVIEW`, `REJECTED`. A fourth value `ARCHIVED` is deferred to a future tier.

`ARCHIVED` would model a soft-delete for approved nodes that have already been through resolution and carry relationships — a hard delete would orphan those edges and remove provenance. An archived node remains in the KB for audit purposes but is excluded from retrieval (VS embedding deleted on archival, filter added to all query paths).

When implemented, the required changes are:
- Add `NodeStatus.ARCHIVED` to the enum
- Add a `PATCH /graph/nodes/{id}/archive` endpoint (operator-only)
- Delete the VS embedding when the status transitions to `ARCHIVED`
- Exclude `ARCHIVED` nodes from all `NodeFilter` query paths by default

Until then, the only node removal path is hard delete via `DELETE /graph/nodes/{id}` (admin only). Hard deletion is safe for nodes with no relationship history. For nodes that were the source of `SUPERSEDES` or `AMENDS` relationships, `NodeRepository.delete_node` automatically reverts any affected targets back to `CURRENT` state — provided no other surviving node still supersedes/amends them. This applies to both manual and pipeline nodes, since manual nodes can also accumulate resolved relationships via `POST /graph/nodes/resolve`.

---

## 5. RAG + Resolution Layer

RAG runs **after** extraction, not before. Extraction agents receive a lightweight same-type KB hint at prompt time (see below); the full retrieval and relationship resolution pass happens once all new nodes are extracted.

```
ExtractionResult (new nodes + relationships from Pass 2)
      │
      ▼
 RAG Service — per new node:
 1. Embed node (title + description)
 2. Vector search → top-K candidate KB nodes (same type)
 3. Graph traversal on top-K nodes (KB Store)
      │
      ▼
 Resolution — two parallel orchestrator calls:
 ┌─────────────────────────┐
 ▼                         ▼
Same-type resolution    Cross-type resolution
(SUPERSEDES/AMENDS/     (MITIGATES/BLOCKS/
 CONFLICTS per type)     DEPENDS_ON/RESOLVES)
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

- **Embedding target:** each new `KBNode` is embedded from `title + description` — node-to-node comparison is homogeneous and avoids the semantic distance problem of comparing raw transcript chunks against distilled KB summaries. **Known limitation:** `text-embedding-3-small` is a general-purpose semantic model — it conflates semantic similarity with logical coupling. Two Decisions on the same topic but independent may score highly similar; a Risk and a Decision with a `MITIGATES` relationship may score dissimilar. This is the primary reason the retrieval baseline must be measured before real use. If recall@5 < 0.7, switching the embedding model (e.g. a domain-specific or fine-tuned encoder) is the first tuning lever, before increasing `top_k`.
- **Vector search:** handled by `SearchEngine` — SEMANTIC, KEYWORD, or HYBRID mode per `rag_config.search_mode`. Returns top-K candidates. Cross-type search is intentional: a DECISION source node may need to resolve against RISK or OPEN_QUESTION targets; restricting to same-type would miss valid candidates for cross-type resolution agents. Optional reranking via `AbstractReranker` (`CohereReranker` or `VoyageReranker`) is applied after retrieval if `rag_config.reranker` is configured.
- **Graph traversal:** structural retrieval from KB Store — direct neighbours of top-K candidates (both inbound and outbound edges, depth=1). For MVP (Postgres): SQL join on `ops.kb_relationships`. For Neo4j: Cypher query. Same interface.

### Resolution

Resolution is two parallel LLM calls routed through a `ResolutionRegistry` — both pass run concurrently. The registry fans out to typed agents:

- `SameTypeResolutionRegistry` — one `BaseSameTypeResolutionAgent` per `ConceptType`; validates anti-symmetry (`supersedes`, `blocks`, `depends_on` cannot have A→B and B→A simultaneously) and mutual exclusion (`supersedes` + `conflicts_with` or `supersedes` + `amends` on the same pair are invalid).
- `CrossTypeResolutionRegistry` — one `BaseCrossTypeResolutionAgent` per allowed `(source_type, target_type)` pair (9 pairs total: all permitted combinations from the relationship schema table below).

Each agent receives per-source candidate lists (`per_source_targets: dict[UUID, list[KBNode]]`) from the RAG retrieval step, runs one LLM call per source node, and collects `ResolvedRelationship` objects. Global concurrency is bounded by `ResolutionLLMConfig.max_global_calls`; per-agent concurrency by `ResolutionLLMConfig.max_concurrent_calls`. Sources that exhaust retries are collected as `FailedResolutionSource` records and included in `ResolutionResult.failed_sources`.

**LLM interface:** agents use positional indices (0 = source node, 1+ = target nodes) in prompts to avoid UUID parsing complexity in LLM output. The agent maps indices back to UUIDs after the call.

Resolution is two parallel LLM calls routed through the registry — both run concurrently:

- **Call 1 — same-type resolution:** for each concept type, classifies each new node against its KB candidates as `SUPERSEDES`, `AMENDS`, `CONFLICTS_WITH`, or no relationship. The resolution agent prompt must include the following operational criteria — without them the agent will guess and produce inconsistent history:

  - **`SUPERSEDES`**: the new node renders the prior decision actionable-irrelevant — the old decision would no longer be followed. The prior node's `NodeState` transitions to `SUPERSEDED`.
  - **`AMENDS`**: the new node narrows, extends, conditionally qualifies, or corrects a detail of the prior decision while leaving it broadly active. The prior node's `NodeState` transitions to `AMENDED`.
  - **`CONFLICTS_WITH`**: both decisions are currently active but mutually incompatible. Neither node's state changes — see Node Lifecycle Invariant (Section 4).
  - **No relationship**: the new node covers the same topic but is independently valid (e.g. a separate decision about a different component). No state change.
  - **Ambiguity signal (`alt_rel_type`):** when the agent is genuinely uncertain between two specific relationship types (never for null assignments, never for clear-cut cases), it sets `alt_rel_type` to the runner-up. When `resolution_self_review.enabled` is `True`, entries with `alt_rel_type` set are sent to a tiebreaker call — see §4 Reflective Agents. Uncontested entries bypass the tiebreaker entirely.

  The eval corpus must include at least 2 labelled examples per relationship type (SUPERSEDES, AMENDS, CONFLICTS, no-relationship) to validate that the agent applies these criteria correctly.
- **Call 2 — cross-type resolution:** across all new nodes, resolves `MITIGATES`, `BLOCKS`, `DEPENDS_ON`, and `RESOLVES`. The relationship schema constrains which pairings are evaluated — no N×N comparison across all types:

| Relationship | Source → Target |
|---|---|
| `MITIGATES` | Risk → Decision \| ActionItem → Risk |
| `BLOCKS` | Risk → Decision \| Risk → OpenQuestion \| Risk → ActionItem \| OpenQuestion → Decision \| OpenQuestion → ActionItem |
| `DEPENDS_ON` | Decision → Decision \| ActionItem → Decision (same-type only) |
| `RESOLVES` | Decision → OpenQuestion |

Nine cross-type agent instances in total, one per permitted `(source_type, target_type)` pair.

Once both calls return, a **heuristic validation step** merges the outputs and rejects malformed relationships before the result is finalised:

- A node cannot both `SUPERSEDES` and `CONFLICTS_WITH` with the same target
- `SUPERSEDES` and `AMENDS` are mutually exclusive on the same (source, target) pair
- Relationship direction must match the schema (e.g. a Risk cannot `DEPENDS_ON` a Decision)
- A new node cannot relate to a target node of a different `ConceptType` unless the schema permits it

Validation failures are logged and the offending relationship is dropped — they do not fail the job.

Heuristic validation operates on the `KBRelationship` list only — validation failures drop the offending relationship but do not otherwise affect the resolution result.

### NodeRetriever and SearchEngine

**`NodeRetriever`** is the RAG retrieval class used by `ExtractionOrchestrator` during the resolution pass. For each approved node it builds a `query = node.vector_store_text`, calls `SearchEngine.search(...)` (with `score_threshold=rag_config.min_similarity_score`), optionally reranks via `AbstractReranker`, then expands results with direct KB neighbours. A `_ContextBudget` caps total retrieved token cost. Both `SearchEngine` and `AbstractReranker` are wired at startup via `build_extraction_orchestrator`.

**`SearchEngine`** handles `SEMANTIC`, `KEYWORD`, and `HYBRID` modes with optional multi-query fan-out (RRF fusion) and keyword extraction via LLM. `SearchMode.AGENT` is not handled — the catch-all raises `ValueError`. It is constructed by `get_search_engine(config, vector_store)` in `bootstrap.py` and passed to `NodeRetriever`.

`SearchEngine` constructor: `(rag_config, vector_store: AbstractVectorStore, keyword_llm, multi_query_llm)`. It holds a direct reference to the `AbstractVectorStore` instance (not a callable `VectorSearchFn`). Its `search()` method delegates to `_semantic`, `_keyword`, or `_hybrid` based on `rag_config.search_mode`. `SearchMode.AGENT` is not handled by a dedicated branch — the catch-all raises `ValueError`.

### Retrieval Quality Baseline

The `top_k=5` default must be justified against a measured baseline before MVP ships — not tuned by intuition.

**Baseline approach:**
1. Seed a test KB from the eval corpus (`tests/eval/corpus/`) by running the extraction pipeline directly against the corpus fixtures and writing results as `APPROVED` nodes.
2. For each labelled transcript in the eval corpus, run extraction to produce new nodes, then embed them and search against the seeded KB.
3. For each new node with a known ground-truth match in the KB, measure **recall@5**: fraction of known matches appearing in the top-5 retrieved candidates.
4. If recall@5 is below 0.7 with default settings, tune `top_k` upward or adjust the embedding model before locking defaults.
5. Also measure **precision@5** alongside recall@5. High recall with low precision@5 (suggested floor: ≥ 0.6) means resolution agents receive many irrelevant candidates — noisy resolution, not just incomplete retrieval. Low precision@5 with acceptable recall@5 is the signal to invest in reranking (see Decisions Deferred).

**Threshold calibration** is performed by `RetrievalMetaScorer.sweep_threshold()`, which
sweeps `score_threshold` across [0, 1] and selects the value that maximises **macro-F2**
(beta=2) — a recall-weighted metric that prevents the degenerate threshold=0 solution.
Positive corpus examples contribute F2; negative examples (no expected matches) contribute
specificity (1 if nothing returned, 0 otherwise). Both feed the macro average.

Calibration is **per search mode** — each mode uses a different score scale (cosine
similarity for `SEMANTIC`, `ts_rank_cd` for `KEYWORD`, RRF `1/(60+rank)` for `HYBRID`),
so thresholds are not portable across modes. `EvalConfig.retrieval_score_thresholds` is
a `dict[SearchMode, float]`; absent keys default to 0.0. The `SEMANTIC` threshold is also
set as `RAGConfig.min_similarity_score` for the production pipeline; `KEYWORD` and `HYBRID`
suggest 0.0 because their score distributions have no exploitable gap.

**Sparse leg design:** The sparse leg requires a `keyword_extraction_llm` — raw natural
language fed directly to a tsquery function uses implicit AND semantics, which filters out
short KB nodes that contain only some query terms. The LLM extractor produces a tight set
of 3–6 discriminating keywords (proper nouns, named tools, specific technical terms); these
are joined with `|` and passed to `to_tsquery("english", ...)`, so any matching keyword is
sufficient. The `ts_content` column (`tsvector GENERATED ALWAYS AS (to_tsvector('english',
document)) STORED`) is created lazily on first sparse or hybrid search and backed by a GIN
index for index-speed lookups. Ranking uses `ts_rank_cd` (cover density), which rewards
documents where query terms cluster together. The extractor model ID is included in the
eval cache fingerprint so cached results are invalidated when the extractor changes.

**Hybrid eval pre-filter:** The eval runner applies the calibrated `SEMANTIC` threshold as
a dense pre-filter for `HYBRID` mode, matching production behaviour where `min_similarity_score`
is forwarded to the dense leg before RRF fusion. The cache key embeds this threshold, so
clearing the hybrid cache is required if the semantic threshold is recalibrated.

`seshat eval` runs this retrieval baseline as an independent pass — each pass produces its own MLflow run, linked to the same experiment.

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
| `AbstractTranscriber` | Abstract + implementations | Three providers are enumerated (AssemblyAI, OpenAI, Deepgram — no `DeepgramTranscriber` implementation yet); factory-swappable at startup via `get_transcriber()`, which wraps the concrete transcriber in `TrackingTranscriber` (usage tracking) before returning |
| `AbstractDocumentLoader` | Abstract + implementations | v2 loaders (Notion, Confluence) are network-backed and behaviourally different from `MarkdownDocumentLoader` |
| `AbstractSecretsResolver` | Abstract + implementations | Two providers in MVP (ENV, AWS); v2 adds Azure and Vault — startup factory swap |
| `AsyncioTaskQueue` | Concrete class, duck-typed swap | One queue for MVP; the v2 `ARQTaskQueue` exposes the same three methods (`enqueue / get_status / cancel`) — no formal protocol needed, the swap is a one-line change at the worker entrypoint |

### Shared Filter and Result Types

```python
class NodeFilter(BaseModel):
    # All fields optional. Filters AND together; None = no constraint on that field.
    node_type: ConceptType | None = None
    job_id: str | None = None             # filter by originating job ID
    team: str | None = None
    project: str | None = None
    domain: str | None = None
    ingestion_source: IngestionSource | None = None
    min_confidence: float | None = None   # applied as confidence >= min_confidence
    status: NodeStatus | None = None
    state: NodeState | None = None
    meeting_date_from: date | None = None   # inclusive lower bound on NodeMetadata.meeting_date
    meeting_date_to: date | None = None     # inclusive upper bound on NodeMetadata.meeting_date
    limit: int = 1000                       # max nodes to return (1–10 000); default 1000
    offset: int = 0                         # pagination offset
    # AbstractVectorStore.search() ignores state, meeting_date_from, meeting_date_to, limit, offset —
    # these fields are only applied by PostgresKBStore.query() (WHERE clauses on ops.kb_nodes).
```

```python
class SearchResult(BaseModel):
    node_id: UUID
    score: float   # provider-native similarity score; higher = more similar
```

Both stores accept the same `NodeFilter` type for runtime filtering so filter semantics stay identical whether a request hits the KB store (for graph queries) or the vector store (for similarity search with metadata narrowing).

### Interfaces

All methods are async — the pipeline runs in an asyncio context. `PostgresKBStore` uses an async-native Postgres client (`asyncpg` via `langchain-postgres`). The async decision must be made before implementation — retrofitting sync-to-async is expensive.

**`PostgresKBStore`** exposes: write a node (plain `INSERT`, returns its UUID as `str`; relationships are always written separately); write a relationship (both source and target UUIDs required); transition a node's state (the only pipeline-legal mutation on an existing node); retrieve a node by ID; retrieve a node's neighbours filtered by relationship type(s) and direction (`inbound`, `outbound`, or `both`); query nodes by `NodeFilter`.

**`AbstractVectorStore`** exposes: upsert a node embedding (node ID + text + metadata); `search_dense(query, top_k, node_filter, exclude_job_id, score_threshold)` — cosine similarity over the dense vector index; `search_sparse(query, top_k, node_filter, exclude_job_id)` — full-text search via `ts_rank_cd`; `search(query, top_k, ..., mode)` — concrete convenience dispatcher that routes to `search_dense` or `search_sparse` based on `mode` (used by callers that receive `SearchMode` at runtime, e.g. `NodeRepository` and eval runner); `update_metadata(node_id, patch)` — JSONB merge patch on an existing embedding; `delete(node_id)`. `SearchEngine` calls `search_dense`/`search_sparse` directly. Supported filter fields are declared via `get_supported_filter_fields()` — filters outside this set are silently ignored with a warning log.

**`S3BlobStore`** exposes: put an artifact at a path key; get an artifact by key; check whether a key exists.

```python
class S3BlobStore:
    async def put(self, key: str, data: bytes) -> None: ...
    async def get(self, key: str) -> bytes | None: ...  # returns None on 404; never raises on missing key
    async def exists(self, key: str) -> bool: ...
```

### Write Order and Consistency

`PostgresKBStore` and `PGVectorStore` share the same Postgres instance. Each node write (KB row + vector embedding) is a single database transaction — no coordination protocol needed. If the transaction fails, neither store is written; the job transitions to `FAILED` with `recoverable=True` and the full pipeline can be retried.

`S3BlobStore` artifact writes (`curated/extraction.json`) happen at the start of WRITING, before KB transactions begin, and are non-fatal — if the blob write fails, the job continues and nodes are still written to Postgres. The raw transcript is already in blob storage for reprocessing regardless.

### MVP: PostgresKBStore

`PostgresKBStore` needs no config object — it resolves the connection string from secrets at startup (`seshat/postgres_url`), shared with the pgvector store.

KB nodes and relationships are stored in the `ops` schema alongside the operational tables. Two tables, managed by Alembic (same migration path as `ops.jobs`, `ops.api_keys`, and `ops.init_runs`):

**`knowledge_base.kb_nodes`** — one row per `KBNode`. The `PostgresKBStore` stores nodes in the `knowledge_base` schema (not `ops`). Columns: `node_id` (PK), `schema_version`, `job_id`, `type` (ConceptType), `title`, `description`, `confidence`, `quote_anchors` (JSONB), `status` (NodeStatus), `state` (NodeState, default `current`), `metadata` (JSONB), `created_at` (TIMESTAMPTZ).

**`knowledge_base.kb_relationships`** — one row per `KBRelationship`. Columns: `rel_id` (surrogate PK, UUID), `source_id` (FK → kb_nodes), `target_id` (FK → kb_nodes), `rel_type` (RelationshipType), `job_id` (UUID4, which job created this relationship), `source` (RelationshipSource, default `pipeline`), `created_at` (TIMESTAMPTZ). Composite unique constraint on `(source_id, target_id, rel_type)`. Index on `target_id` for inbound traversal.

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

> **Future hardening:** once provider requirements are stable, replace the flat `SecretsConfig` with a Pydantic v2 discriminated union (`EnvSecretsConfig | AWSSecretsConfig` on the `provider` discriminator; Azure and Vault variants added when needed). Each provider gets its own model with only the fields that make sense for it — invalid combinations become impossible at startup. The same pattern can be applied to `_LLMConfig` and `TranscriptionConfig` for the same reason.

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

**`ops.api_keys`** — one row per issued API key. Fields: `id` (PK, serial integer), `key_hash` (bcrypt), `user_id`, `role` (`viewer | reviewer | operator | admin`), `created_at`, `revoked_at` (NULL while active). The PK changed from `key_hash` to `id` (migration 006) to enable stable foreign-key referencing and revocation by integer ID.

**`ops.jobs`** — authoritative job state. Fields: `job_id` (PK, UUID4), `user_id` (FK → api_keys.user_id), `status` (mirrors `JobStatus`), `idempotency_key` (UNIQUE nullable), `source_type`, `created_at`, `updated_at`, `finished_at` (TIMESTAMP, set when status transitions to `DONE` or `FAILED`; cleared on `reset_failed_job`), `error_payload` (JSONB, null until FAILED), `mlflow_run_id` (null while PENDING), `meeting_date` (DATE NOT NULL), `submission` (JSONB NOT NULL — the full `JobSubmissionRequest` JSON), `raw_blob_key` (TEXT NOT NULL — S3 key of the raw uploaded file). Index on `(user_id, created_at)` for the per-user rate-limit query. The three `meeting_date`/`submission`/`raw_blob_key` columns are written atomically by `OpsRepository.set_job_submission()` immediately after the raw file is stored — they power both `GET /jobs/{id}/results` blob fallback and `POST /jobs/{id}/retry` without additional DB calls.

`ops.jobs` is the authoritative source for API job lifecycle state. Idempotency key deduplication on `POST /jobs` is a single `SELECT` against the `UNIQUE` constraint on `idempotency_key`. KB nodes and relationships live in `knowledge_base.kb_nodes` and `knowledge_base.kb_relationships` — see §6, MVP: PostgresKBStore. The `store` schema is entirely managed by `langchain-postgres` — Seshat code never writes DDL against it directly.

> **Deferred:** `ops.init_runs` (coordination table for `seshat init`) is not implemented — the `seshat init` feature is deferred to v2.

**Roles:**

| Role | Allowed actions |
|------|----------------|
| `viewer` | `GET /v1/jobs`, `GET /v1/jobs/{id}`, `GET /v1/jobs/{id}/results`, `GET /v1/jobs/{id}/transcript/excerpt`, `GET /v1/graph`, `GET /v1/graph/search`, `GET /v1/graph/{node_id}`, `GET /v1/graph/{node_id}/neighbours`, `GET /v1/graph/{node_id}/detail`, `GET /v1/graph/{node_id}/impact`, `GET /v1/graph/relationships`, `GET /v1/health`, `GET /v1/health/components`, `GET /v1/me` |
| `reviewer` | All `viewer` actions + `POST /v1/jobs` (without `overrides` or `force`), `POST /v1/jobs/{id}/approve` |
| `operator` | All `reviewer` actions + `POST /v1/jobs/{id}/retry`, `POST /v1/jobs` with `overrides`, `POST /v1/graph/nodes`, `POST /v1/graph/nodes/bulk`, `POST /v1/graph/nodes/resolve`, `PUT /v1/graph/nodes/{node_id}`, `PUT /v1/graph/nodes/{node_id}/override`, `POST /v1/graph/relationships` |
| `admin` | All `operator` actions + `POST /v1/jobs` with `force=True`, `DELETE /v1/graph/nodes/{node_id}`, `DELETE /v1/graph/nodes/bulk`, `DELETE /v1/graph/relationships/{rel_id}` |

Key provisioning is via `POST /v1/admin/api-keys` (root-key authenticated), which returns the plaintext key once. The `/v1/admin` router also exposes `GET /v1/admin/api-keys` (list with revocation status) and `DELETE /v1/admin/api-keys/{key_id}` (revoke). The root key itself is stored in Secrets Manager under `APIConfig.root_api_key_secret_key` (default: `"root-api-key"`).

> **JWT (deferred to v2):** JWT with an external IdP (e.g. Azure AD) is the natural upgrade when the user base grows or SSO is needed. See Deferred Decisions.

### Endpoints

All routes are under the `/v1` prefix.

```
GET  /v1/jobs                               List jobs; optional filters (viewer+)
POST /v1/jobs                               Submit a new job (audio file or text); see Job Submission below (reviewer+)
GET  /v1/jobs/{id}                          Job status + timestamps; see Job Progress Contract below (viewer+)
GET  /v1/jobs/{id}/results                  ExtractionResult; available from AWAITING_REVIEW onwards; HTTP 409 if not ready (viewer+)
GET  /v1/jobs/{id}/transcript/excerpt       Fetch a character-range excerpt from the raw transcript (viewer+)
POST /v1/jobs/{id}/approve                  Submit node review decisions (reviewer+)
POST /v1/jobs/{id}/retry                    Retry a FAILED job (operator only)

GET  /v1/graph                              Query KB nodes with NodeFilter (viewer+)
GET  /v1/graph/search                       Semantic/keyword/hybrid search over KB nodes (viewer+)
GET  /v1/graph/{node_id}                    Single node (viewer+)
GET  /v1/graph/{node_id}/neighbours         Direct neighbours of a node (viewer+)
GET  /v1/graph/{node_id}/detail             Node + neighbours + relationships (viewer+)
GET  /v1/graph/{node_id}/impact             Traversal from node; see Impact Traversal below (viewer+)
POST /v1/graph/nodes                        Create a manual KB node (operator+)
POST /v1/graph/nodes/bulk                   Bulk-create nodes with on_error semantics (operator+)
POST /v1/graph/nodes/resolve                Run resolution for manually-created APPROVED nodes (operator+)
PUT  /v1/graph/nodes/{node_id}              Update a manually-created node (operator+)
PUT  /v1/graph/nodes/{node_id}/override     Override any node regardless of ingestion source (operator+)
DELETE /v1/graph/nodes/{node_id}            Delete a node, cascade by default (admin only)
DELETE /v1/graph/nodes/bulk                 Bulk-delete nodes with on_error semantics (admin only)
GET  /v1/graph/relationships                List relationships with optional filters (viewer+)
POST /v1/graph/relationships                Create a manual relationship (operator+)
DELETE /v1/graph/relationships/{rel_id}     Delete a relationship (admin only)

GET  /v1/health                             API liveness; always returns {status: "ok"}
GET  /v1/health/components                  Component readiness; checks postgres, mlflow, blob_store; returns 503 when degraded
GET  /v1/me                                 Current user identity and role (viewer+)

GET  /v1/admin/api-keys                     List all API keys with revocation status (root-key only)
POST /v1/admin/api-keys                     Create a new API key; returns plaintext once (root-key only)
DELETE /v1/admin/api-keys/{key_id}          Revoke an API key by integer ID (root-key only)
```

`GET /v1/graph/search` query parameters: `q: str` (required), `NodeFilter` fields (optional), `limit: int = 10`, `search_mode: SearchMode = SEMANTIC`, `score_threshold: float | None`. Returns `NodeSearchResponse` (list of `NodeSearchResult`, each with a `NodeDetailResponse` and optional `score`).

`GET /v1/graph` filter fields from `NodeFilter` are passed as query params: `?node_type=adr&team=platform&min_confidence=0.7&ingestion_source=job&job_id=abc&limit=100&offset=0`. All fields are optional and AND-combined. Enum fields accept the lowercased `StrEnum` value. `limit` defaults to 1000 (max 10 000); `offset` enables pagination.

### Job Submission

`POST /jobs` is a `multipart/form-data` request with two parts:
- **`file`** — the audio or text payload (required for all `source_type` values; the text variant carries the caller's pre-formatted YAML/JSON file). The file must have a file extension (e.g. `.yaml`, `.mp3`) — requests with an extensionless filename are rejected with HTTP 400.
- **`body`** — a `JobSubmissionRequest` JSON document (form field named `body`, not `request`):

```python
class JobSubmissionRequest(BaseModel):
    source_type: Literal["audio", "text"]         # "video" deferred to v2 (ffmpeg dependency)
    metadata: TranscriptMetadata                  # meeting_date, participants, etc.
    auto_mode: bool = False                       # shorthand: operator sets True to skip human review for this job
    idempotency_key: str | None = None
    force: bool = False                           # re-ingest even if content hash already exists (admin only)
    overrides: SeshatConfigOverride | None = None  # per-request config, deep-merged onto SeshatConfig
    retrieval_filters: NodeFilter | None = None    # runtime RAG retrieval scope; not config (see Section 3)
```

The API constructs a `TranscriptDocument` from this request (generating `id`; `blob_key` is set after the transcript is written to blob storage by the transcription stage for audio, or by the text validator for `source_type="text"`) and enqueues the job.

**Content-hash deduplication:** before creating a new job, `POST /jobs` SHA-256 hashes the uploaded file bytes. If a prior `DONE` job has the same content hash, the request is rejected as a duplicate (HTTP 409) unless `force=True` is set (admin only). When `force=True` is set, the existing job's approved nodes are hard-deleted and the job is re-ingested from scratch.

**Rate limiting:** `POST /jobs` enforces two additional checks before creating a job:

1. **Per-user hourly cap:** counts the user's jobs submitted in the last hour using `ops.jobs` (sliding window: current UTC time − 3600 seconds, indexed on `user_id, created_at`). If the count meets or exceeds `api.max_jobs_per_user_per_hour` (default: 10), the request is rejected with HTTP 429.
2. **Global concurrency cap:** counts jobs system-wide in `TRANSCRIBING`, `IDENTIFYING`, `RESOLVING`, or `WRITING` state. If the count meets or exceeds `api.max_concurrent_jobs` (default: 1), the request is rejected with HTTP 429 and a message indicating a job is already in progress. This prevents LLM cost blowup from simultaneous pipeline runs at MVP scale.

Both checks run before the job is created. Violations are logged with `user_id` and timestamp.

### Job Status Model

```python
class JobStatus(StrEnum):
    PENDING = auto()
    TRANSCRIBING = auto()
    IDENTIFYING = auto()       # renamed from EXTRACTING; covers the identification pass
    AWAITING_REVIEW = auto()   # pipeline pauses here for human review
    RESOLVING = auto()         # resolution pass running after approval
    WRITING = auto()
    DONE = auto()
    FAILED = auto()
```

`AWAITING_REVIEW` is skipped entirely when `auto_mode=True`. `RESOLVING` is the post-approval resolution pass; in auto-mode the pipeline transitions directly from `IDENTIFYING` to `RESOLVING`.

### Job Progress Contract

`GET /jobs/{id}` and `GET /jobs` (list) return the following shape:

```python
class JobResponse(BaseModel):
    job_id: str
    status: JobStatus
    created_at: datetime
    updated_at: datetime
    finished_at: datetime | None        # set when status transitions to DONE or FAILED; None otherwise
    idempotency_key: str | None         # echoed from the original JobSubmissionRequest; None when not provided
    stage_progress: str | None          # human-readable; None when no meaningful message available
    error: ErrorPayload | None          # populated when status=FAILED
    mlflow_run_id: str | None = None           # set when the worker starts the first pipeline stage; NULL while PENDING
    confidence_threshold: float | None = None  # the effective threshold used for this job
```

**`stage_progress` examples per stage:**
- `TRANSCRIBING` — `"Transcribing audio"` (AssemblyAI does not expose per-call progress in the polling API)
- `IDENTIFYING` — `"Extracting: {n}/{total} agents complete"`. `n` counts completed individual agent calls (one per concept_type). `total` is `len(concept_types)` — e.g. 4 concept types = 4 total; progress increments per completed call. When chunking is introduced, `total` becomes `len(concept_types) × chunk_count`.
- `AWAITING_REVIEW` — `"{n} nodes pending review"`
- `RESOLVING` — `"Resolving relationships"`
- `WRITING` — `"Writing nodes to KB"`

> **v2 — Server-Sent Events:** a `GET /jobs/{id}/stream` SSE endpoint is the natural upgrade for lower-latency progress. Deferred until the pipeline stage model is stable and the Streamlit UI is mature enough to warrant it.

**`auto_mode` authorization and audit:** setting `auto_mode=True` requires the `operator` role — the `Depends` function rejects it for `viewer` and `reviewer`. Every job run with `auto_mode=True` is logged in MLflow with: requesting `user_id`, timestamp, job ID, and the full list of nodes that were auto-approved without human review. Auto-approved nodes have `approval_method=ApprovalMethod.AUTO`, `approved_by=user_id`, and `approved_at` set to the job submission timestamp.

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
    stage: str                                    # which stage failed (human-readable label)
    reason: str                                   # human-readable description
    recoverable: bool                             # if True, POST /jobs/{id}/retry is available
    status: JobStatus                             # job status at time of failure
    usage: dict[str, list[UsageRecord]] = {}      # stage → [records] for this attempt
```

**Usage tracking:** usage is tracked per stage and per attempt as a list of `UsageRecord` — one record per `CallType` active in that stage. `call_type` determines what `units` means (tokens for LLM/embedding, audio seconds for transcription). Accumulated usage across all retries is reported in MLflow. Cost estimation from unit counts is informational only (shown in the MLflow UI) — enforcement uses token/duration caps, not dollar amounts, so no price table needs to be maintained.

**Per-stage usage caps:** each pipeline stage enforces aggregate usage limits in its config:
- `ExtractionConfig.max_total_input_tokens` / `max_total_output_tokens` — checked **after each individual agent call**. Before dispatching the next call, the worker sums cumulative tokens for the stage and aborts if either cap is exceeded. This prevents a job overshooting by more than one agent call's worth of tokens.
- `RAGConfig.max_embedding_tokens` — checked after each embedding call in the RAG stage. Up to 200 embedding calls per job (50 chunks × 4 concept types); the cap covers all embedding calls in the stage and is tracked via `CallType.EMBEDDING` in `UsageRecord`.
- `TranscriptionConfig.max_audio_seconds` — checked before invoking the transcription service. If `duration.total_seconds() > max_audio_seconds`, the job transitions to `FAILED` immediately with `recoverable=False` (audio duration is a property of the input — retrying will not help).

If a cap is exceeded, the job transitions to `FAILED` with `recoverable=True`. The `POST /jobs/{id}/retry` endpoint takes no body — to raise a cap, update the config and retry. The UI surfaces a retry button when `recoverable=True`.

**Automatic retry policy:** transient errors (API timeout, HTTP 429 rate limit) are retried automatically before the stage transitions to `FAILED`. Per-stage policy:
- **Max attempts:** 3 (configurable as `TranscriptionConfig.max_retries` and `_LLMConfig.max_retries` — inherited by `IdentificationLLMConfig`, `GroundingLLMConfig`, `ResolutionLLMConfig`)
- **Backoff:** exponential with jitter — base 2s, multiplier 2×, max 60s. When a `Retry-After` header is present, it sets the minimum delay floor; jitter is applied on top: `delay = max(computed_backoff, retry_after_seconds) * uniform(0.8, 1.2)`
- **Scope:** per-call retry within the stage. A stage that exhausts retries on any single call transitions to `FAILED` with `recoverable=True`

**Recoverable failures:**
- Transcription transient error (API timeout, rate limit) — retried automatically; if exhausted, raw input file in blob store; user re-submits via `POST /jobs` with original `idempotency_key`, or operator retries via `POST /jobs/{id}/retry`
- Extraction failure (LLM timeout, rate limit) — retried automatically per agent call; if exhausted, raw input in blob store; user re-submits via `POST /jobs` with original `idempotency_key`, or operator retries via `POST /jobs/{id}/retry`
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
| `POST /jobs` with original `idempotency_key` | `reviewer`+ | New job ID | Normal client retry — the Streamlit retry button uses this path |
| `POST /jobs/{id}/retry` | `operator` only | Same job ID | Operator-initiated recovery where job ID continuity matters (e.g. audit trail, config change before re-run) |

The Streamlit UI retry button always uses `POST /jobs` with the original `idempotency_key` — it creates a new job ID. `POST /jobs/{id}/retry` is an operator tool not surfaced in the UI. `POST /jobs/{id}/retry` reads the raw input file and original submission from `ops.jobs` (`raw_blob_key` + `submission` columns) — no re-upload needed; returns HTTP 409 if those fields are missing (jobs submitted before migration 004).

### Review Flow

`PENDING_REVIEW` nodes have no MVP SLA — a job stays in `AWAITING_REVIEW` until a reviewer acts on it. Timeout-based auto-rejection is deferred to v2 (see Decisions Deferred).

`POST /jobs/{id}/approve` accepts an `ApproveRequest`:

```python
class KBNodeEdit(BaseModel):
    title: str        # required; min_length=1
    description: str  # required; min_length=1

class NodeDecision(BaseModel):
    node_id: str
    action: Literal["approve", "reject"]
    edited_content: KBNodeEdit | None = None   # optional: user may edit before approving
    reason: str | None = None                  # optional: human explanation for rejection or override

class BulkApproveRule(BaseModel):
    threshold: float                           # approve all PENDING_REVIEW nodes with confidence >= threshold
    exclude: list[str] | None = None           # skip these node_ids even if confidence >= threshold

class ManualNodeCreate(BaseModel):
    type: ConceptType
    title: str
    description: str
    source_quote: str | None = None        # verbatim excerpt from the transcript
    blob_key: str | None = None            # blob storage key of the transcript; required if source_quote is provided
    participants: list[str] | None = None
    team: str | None = None
    project: str | None = None
    domain: str | None = None
    meeting_date: date | None = None       # meeting this node was extracted from
    concept_fields: dict[str, Any] | None = None   # type-specific fields (assignee, due, etc.)
    relationships: list[RelationshipInput] | None = None   # optional initial relationships to write alongside the node

    # model_validator: source_quote and blob_key are co-required —
    # providing one without the other is a validation error

class ApproveRequest(BaseModel):
    approve_above_threshold: BulkApproveRule | None = None
    decisions: list[NodeDecision] | None = None
    # Model validator: at least one of approve_above_threshold or decisions must be set
```

**Processing order:**
1. `approve_above_threshold` runs first — approves all `PENDING_REVIEW` nodes with `confidence >= threshold`, skipping any `exclude`d IDs. Sets `approval_method=ApprovalMethod.BULK` on affected nodes.
2. `decisions` runs next — per-node overrides. Can approve, reject, or edit any node regardless of whether the bulk rule already touched it. Sets `approval_method=ApprovalMethod.INDIVIDUAL` on affected nodes. `KBNodeEdit` patches only `title` and `description` — other node fields are not editable at review time.
3. Job transitions to `RESOLVING`; resolution pass runs.
4. **WRITING** — KB + vector store writes.

In steps 1–2, `approved_by` is set to the requesting user's `user_id` and `approved_at` to the current UTC timestamp.

**Role enforcement summary:**

| Field | Minimum role |
|---|---|
| `approve_above_threshold` | `reviewer` |
| `decisions` | `reviewer` |

Manual node creation during review has been moved to `POST /v1/graph/nodes` (operator+), separate from the approval flow.

Once all `pending_review` nodes have a decision, the pipeline resumes to `WRITING`. If all nodes were rejected (zero approved nodes), the pipeline still transitions to `WRITING` — the writing stage writes zero nodes and the job reaches `DONE` with an empty result. This is valid: a reviewer may legitimately reject all extracted nodes if the meeting produced no recordable decisions. The `DONE` response body will contain an empty `nodes` list; no special terminal state is introduced.

### Streamlit UI Scope

The Streamlit app is the primary interface for human operators. Four screens, defined at the functional level — pixel-level design is out of scope for this spec.

**Screen 1 — Job submission**
- File upload field (audio: `.mp3`, `.wav`, `.m4a`; text: `.yaml`, `.json`)
- Metadata form: `meeting_date` (required), `participants` (optional, comma-separated), `language` (default `en`)
- Optional: per-request overrides (`confidence_threshold`, `auto_mode` — operator role only)
- Submit button → polls `GET /jobs/{id}` and transitions to Screen 2

**Screen 2 — Job status**
- Pipeline stage progress bar: `PENDING → TRANSCRIBING → IDENTIFYING → AWAITING_REVIEW → RESOLVING → WRITING → DONE`
- `stage_progress` text (e.g. `"Extracting: 3/8 agents complete"`)
- Elapsed time
- "View in MLflow" deep-link button (visible once `mlflow_run_id` is set on `JobResponse`)
- On `FAILED`: error reason + retry button (visible when `recoverable=True`)
- On `AWAITING_REVIEW`: "Review nodes" button → transitions to Screen 3

**Screen 3 — Node review**

This is the primary correctness gate for the knowledge base. For each `PENDING_REVIEW` node the UI must surface:

- Node title, type (`ConceptType`), and description
- source quote inline (the transcript excerpt grounding this node, resolved from `quote_anchors`)
- `ConfidenceBreakdown`: final score + per-component breakdown (verification / heuristics)
- Per-node action: approve / reject / edit (opens an inline form pre-filled with current `title` + `description`; also exposes optional metadata fields `participants`, `team`, `project`, `domain`)
- Bulk approve rule: threshold slider + optional exclude list → maps to `BulkApproveRule` in `ApproveRequest`
- **Manual node creation panel** (operator role only — hidden for `reviewer`): form with `type` dropdown (required), `title` (required), `description` (required), optional `source_quote` + `blob_key` (co-required — UI disables "Add node" if exactly one is filled), optional `participants`, `team`, `project`, `domain`. "Add node" submits via `POST /v1/graph/nodes` as a separate API call (not via `ApproveRequest`).
- Submit decisions button → `POST /jobs/{id}/approve`

**Screen 4 — Knowledge base query**
- Filter panel: `ConceptType`, `team`, `project`, `domain`, `NodeState`, date range
- Node list: title, type, confidence, state (`CURRENT` / `AMENDED` / `SUPERSEDED`)
- Node detail panel: full node + its relationships from `ops.kb_relationships` (source node + type + target node for each); CONFLICTS relationships highlighted; stale CONFLICTS edges (where either party has `state != CURRENT`) are filtered out and not displayed
- "View in MLflow" deep-link from node metadata

### Impact Traversal

```
GET /graph/{node_id}/impact?depth=2&rel_types=supersedes,depends_on&min_confidence=0.0&direction=outbound
```

Returns `list[KBNode]` with each node annotated with `traversal_depth: int`. Query parameters:

- `depth` — BFS traversal depth (default 2, max 3; unbounded traversal on a dense KB is a cost risk)
- `rel_types` — comma-separated `RelationshipType` values to follow during traversal; default: all types
- `min_confidence` — filters traversed nodes by `KBNode.confidence >= min_confidence`; default 0.0
- `direction` — `GraphDirection` enum value (`outbound` | `inbound` | `both`); default `outbound`

Role requirement: `viewer` (read-only, same as `GET /graph/{node_id}`).

`get_neighbours()` is called with the given `direction` at each BFS level. `traversal_depth` on each returned node is the hop count from the seed node. Example with `direction=inbound`: if Decision-A `DEPENDS_ON` Decision-B, traversing inbound from Decision-B returns Decision-A at depth=1 — the node that would break if Decision-B changes.

#### `POST /graph/nodes/resolve` — trigger resolution for manually-created nodes

```
POST /graph/nodes/resolve
```

Runs relationship resolution for a set of manually-created KB nodes that have already been approved but never had their relationships resolved (because they were created outside the extraction pipeline).

Request body: `ResolveRequest`:

```python
class ResolveRequest(BaseModel):
    node_ids: list[UUID] = Field(..., min_length=1, max_length=50)
```

Response: `ResolveResponse`:

```python
class ResolveResponse(BaseModel):
    relationships_created: list[KBRelationship]
```

**Validation:**
- 404 if any `node_id` is not found in the KB.
- 422 if any node is not in `APPROVED` status.

**Processing:** delegates to `GraphService.resolve_by_ids()`, which calls `ExtractionOrchestrator.run_resolution(job_id=f"manual_resolve_{uuid4()}", approved=nodes)` and persists the resulting `KBRelationship` objects via `NodeRepository.write_relationship()`.

Role requirement: `operator`.

### Reviewer Data Contract

`GET /jobs/{id}/results` returns `ExtractionResult`. The result is served from the in-memory job state (owned by `JobService`) while the server is running. After a server restart, the endpoint falls back to the curated extraction blob (`blob_store.curated_extraction_key(meeting_date, job_id)`) written by `ExtractionOrchestrator` at the start of the WRITING stage — this blob is the durable source of truth for completed jobs. Returns HTTP 409 if the job is not yet in a reviewable or completed state; HTTP 404 if neither in-memory result nor blob is available.

The Streamlit review screen is the only consumer of this endpoint for MVP — the data contract between the API and the UI is:

| Field | Source | Used by |
|---|---|---|
| `nodes` | `ExtractionResult.nodes` | All node fields in Screen 3 |
| `confidence_breakdowns[node_id]` | `ConfidenceBreakdown` per node | Confidence breakdown display |
| `nodes[].quote_anchors` | `KBNode.quote_anchors` | Inline source quote (resolved from anchors) |

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

**`setup_mlflow(config: ObservabilityConfig) -> str`** — called once at worker startup. Sets the tracking URI, enables `mlflow.langchain.autolog()`, resolves the experiment by name, and returns the experiment ID. The caller is responsible for caching the returned experiment ID in-process.

**`mlflow_run_url(tracking_uri, experiment_id, run_id) -> str`** — helper that builds the deep-link URL (`{tracking_uri}/#/experiments/{experiment_id}/runs/{run_id}`). Defined in `src/seshat/app/platform/observability/mlflow_setup.py`.

**Token budget tracking** — `src/seshat/app/platform/observability/usage_tracker.py` provides `UsageTracker`, `TokenBudgetCallback`, `TrackingEmbeddings`, `TrackingTranscriber`, and the `@track_token_budget` decorator. The decorator is wired into orchestrator methods and tracks token spend per pipeline stage. `set_run_tracker` / `get_run_tracker` manage the per-run callback via a context variable. `UsageTracker` tracks: `input_tokens`, `output_tokens`, `cache_read_tokens`, `cache_write_tokens`, `embedding_tokens`, `transcription_seconds`, and `reranker_input_tokens` — the last accumulated by `CohereReranker` (from `response.meta.tokens.input_tokens`) and `VoyageReranker` (from `response.total_tokens`) via `get_run_tracker()`.

**Streamlit integration:**
- MVP: "View in MLflow" button deep-links to `{mlflow_tracking_uri}/#/experiments/{experiment_id}/runs/{run_id}`. `mlflow_run_id` comes from `JobResponse`; `experiment_id` is the return value of `setup_mlflow()`, cached at worker startup — not exposed on the API.
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
1. Check for active jobs in `TRANSCRIBING`, `IDENTIFYING`, `RESOLVING`, or `WRITING` state.
2. Wait for them to complete or fail, or accept the loss — `recoverable=True` on most failure modes means users can retry via `POST /jobs/{id}/retry`.
3. Jobs in `AWAITING_REVIEW` are safe to deploy through — the worker is idle for those jobs and no in-memory state is held.

Jobs stranded in `WRITING` are handled automatically — the worker boot recovery procedure (see Section 8, Worker boot recovery) detects them on startup and marks them `FAILED` with `recoverable=True`. Because KB and vector writes are a single Postgres transaction, no partial KB state can exist — only fully committed or fully rolled-back writes are possible, so no cleanup is needed.

**Rollback:** redeploy the previous image tag:
```bash
docker compose up -d --no-deps api worker
```
with the previous SHA tag set in `.env` or the Compose file. KB data and MLflow artifacts are on named volumes and are unaffected by image rollbacks.

---

## 11. Project Structure

```
src/seshat/
├── core/                        # Pure data and config — no I/O, no AI
│   ├── models/                  # Pydantic domain models (KBNode, enums, …)
│   ├── config/                  # Pydantic settings (SeshatConfig, LLMConfig, ExtractionConfig, …)
│   └── utils/                   # Shared pure utilities (audio, retry, tokens, logging)
├── infra/                       # External system adapters — I/O only, no business logic
│   ├── blob_store/              # S3 blob store abstraction (aiobotocore)
│   ├── vector_store/            # pgvector semantic search abstraction
│   ├── knowledge_store/         # Postgres-backed KB node persistence
│   ├── ops_store/               # Postgres-backed job/ops ledger
│   └── secrets/                 # AWS Secrets Manager helpers
├── app/                         # Runtime application — orchestration, AI, and services
│   ├── agents/                  # LLM agents
│   │   ├── identification/      # Extraction agents (grouping, registry)
│   │   └── resolution/          # Resolution agents (same_type, cross_type)
│   ├── transcription/           # Transcriber interface and provider implementations
│   ├── pipeline/                # Orchestration
│   │   ├── extraction/          # Extraction sub-pipeline (identification, scoring, resolution)
│   │   └── ingestion/           # Ingestion sub-pipeline (audio/text validation, blob upload)
│   ├── repositories/            # NodeRepository and ops/blob repository facades
│   ├── services/                # Domain services (GraphService, JobService, AdminService, …)
│   └── platform/                # Deployment-layer concerns
│       ├── api/                 # FastAPI routers, auth, app state, startup
│       ├── worker/              # Async task queue and job worker
│       └── observability/       # MLflow tracing, usage tracking, latency metrics
├── eval/                        # Eval harnesses and calibration meta-scorers (tooling, not runtime)
└── cli/                         # CLI entry points (seshat eval, seshat api, seshat migrate; seshat init deferred to v2)

tests/
  unit/          # Fast unit tests — mirrors src/seshat/ hierarchy
  integration/   # Slow tests requiring Postgres, LocalStack, or LLM APIs
scripts/         # Operational scripts; not part of the application package
data/            # Local KB, MLflow artifacts (gitignored)
eval_gate.json   # Written by seshat eval; read by worker at startup; gitignored
```

---

## 12. Evaluation Strategy

Evaluation is an MVP requirement — the `confidence_threshold=0.7` already in production config must have a calibration basis before any real data is processed.

### Three Eval Passes

The eval harness covers three pipeline stages independently:

| Pass | Runner | What it measures |
|------|--------|-----------------|
| **Identification** | `IdentificationEvalRunner` | Precision/recall of node extraction per `ConceptType`, plus field-level accuracy |
| **Resolution** | `ResolutionEvalRunner` | Precision/recall of inferred `KBRelationship`s |
| **Retrieval** | `RetrievalEvalRunner` | recall@5 for vector search candidate surfacing |

All three runners are in `src/seshat/eval/` and are exported from `seshat.eval`. Each is an independent library class — no CLI entrypoint exists yet; eval is invoked programmatically or from integration tests.

### Labelled Corpus

Each pass has its own corpus directory under `data/eval/corpus/`:

```
data/eval/
  corpus/
    identification/   # one YAML per transcript
    resolution/       # one YAML per scenario
    retrieval/        # one YAML per query
  test_corpus/
    identification/   # small subset for integration tests
    resolution/
    retrieval/
```

Corpus files are **not** in `tests/`. They live in `data/eval/` because they are runtime artifacts that feed live LLM/embedding calls — not static test fixtures.

**Identification corpus** (one YAML per transcript):

```yaml
corpus_id: "example-001"
transcript: "<transcript text>"
expected_nodes:
  - quote: "<ground-truth span>"
    type: decision
    title: "<title>"
    description: "<description>"
    extra_fields:               # optional type-specific fields for field-accuracy scoring
      rationale: "..."
      assignee: "Priya"
```

`IdentificationCorpusNode` carries `extra_fields: dict[str, Any]` for field-level accuracy scoring (see §12 quality scoring below). `expected_nodes` is a list of `IdentificationCorpusNode`, not `KBNode`.

**Resolution corpus** (one YAML per scenario):

```yaml
corpus_id: "clickhouse-vs-postgres"
description: "ClickHouse scopes an exception to the PostgreSQL standard — AMENDS not SUPERSEDES"
source_nodes:
  - id: clickhouse-analytics   # human-readable slug
    type: decision
    title: "..."
    description: "..."
    quote: "..."
kb_nodes:
  - id: postgres-all
    type: decision
    ...
expected_relations:
  - source: clickhouse-analytics
    target: postgres-all
    rel_type: AMENDS
# pairs not listed are implicitly UNRELATED
```

**Retrieval corpus** (one YAML per query):

```yaml
corpus_id: "action-surfaces-risk"
description: "Action item about deadline should surface related risk in top-5"
query_node:
  id: delay-deadline
  type: action_item
  ...
candidate_nodes:
  - id: scope-creep-risk
    ...
expected_relevant_ids:
  - scope-creep-risk
```

**Minimum corpus size** (identification pass): ≥15 annotated instances per `ConceptType` before threshold calibration. Below that, a single mis-annotated example shifts precision or recall by ~7%, making the gate targets meaningless.

**Adversarial transcripts** (add before first real-data run): one transcript with unsupported claims, one with ambiguous pronouncements across `ConceptType`s, one with injected instruction-like text. OOD testing deferred to v2.

### Precision / Recall Targets

Targets live in `src/seshat/eval/thresholds.py` — changing them requires a code review, not a config change.

> **Statistical caveat:** these targets are directional signals, not statistically validated thresholds. At the minimum corpus size, a single instance shifts precision or recall by ~7%. Treat them as a floor for catching gross failures, not a precise calibration.

**Identification** (per `ConceptType`):

| ConceptType | Precision target | Recall target | Notes |
|-------------|-----------------|---------------|-------|
| `DECISION` | ≥ 0.80 | ≥ 0.75 | High-stakes; false positives costly |
| `RISK` | ≥ 0.75 | ≥ 0.80 | Recall-biased — missed risks worse than false positives |
| `OPEN_QUESTION` | ≥ 0.75 | ≥ 0.75 | Moderate difficulty |
| `ACTION_ITEM` | ≥ 0.85 | ≥ 0.85 | Simpler extraction; higher bar |

**Resolution** (global, not per-`RelationshipType` — corpus is too small for per-type breakdowns in MVP):

| Metric | Target |
|--------|--------|
| Precision | ≥ 0.80 |
| Recall | ≥ 0.80 |

Per-`RelationshipType` gate targets are deferred until the corpus is large enough. The scorer redesign needed to emit per-type feedback is also deferred; current global 0.80 thresholds are documented with a TODO in `thresholds.py`.

**Retrieval:**

| Metric | Target |
|--------|--------|
| recall@5 | ≥ 0.70 |

### Gate File

Each eval run writes to a configurable `EvalConfig.gate_path` (typically `data/eval_gate.json`). The file is **gitignored** — it is a local runtime artifact and must be regenerated per environment.

```json
{
  "run_id": "<mlflow_run_id>",
  "timestamp": "<iso8601>",
  "passed": true,
  "identification_metrics": {
    "decision.precision": 0.83,
    "decision.recall": 0.78,
    "decision.f1": 0.80,
    "risk.precision": 0.76,
    "risk.recall": 0.81,
    "open_question.precision": 0.77,
    "open_question.recall": 0.76,
    "action_item.precision": 0.88,
    "action_item.recall": 0.86
  },
  "resolution_metrics": {
    "precision": 0.82,
    "recall": 0.81,
    "f1": 0.81
  },
  "retrieval_metrics": {
    "recall_at_5": 0.82,
    "precision_at_5": 0.60
  }
}
```

`passed` is a `@computed_field` (not stored as a literal field) — it is recomputed when the gate file is loaded. A gate where all three metric blocks are `None` is treated as failed (fail-closed). A `None` block means that pass was not run and is not evaluated; only non-`None` blocks are checked against their targets.

**Upsertable gate:** `upsert_gate(gate_path, run_id, identification_metrics=..., ...)` reads the existing gate file (if present), carries forward any blocks not explicitly supplied, recomputes `passed`, and writes back. This lets the three passes run independently — running only identification does not clear a previously-passed retrieval result.

### EvalConfig

```python
class EvalConfig(BaseSettings):  # inherits from BaseSettings, not BaseConfig
    corpus_base_dir: Path = PROJECT_ROOT / "data/eval/corpora"
    gate_path: Path = DEFAULT_EVAL_GATE_PATH   # validated: must end in .json; parent dir created if missing
    observability: ObservabilityConfig = ObservabilityConfig()
    run_identification: bool = True
    run_resolution: bool = True
    run_retrieval: bool = True
    run_grounding: bool = True
    run_grouping: bool = True
    max_concurrent_predictions: int = 10
    retrieval_score_thresholds: dict[SearchMode, float] = {}  # per-mode; absent = 0.0; set by RetrievalMetaScorer per mode

    # Computed fields (corpus dirs and cache dirs for all five eval passes):
    identification_corpus_dir: Path   # corpus_base_dir / "identification"
    resolution_corpus_dir: Path       # corpus_base_dir / "resolution"
    retrieval_corpus_dir: Path        # corpus_base_dir / "retrieval"
    grounding_corpus_dir: Path        # corpus_base_dir / "grounding"
    grouping_corpus_dir: Path         # corpus_base_dir / "grouping"
    # Cache dirs live under PROJECT_ROOT / ".seshat" / "eval_cache" / <pass>
```

`result_cache_enabled` does **not** exist in `ExtractionConfig` — every eval run makes full LLM calls. Intermediate prediction results are cached via `read_or_run` in `src/seshat/eval/cache.py` (keyed on corpus file hash) to avoid re-running the same example during iteration.

### Quality Scoring (Identification Pass)

Quality scoring layers on top of the precision/recall match result without affecting it. For each matched pair, additional scorers fire independently:

- **Field-level accuracy** (always enabled): compares type-specific structured fields (`assignee`, `due`, `rationale`, `type`, `context`) from `extra_fields` against predicted values using deterministic fuzzy match. Output: `{ctype}.{field}/value` per field.
- **NLI faithfulness** (deferred — `nli_scorer_enabled=False`): would check whether predicted `description` and `title` are entailed by the predicted quote span using a local cross-encoder. Blocked on model download availability.
- **LLM-as-judge** (dropped): asymmetric with resolution/retrieval passes; field accuracy + optional NLI is the appropriate ceiling.

Quality metrics are logged to MLflow for regression visibility. Quality gates are not yet defined — baselines must be established first.

### Threshold Calibration

1. Run the identification eval across a sweep of `confidence_threshold` values (0.5 → 0.9 in 0.05 steps).
2. Plot the precision-recall curve **per `ConceptType`** — a single global curve conflates types with opposing biases.
3. Select per-type optimal thresholds. If a single global value satisfies all targets, use it — otherwise document the trade-off.
4. The default `confidence_threshold=0.7` is the starting point. If per-type thresholds are warranted, configure them via `ExtractionConfig.per_type_thresholds` — `None` means use the global default for all types.

### Evaluation Entrypoint

No `seshat eval` CLI command exists yet — eval is currently invoked programmatically or via integration tests. The `seshat eval` CLI is deferred; it will wrap `run_all()` from a future `eval/run_all.py` orchestrator module.

**Bootstrap note:** eval does not check or require a pre-existing gate file and does not submit jobs through the API worker — it invokes pipeline components directly in-process. This breaks the circular dependency: the gate file can't exist before eval runs, and eval doesn't need it to run.

Each eval run is a versioned MLflow experiment — runs can be compared in the MLflow UI to detect regressions from prompt or model changes.

### Release Gate

No real meeting recordings may be processed until eval has been run and all three conditions are met:
1. `identification_metrics` pass all per-`ConceptType` precision/recall targets
2. `resolution_metrics` precision and recall ≥ 0.80
3. `retrieval_metrics` recall@5 ≥ 0.70

**Enforcement:** on startup, the worker reads the gate file at `EvalConfig.gate_path`. If the file is absent or `passed=False`, the worker refuses to accept jobs and logs a clear error. The check can be bypassed by setting `SESHAT_SKIP_EVAL_GATE=true`.

### Regression Gate

Any change to an agent system prompt, model, or confidence scoring logic must be run through eval before promotion. A change that improves one `ConceptType` at the cost of another is a regression — visible in MLflow as a metric degradation vs the baseline run. A failing run sets `passed=False` (computed on next gate file load) and the worker will refuse to start until a passing run is recorded.

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
| Replace grounding agent with ONNX NLI model | Textual entailment (does source quote support extracted claim?) is well-served by discriminative models (DeBERTa, MiniLM-NLI) via `onnxruntime` — no PyTorch needed. Evaluate once MVP eval data shows whether LLM accuracy is actually needed. |
| Replace same-type resolution with ONNX NLI model + embedding similarity | Node-to-node paraphrase/conflict detection is a constrained classification problem. Reuse retrieval-time embeddings for cosine similarity pre-filtering; ONNX NLI model for classification. Evaluate against LLM baseline using `seshat eval`. |
| `AWAITING_REVIEW` timeout SLA (auto-reject stale pending nodes) | Requires a durable scheduled task (ARQ/Redis). The MVP asyncio queue is in-memory and does not survive worker restarts, so a 72h-ahead timer cannot be reliably fired. Ships together with the ARQ/Redis queue swap. |
| Job state push notifications (webhook callbacks + SSE stream) | MVP has no inbound-HTTP consumer — Streamlit polls `GET /jobs/{id}`. Add `callback_url` + `POST` fan-out when an external consumer (CI, Teams bot, etc.) materialises, and `GET /jobs/{id}/stream` (SSE) when the UI matures enough to want low-latency progress. |
| Post-approval node correction API (`PATCH /graph/{node_id}`) | Reviewers edit at approval time via `ApproveRequest.decisions[].edited_content`; already-approved nodes require a direct Postgres update + vector re-embed on MVP. The endpoint earns its keep once the KB moves to Notion/Neo4j. Ships with `NodeMetadata.last_edited_by` / `last_edited_at` at that point. |
| Reranking (wiring into live pipeline) | `AbstractReranker`, `CohereReranker`, `VoyageReranker`, and `RerankerConfig` are implemented; `build_reranker()` factory exists. However, no reranker is passed to `NodeRetriever` in `bootstrap.py` — the wiring is deferred. Adopt once the retrieval baseline (`seshat eval`) shows top-K-only recall@5 is insufficient. The `Reranker` Protocol in `NodeRetriever` expects `list[SearchResult]`; the concrete `AbstractReranker` takes `list[KBNode]` — interface alignment is required before wiring. |
| Transcript chunking (TextTiling / RecursiveCharacterTextSplitter) | Full transcript passed to agents in MVP — context window (200k tokens) is sufficient. Introduce chunking if per-call cost becomes a concern or transcripts regularly exceed context limits. When introduced, add `max_chunk_count` and `max_transcript_chunk_tokens` to `ExtractionConfig` and extend deduplication with the cosine-similarity fallback. |
