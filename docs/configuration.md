# Seshat Configuration Reference

This document describes every configuration class defined under `src/seshat/core/config/` (`settings.py` and `eval_settings.py`), organized top-to-bottom: each settings root is documented first, followed by its nested config classes as sub-sections, drilling down in the same order fields appear in the code.

Seshat configuration is built on [Pydantic Settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/). There are two independent settings roots, each documented as its own top-level section below: **[`SeshatConfig`](#seshatconfig)** (the runtime application configuration) and **[`EvalConfig`](#evalconfig)** (configuration for the `seshat eval` harnesses). **[`SeshatConfigOverride`](#seshatconfigoverride)** is a third, related root used for per-request overrides.

Env var names are case-insensitive and derived from the field path with `__` as the nesting delimiter, e.g. `extraction.identification.model` → `EXTRACTION__IDENTIFICATION__MODEL`. `SeshatConfig` reads from `.env` with no prefix; `EvalConfig` reads from the same `.env` but requires an `EVAL__` prefix on every variable. Fields typed as `list`, `set`, or `dict` accept either a JSON-encoded value or, for dicts, per-key nested env vars (e.g. `EXTRACTION__PER_TYPE_THRESHOLDS__DECISION=0.6`).

A class name prefixed with `_` (e.g. `_LLMConfig`, `_PostgresStoreConfig`) marks a private base class that is never instantiated directly — its fields are inherited by the concrete subclasses that appear as actual fields (`IdentificationLLMConfig`, `KBStoreConfig`, etc.), or, in a couple of spots, used directly as a field's type.

---

## `SeshatConfig`

Top-level `BaseSettings` root for the runtime application. Loads from `.env` (no prefix) with `__` as the nested delimiter; unknown env vars are ignored.

| Field | Default | Definition |
|---|---|---|
| `logging` | `LoggingConfig()` | See [`logging` → `LoggingConfig`](#logging--loggingconfig). |
| `transcription` | `TranscriptionConfig()` | See [`transcription` → `TranscriptionConfig`](#transcription--transcriptionconfig). |
| `vector_store` | `VectorStoreConfig()` | See [`vector_store` → `VectorStoreConfig`](#vector_store--vectorstoreconfig). |
| `vector_index` | `VectorIndexConfig()` | See [`vector_index` → `VectorIndexConfig`](#vector_index--vectorindexconfig). |
| `kb_store` | `KBStoreConfig()` | See [`kb_store` → `KBStoreConfig`](#kb_store--kbstoreconfig). |
| `ops_store` | `OpsStoreConfig()` | See [`ops_store` → `OpsStoreConfig`](#ops_store--opsstoreconfig). |
| `blob_store` | `BlobStoreConfig()` | See [`blob_store` → `BlobStoreConfig`](#blob_store--blobstoreconfig). |
| `extraction` | `ExtractionConfig()` | See [`extraction` → `ExtractionConfig`](#extraction--extractionconfig). |
| `rag` | `RAGConfig()` | See [`rag` → `RAGConfig`](#rag--ragconfig). |
| `secrets` | `SecretsConfig()` | See [`secrets` → `SecretsConfig`](#secrets--secretsconfig). |
| `observability` | `ObservabilityConfig()` | See [`observability` → `ObservabilityConfig`](#observability--observabilityconfig). |
| `api` | `APIConfig()` | See [`api` → `APIConfig`](#api--apiconfig). |
| `document_loader` | `None` | See [`document_loader` → `DocumentLoaderConfig`](#document_loader--documentloaderconfig). Optional; only used by `seshat init`. |
| `max_concurrent_init_runs` | `1` | Maximum number of concurrent `seshat init` runs. |

### `logging` → `LoggingConfig`

Configures root and per-library log levels.

| Field | Default | Definition |
|---|---|---|
| `level` | `INFO` | Root log level (`DEBUG`, `INFO`, `WARNING`, `ERROR`). |
| `noisy_loggers` | `{aiobotocore: WARNING, botocore: WARNING, httpx: WARNING, langchain: WARNING, langchain_core: WARNING, langchain_aws: WARNING, langchain_openai: WARNING, mlflow: WARNING, urllib3.connectionpool: ERROR}` | Per-logger level overrides for verbose third-party libraries. |

### `transcription` → `TranscriptionConfig`

Configures the audio transcription provider and its limits.

| Field | Default | Definition |
|---|---|---|
| `provider` | `assemblyai` | `TranscriptionProvider` enum: `assemblyai`, `openai`, `deepgram`. |
| `model` | `None` | Provider-specific model name; `None` uses the provider default. |
| `language` | `en` | BCP-47 language code for the audio being transcribed. |
| `max_file_bytes` | `524_288_000` (500 MiB) | Maximum accepted audio file size, in bytes. |
| `max_audio_seconds` | `7200` | Maximum accepted audio duration, in seconds. |
| `max_retries` | `3` | Maximum retry attempts on transient errors. |
| `timeout_seconds` | `None` | Per-request timeout for transcription calls in seconds; `None` means no limit. |
| `api_key_secret_key` | `None` → `<provider>_api_key` | Secrets key for the transcription provider API key. |

### `vector_store` → `VectorStoreConfig`

Identifies the vector store backend and its connection secret.

| Field | Default | Definition |
|---|---|---|
| `provider` | `pgvector` | `VectorStoreProvider` enum: `pgvector`. |
| `connection_secret_key` | `postgres_url` | Secrets key for the vector store connection string. |

### `vector_index` → `VectorIndexConfig`

Configures the embedding model and indexing limits used to populate the vector store.

| Field | Default | Definition |
|---|---|---|
| `collection` | `seshat-docs` | Name of the collection (table or index) within the vector store. |
| `embedding_provider` | `openai` | `EmbeddingProvider` enum: `openai`, `azure_openai`, `cohere`. |
| `embedding_model` | `text-embedding-3-small` | Embedding model name. |
| `api_key_secret_key` | `None` → `<embedding_provider>_api_key` | Secrets key for the embedding API key. |
| `max_indexing_tokens` | `500_000` | Maximum total tokens that may be embedded in a single RAG indexing run. |

### `kb_store` → `KBStoreConfig`

Postgres-backed knowledge base node store. Extends the private base `_PostgresStoreConfig` (shared with `OpsStoreConfig`), which requires `schema_name` and validates it against `^[a-z_][a-z0-9_]*$`.

| Field | Default | Definition |
|---|---|---|
| `schema_name` | `knowledge_base` | PostgreSQL schema name used by the KB store; must start with a letter or underscore and contain only lowercase letters, digits, and underscores. |
| `pool_min_size` | `2` | Minimum connection pool size. |
| `pool_max_size` | `10` | Maximum connection pool size. |
| `connection_secret_key` | `postgres_url` | Secrets key for the Postgres connection string. |

### `ops_store` → `OpsStoreConfig`

Postgres-backed job/ops ledger store. Extends the same private base `_PostgresStoreConfig` as `KBStoreConfig`.

| Field | Default | Definition |
|---|---|---|
| `schema_name` | `ops` | PostgreSQL schema name used by the Ops store. |
| `pool_min_size` | `2` | Minimum connection pool size. |
| `pool_max_size` | `10` | Maximum connection pool size. |
| `connection_secret_key` | `postgres_url` | Secrets key for the Postgres connection string. |

### `blob_store` → `BlobStoreConfig`

Configures the S3-compatible blob store used for audio/document uploads.

| Field | Default | Definition |
|---|---|---|
| `bucket` | `seshat-mvp` | S3 bucket name. |
| `region` | `eu-west-1` | S3 region. |
| `endpoint_url` | `None` | Optional custom endpoint URL; set for LocalStack or other S3-compatible stores. |

### `extraction` → `ExtractionConfig`

Governs the full extraction sub-pipeline: which concept types to extract, the identification/resolution LLMs, grouping, grounding, auto-approval thresholds, and token/time budgets.

| Field | Default | Definition |
|---|---|---|
| `concept_types` | all `ConceptType` values (`decision`, `risk`, `action_item`, `open_question`) | Concept types that the extraction pipeline will attempt to extract. |
| `identification` | `IdentificationLLMConfig()` | See [`identification` → `IdentificationLLMConfig`](#identification--identificationllmconfig). |
| `identification_self_review` | `ReflectiveLLMConfig()` (disabled) | See [`identification_self_review` → `ReflectiveLLMConfig`](#identification_self_review--reflectivellmconfig). |
| `resolution` | `ResolutionLLMConfig()` | See [`resolution` → `ResolutionLLMConfig`](#resolution--resolutionllmconfig). |
| `resolution_self_review` | `ReflectiveLLMConfig()` (disabled) | Same class as [`identification_self_review`](#identification_self_review--reflectivellmconfig); self-review loop settings for resolution agents. |
| `grouped_identification_types` | `{decision}` | Concept types for which identified items are passed through the grouping step. |
| `grounding` | `None` | See [`grounding` → `GroundingLLMConfig`](#grounding--groundingllmconfig). Optional; `None` disables grounding. |
| `confidence_threshold` | `0.7` | Minimum heuristics score required to auto-approve an identified node. `None` disables threshold auto-approval — all nodes go to `PENDING_REVIEW`. Incompatible with `auto_mode=True`. |
| `per_type_thresholds` | `None` | Optional per-concept-type confidence thresholds that override the global threshold. |
| `auto_mode` | `False` | When `True`, auto-approve extraction results without manual review. |
| `max_total_input_tokens` | `2_000_000` | Hard cap on total input tokens consumed in one extraction run. |
| `max_total_output_tokens` | `500_000` | Hard cap on total output tokens generated in one extraction run. |
| `max_total_embedding_tokens` | `10_000_000` | Hard cap on total embedding input tokens in one extraction run. |
| `max_hint_nodes` | `20` | Maximum number of KB hint nodes injected into the extraction prompt. |
| `max_hint_tokens` | `1000` | Maximum tokens consumed by hint nodes injected into the extraction prompt. |
| `identification_timeout_seconds` | `None` | Optional wall-clock timeout for a full extraction run; `None` means no limit. |
| `resolution_timeout_seconds` | `None` | Optional wall-clock timeout for a full resolution run; `None` means no limit. |

Validated invariants: `grounding.provider` must differ from `identification.provider`; `confidence_threshold=None` cannot be combined with `auto_mode=True`.

#### `identification` → `IdentificationLLMConfig`

Concrete subclass of the private base `_LLMConfig`, used for the identification (extraction) step.

| Field | Default | Definition |
|---|---|---|
| `provider` | `anthropic` | `LLMProvider` enum: `openai`, `anthropic`, `azure_openai`, `bedrock_converse`. |
| `model` | `claude-sonnet-4-6` | Model identifier string passed to the provider. |
| `temperature` | `0.0` | Sampling temperature (`>= 0`). |
| `max_retries` | `3` | Maximum retry attempts on transient errors. |
| `timeout_seconds` | `300.0` | Per-request HTTP timeout in seconds. |
| `max_concurrent_calls` | `5` | Maximum number of simultaneous LLM calls. |
| `max_output_tokens` | `None` | Maximum tokens the LLM may generate per call; `None` means no limit. |
| `api_key_secret_key` | `None` → `anthropic_api_key` | Secrets key for the LLM API key. |

#### `identification_self_review` → `ReflectiveLLMConfig`

Optional extract → validate → filter self-review loop. Also used, with independent settings, as `extraction.resolution_self_review`.

| Field | Default | Definition |
|---|---|---|
| `enabled` | `False` | When `True`, the agent runs an extract → validate → filter pass. |
| `llm` | `None` | See [`llm` → `_LLMConfig`](#llm--_llmconfig). Falls back to the stage's primary LLM when `None`. |

##### `llm` → `_LLMConfig`

Private base class for all LLM-backed configs. Used directly (unsubclassed) as the type of `ReflectiveLLMConfig.llm`, `RAGConfig.keyword_extraction_llm`, and `MultiQueryConfig.llm`; also inherited by the concrete subclasses `IdentificationLLMConfig`, `GroundingLLMConfig`, and `ResolutionLLMConfig` documented elsewhere in this doc. `provider` and `model` have no default here — only subclasses supply them, so a raw `_LLMConfig` field must set both explicitly when enabled.

| Field | Default | Definition |
|---|---|---|
| `provider` | *(required)* | `LLMProvider` enum: `openai`, `anthropic`, `azure_openai`, `bedrock_converse`. |
| `model` | *(required)* | Model identifier string passed to the provider. |
| `temperature` | `0.0` | Sampling temperature (`>= 0`). |
| `max_retries` | `3` | Maximum retry attempts on transient errors. |
| `timeout_seconds` | `300.0` | Per-request HTTP timeout in seconds. |
| `max_concurrent_calls` | `5` | Maximum number of simultaneous LLM calls. |
| `max_output_tokens` | `None` | Maximum tokens the LLM may generate per call; `None` means no limit. |
| `api_key_secret_key` | `None` → `<provider>_api_key` | Secrets key for the LLM API key. |

#### `resolution` → `ResolutionLLMConfig`

Concrete subclass of `_LLMConfig`, used for the resolution step (inferring relationships between nodes).

| Field | Default | Definition |
|---|---|---|
| `provider` | `anthropic` | `LLMProvider` enum: `openai`, `anthropic`, `azure_openai`, `bedrock_converse`. |
| `model` | `claude-sonnet-4-6` | Model identifier string passed to the provider. |
| `temperature` | `0.0` | Sampling temperature (`>= 0`). |
| `max_retries` | `3` | Maximum retry attempts on transient errors. |
| `timeout_seconds` | `300.0` | Per-request HTTP timeout in seconds. |
| `max_concurrent_calls` | `10` | Maximum simultaneous LLM calls per resolution agent (overrides the `_LLMConfig` default of `5`). |
| `max_global_calls` | `30` | Global cap on simultaneous LLM calls across all resolution agents. |
| `max_output_tokens` | `None` | Maximum tokens the LLM may generate per call; `None` means no limit. |
| `api_key_secret_key` | `None` → `anthropic_api_key` | Secrets key for the LLM API key. |

#### `grounding` → `GroundingLLMConfig`

Concrete subclass of `_LLMConfig`, used for the optional grounding pass, which double-checks that identified content is grounded in the source transcript.

| Field | Default | Definition |
|---|---|---|
| `provider` | `openai` | `LLMProvider` enum: `openai`, `anthropic`, `azure_openai`, `bedrock_converse`. |
| `model` | `gpt-5.4-nano` | Model identifier string passed to the provider. |
| `use_full_transcript` | `True` | When `False`, grounding uses only the identified quote instead of the full transcript. |
| `temperature` | `0.0` | Sampling temperature (`>= 0`). |
| `max_retries` | `3` | Maximum retry attempts on transient errors. |
| `timeout_seconds` | `300.0` | Per-request HTTP timeout in seconds. |
| `max_concurrent_calls` | `5` | Maximum number of simultaneous LLM calls. |
| `max_output_tokens` | `None` | Maximum tokens the LLM may generate per call; `None` means no limit. |
| `api_key_secret_key` | `None` → `openai_api_key` | Secrets key for the LLM API key. |

### `rag` → `RAGConfig`

Configures retrieval-augmented generation: search strategy, similarity/context limits, graph traversal, and the optional multi-query and reranking add-ons.

| Field | Default | Definition |
|---|---|---|
| `enabled` | `True` | Whether RAG retrieval is enabled. |
| `top_k` | `5` | Number of top results to retrieve. |
| `min_similarity_score` | `0.5` | Minimum similarity score `[0, 1]` to retain a retrieved result. |
| `max_context_tokens` | `4000` | Maximum tokens the retrieved context may occupy in the prompt. |
| `traversal_max_depth` | `1` | Maximum graph-traversal depth when expanding retrieved nodes via relationships. |
| `traversal_rel_types` | `None` | Relationship types to follow during traversal; `None` means all. |
| `max_concurrent_retrievals` | `20` | Maximum number of simultaneous RAG retrieval calls. |
| `search_mode` | `semantic` | `SearchMode` enum: `semantic` (pgvector ANN), `keyword` (GIN tsvector full-text), `hybrid` (RRF fusion of both), `agent` (reserved). Can be toggled per-request via `SeshatConfigOverride`. |
| `keyword_extraction_llm` | `None` | Same class as [`llm` → `_LLMConfig`](#llm--_llmconfig). When set, the sparse leg uses it to extract discriminating keywords from the query before `plainto_tsquery`. Applies to `keyword`/`hybrid` modes. |
| `multi_query` | `None` | See [`multi_query` → `MultiQueryConfig`](#multi_query--multiqueryconfig). `None` disables multi-query. |
| `reranker` | `None` | See [`reranker` → `RerankerConfig`](#reranker--rerankerconfig). `None` disables reranking. |

#### `multi_query` → `MultiQueryConfig`

Enables multi-query fan-out in `SearchEngine`: generates alternative phrasings of a search query via an LLM and fuses the parallel retrieval results with RRF. Applies to `semantic`/`hybrid` modes.

| Field | Default | Definition |
|---|---|---|
| `llm` | *(required)* | Same class as [`llm` → `_LLMConfig`](#llm--_llmconfig). LLM used to generate query variants. |
| `num_variants` | `3` | Number of alternative query phrasings to generate and fan out in parallel (`1`–`10`). |

#### `reranker` → `RerankerConfig`

Configures a hosted reranker applied after all retrieval legs are fused; always receives the original query.

| Field | Default | Definition |
|---|---|---|
| `provider` | *(required)* | `RerankerProvider` enum: `cohere`, `voyage`. |
| `model` | *(required)* | Reranker model name, e.g. `rerank-v3.5` (Cohere) or `rerank-2` (Voyage). |
| `top_n` | `None` | Truncate reranked results to the top-N after reranking; `None` keeps all results. |
| `max_retries` | `3` | Maximum number of retry attempts on transient errors. |
| `timeout_seconds` | `None` | Per-request HTTP timeout in seconds; `None` means no limit. |
| `api_key_secret_key` | `None` → `<provider>_api_key` | Secrets key for the reranker API key. |

### `secrets` → `SecretsConfig`

Configures the secrets backend used to resolve API keys and connection strings.

| Field | Default | Definition |
|---|---|---|
| `provider` | `aws` | `SecretsProvider` enum: `env`, `aws`. |
| `region` | `eu-west-1` | AWS region for Secrets Manager. |
| `secret_path_prefix` | `seshat` | Prefix applied to all secret paths/names. |
| `endpoint_url` | `None` | Optional custom endpoint URL; set for LocalStack or VPC endpoints. |

### `observability` → `ObservabilityConfig`

Configures MLflow tracing/experiment tracking. Also reused, unchanged, as `EvalConfig.observability`.

| Field | Default | Definition |
|---|---|---|
| `mlflow_tracking_uri` | `http://mlflow:5000` | MLflow tracking server URI. |
| `mlflow_experiment_name` | `seshat` | MLflow experiment name. |

### `api` → `APIConfig`

Configures the FastAPI application's rate limits and startup gates.

| Field | Default | Definition |
|---|---|---|
| `max_jobs_per_user_per_hour` | `10` | Maximum number of jobs a single user may create per hour. |
| `max_concurrent_jobs` | `1` | Maximum number of jobs that may run concurrently. |
| `eval_gate_path` | `<PROJECT_ROOT>/eval_gate.json` | Path to the eval gate JSON file produced by `seshat eval`. |
| `skip_eval_gate` | `False` | Bypass the eval gate check at startup. Should never be used in production. |
| `skip_llm_ping` | `False` | Skip the LLM ping check at startup. Should never be used in production. |
| `root_api_key_secret_key` | `root-api-key` | Secrets key for the root API key used to create new API keys. |

### `document_loader` → `DocumentLoaderConfig`

Configures the document loader used only by `seshat init` to seed the knowledge base from source documents. Optional field on `SeshatConfig`; `None` means unused.

| Field | Default | Definition |
|---|---|---|
| `provider` | `markdown` | `DocumentLoaderProvider` enum: `markdown`. |
| `source_path` | `./init-docs` | Filesystem path to the source documents to load. |

---

## `SeshatConfigOverride`

Per-request override object (not a `BaseSettings` root — constructed programmatically, not from env vars) that lets a single API request override select sections of the base `SeshatConfig` (e.g. to change `rag.search_mode` for one search call). Only fields explicitly set on the override replace the corresponding fields of the base config's sub-model; unset fields fall back to the base config.

| Field | Default | Definition |
|---|---|---|
| `transcription` | `None` | Optional override; see [`transcription` → `TranscriptionConfig`](#transcription--transcriptionconfig). |
| `extraction` | `None` | Optional override; see [`extraction` → `ExtractionConfig`](#extraction--extractionconfig). |
| `rag` | `None` | Optional override; see [`rag` → `RAGConfig`](#rag--ragconfig). |

---

## `EvalConfig`

Top-level `BaseSettings` root for the `seshat eval` harnesses. Loads from `.env` with every env var prefixed `EVAL__` and `__` as the nested delimiter.

| Field | Default | Definition |
|---|---|---|
| `corpus_base_dir` | `<PROJECT_ROOT>/data/eval/corpora` | Root directory for eval corpora. Expected subdirs: one per eval harness. |
| `gate_path` | `<PROJECT_ROOT>/eval_gate.json` | Full path (including filename) for the `GateResult` JSON output; must end in `.json`. |
| `observability` | `ObservabilityConfig()` | Same class as [`observability` → `ObservabilityConfig`](#observability--observabilityconfig). |
| `run_identification` | `True` | Run the identification eval pass (did the pipeline extract the right nodes from the transcript). |
| `run_resolution` | `True` | Run the resolution eval pass (did the pipeline infer the correct relationships between nodes). |
| `run_retrieval` | `True` | Run the retrieval eval pass (does vector search surface the right nodes). |
| `run_grounding` | `True` | Run the grounding eval pass (does the grounding agent correctly identify grounded vs. hallucinated descriptions). |
| `run_grouping` | `True` | Run the grouping eval pass (does the grouping agent correctly cluster extracted items into thematic groups). |
| `max_concurrent_predictions` | `10` | Maximum number of prediction coroutines that may run in parallel during eval. |
| `retrieval_score_thresholds` | `{}` | Per-`SearchMode` minimum score thresholds `[0, 1]` applied during retrieval eval. Absent keys default to `0.0` (no filtering). Set per key, e.g. `EVAL__RETRIEVAL_SCORE_THRESHOLDS__SEMANTIC=0.77`. |

`EvalConfig` also exposes ten computed, read-only directory properties (`identification_corpus_dir`, `resolution_corpus_dir`, `retrieval_corpus_dir`, `grounding_corpus_dir`, `grouping_corpus_dir` and their `*_cache_dir` counterparts) derived from `corpus_base_dir` and an internal cache directory. These are not independently configurable via env vars.

---

## `.env.example`

All variables below are optional — every field in `SeshatConfig` and `EvalConfig` has a default, so the app runs with zero env vars set (aside from real provider API keys, which are resolved through the `secrets` backend, not shown here). Uncomment and adjust as needed.

```dotenv
# ── Logging ─────────────────────────────────────────────────────────────────
LOGGING__LEVEL=INFO
# LOGGING__NOISY_LOGGERS='{"aiobotocore":"WARNING","botocore":"WARNING","httpx":"WARNING","langchain":"WARNING","langchain_core":"WARNING","langchain_aws":"WARNING","langchain_openai":"WARNING","mlflow":"WARNING","urllib3.connectionpool":"ERROR"}'

# ── Transcription ───────────────────────────────────────────────────────────
TRANSCRIPTION__PROVIDER=assemblyai
# TRANSCRIPTION__MODEL=
TRANSCRIPTION__LANGUAGE=en
TRANSCRIPTION__MAX_FILE_BYTES=524288000
TRANSCRIPTION__MAX_AUDIO_SECONDS=7200
TRANSCRIPTION__MAX_RETRIES=3
# TRANSCRIPTION__TIMEOUT_SECONDS=
# TRANSCRIPTION__API_KEY_SECRET_KEY=assemblyai_api_key

# ── Vector store ────────────────────────────────────────────────────────────
VECTOR_STORE__PROVIDER=pgvector
VECTOR_STORE__CONNECTION_SECRET_KEY=postgres_url

# ── Vector index (embeddings) ───────────────────────────────────────────────
VECTOR_INDEX__COLLECTION=seshat-docs
VECTOR_INDEX__EMBEDDING_PROVIDER=openai
VECTOR_INDEX__EMBEDDING_MODEL=text-embedding-3-small
# VECTOR_INDEX__API_KEY_SECRET_KEY=openai_api_key
VECTOR_INDEX__MAX_INDEXING_TOKENS=500000

# ── Knowledge base store (Postgres) ─────────────────────────────────────────
KB_STORE__SCHEMA_NAME=knowledge_base
KB_STORE__POOL_MIN_SIZE=2
KB_STORE__POOL_MAX_SIZE=10
KB_STORE__CONNECTION_SECRET_KEY=postgres_url

# ── Ops store (Postgres) ─────────────────────────────────────────────────────
OPS_STORE__SCHEMA_NAME=ops
OPS_STORE__POOL_MIN_SIZE=2
OPS_STORE__POOL_MAX_SIZE=10
OPS_STORE__CONNECTION_SECRET_KEY=postgres_url

# ── Blob store (S3) ─────────────────────────────────────────────────────────
BLOB_STORE__BUCKET=seshat-mvp
BLOB_STORE__REGION=eu-west-1
# BLOB_STORE__ENDPOINT_URL=          # e.g. http://localhost:4566 for LocalStack

# ── Extraction pipeline ──────────────────────────────────────────────────────
# EXTRACTION__CONCEPT_TYPES=["decision","risk","action_item","open_question"]
EXTRACTION__IDENTIFICATION__PROVIDER=anthropic
EXTRACTION__IDENTIFICATION__MODEL=claude-sonnet-4-6
EXTRACTION__IDENTIFICATION__TEMPERATURE=0.0
EXTRACTION__IDENTIFICATION__MAX_RETRIES=3
EXTRACTION__IDENTIFICATION__TIMEOUT_SECONDS=300.0
EXTRACTION__IDENTIFICATION__MAX_CONCURRENT_CALLS=5
# EXTRACTION__IDENTIFICATION__MAX_OUTPUT_TOKENS=
# EXTRACTION__IDENTIFICATION__API_KEY_SECRET_KEY=anthropic_api_key

EXTRACTION__IDENTIFICATION_SELF_REVIEW__ENABLED=false
# EXTRACTION__IDENTIFICATION_SELF_REVIEW__LLM__PROVIDER=
# EXTRACTION__IDENTIFICATION_SELF_REVIEW__LLM__MODEL=

EXTRACTION__RESOLUTION__PROVIDER=anthropic
EXTRACTION__RESOLUTION__MODEL=claude-sonnet-4-6
EXTRACTION__RESOLUTION__TEMPERATURE=0.0
EXTRACTION__RESOLUTION__MAX_RETRIES=3
EXTRACTION__RESOLUTION__TIMEOUT_SECONDS=300.0
EXTRACTION__RESOLUTION__MAX_CONCURRENT_CALLS=10
EXTRACTION__RESOLUTION__MAX_GLOBAL_CALLS=30
# EXTRACTION__RESOLUTION__MAX_OUTPUT_TOKENS=
# EXTRACTION__RESOLUTION__API_KEY_SECRET_KEY=anthropic_api_key

EXTRACTION__RESOLUTION_SELF_REVIEW__ENABLED=false
# EXTRACTION__RESOLUTION_SELF_REVIEW__LLM__PROVIDER=
# EXTRACTION__RESOLUTION_SELF_REVIEW__LLM__MODEL=

# EXTRACTION__GROUPED_IDENTIFICATION_TYPES=["decision"]

# EXTRACTION__GROUNDING__PROVIDER=openai
# EXTRACTION__GROUNDING__MODEL=gpt-5.4-nano
# EXTRACTION__GROUNDING__USE_FULL_TRANSCRIPT=true

EXTRACTION__CONFIDENCE_THRESHOLD=0.7
# EXTRACTION__PER_TYPE_THRESHOLDS__DECISION=0.6
EXTRACTION__AUTO_MODE=false
EXTRACTION__MAX_TOTAL_INPUT_TOKENS=2000000
EXTRACTION__MAX_TOTAL_OUTPUT_TOKENS=500000
EXTRACTION__MAX_TOTAL_EMBEDDING_TOKENS=10000000
EXTRACTION__MAX_HINT_NODES=20
EXTRACTION__MAX_HINT_TOKENS=1000
# EXTRACTION__IDENTIFICATION_TIMEOUT_SECONDS=
# EXTRACTION__RESOLUTION_TIMEOUT_SECONDS=

# ── RAG / retrieval ──────────────────────────────────────────────────────────
RAG__ENABLED=true
RAG__TOP_K=5
RAG__MIN_SIMILARITY_SCORE=0.5
RAG__MAX_CONTEXT_TOKENS=4000
RAG__TRAVERSAL_MAX_DEPTH=1
# RAG__TRAVERSAL_REL_TYPES=["mitigates","blocks"]
RAG__MAX_CONCURRENT_RETRIEVALS=20
RAG__SEARCH_MODE=semantic

# RAG__KEYWORD_EXTRACTION_LLM__PROVIDER=
# RAG__KEYWORD_EXTRACTION_LLM__MODEL=

# RAG__MULTI_QUERY__LLM__PROVIDER=
# RAG__MULTI_QUERY__LLM__MODEL=
# RAG__MULTI_QUERY__NUM_VARIANTS=3

# RAG__RERANKER__PROVIDER=cohere
# RAG__RERANKER__MODEL=rerank-v3.5
# RAG__RERANKER__TOP_N=
# RAG__RERANKER__MAX_RETRIES=3
# RAG__RERANKER__TIMEOUT_SECONDS=
# RAG__RERANKER__API_KEY_SECRET_KEY=cohere_api_key

# ── Secrets backend ──────────────────────────────────────────────────────────
SECRETS__PROVIDER=aws
SECRETS__REGION=eu-west-1
SECRETS__SECRET_PATH_PREFIX=seshat
# SECRETS__ENDPOINT_URL=             # e.g. http://localhost:4566 for LocalStack

# ── Observability (MLflow) ──────────────────────────────────────────────────
OBSERVABILITY__MLFLOW_TRACKING_URI=http://mlflow:5000
OBSERVABILITY__MLFLOW_EXPERIMENT_NAME=seshat

# ── API ──────────────────────────────────────────────────────────────────────
API__MAX_JOBS_PER_USER_PER_HOUR=10
API__MAX_CONCURRENT_JOBS=1
# API__EVAL_GATE_PATH=/path/to/eval_gate.json
API__SKIP_EVAL_GATE=false
API__SKIP_LLM_PING=false
API__ROOT_API_KEY_SECRET_KEY=root-api-key

# ── Document loader (only used by `seshat init`) ────────────────────────────
# DOCUMENT_LOADER__PROVIDER=markdown
# DOCUMENT_LOADER__SOURCE_PATH=./init-docs

MAX_CONCURRENT_INIT_RUNS=1

# ── Eval harness (seshat eval) ───────────────────────────────────────────────
# EVAL__CORPUS_BASE_DIR=/path/to/data/eval/corpora
# EVAL__GATE_PATH=/path/to/eval_gate.json
EVAL__OBSERVABILITY__MLFLOW_TRACKING_URI=http://mlflow:5000
EVAL__OBSERVABILITY__MLFLOW_EXPERIMENT_NAME=seshat
EVAL__RUN_IDENTIFICATION=true
EVAL__RUN_RESOLUTION=true
EVAL__RUN_RETRIEVAL=true
EVAL__RUN_GROUNDING=true
EVAL__RUN_GROUPING=true
EVAL__MAX_CONCURRENT_PREDICTIONS=10
# EVAL__RETRIEVAL_SCORE_THRESHOLDS__SEMANTIC=0.77
# EVAL__RETRIEVAL_SCORE_THRESHOLDS__KEYWORD=0.1
# EVAL__RETRIEVAL_SCORE_THRESHOLDS__HYBRID=0.5
```
