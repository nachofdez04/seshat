from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from seshat.models.enums import (
    ConceptType,
    DocumentLoaderProvider,
    EmbeddingProvider,
    LLMProvider,
    RelationshipType,
    SecretsProvider,
    TranscriptionProvider,
    VectorStoreProvider,
)
from seshat.utils.log import get_logger

logger = get_logger(__name__)


class BaseConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    def _set_on_frozen_model(self, field: str, value: object) -> None:
        # Pydantic v2: only valid inside model_validator(mode="after"); bypasses the frozen guard.
        object.__setattr__(self, field, value)


class _LLMConfig(BaseConfig):
    provider: LLMProvider
    model: str
    temperature: float = Field(default=0.0, ge=0)
    max_retries: int = Field(default=3, ge=0)
    timeout_seconds: float = Field(default=300.0, gt=0, description="Per-request HTTP timeout in seconds.")
    max_concurrent_calls: int = Field(default=5, gt=0, description="Maximum number of simultaneous LLM calls.")
    max_output_tokens: int | None = Field(
        default=None, gt=0, description="Maximum tokens the LLM may generate per call; None means no limit."
    )
    api_key_secret_key: str | None = Field(
        default=None,
        description="Secrets key for the LLM API key. Defaults to '<provider>_api_key' if not set.",
    )

    @model_validator(mode="after")
    def _default_api_key_secret_key(self) -> "_LLMConfig":
        if self.api_key_secret_key is None:
            self._set_on_frozen_model("api_key_secret_key", f"{self.provider}_api_key")
        return self


class IdentificationLLMConfig(_LLMConfig):
    provider: LLMProvider = LLMProvider.ANTHROPIC
    model: str = "claude-sonnet-4-6"


class GroundingLLMConfig(_LLMConfig):
    provider: LLMProvider = LLMProvider.OPENAI
    model: str = "gpt-5.4-nano"
    use_full_transcript: bool = Field(
        default=True,
        description="When False, grounding uses only the identified quote instead of the full transcript.",
    )


class ReflectiveLLMConfig(BaseConfig):
    enabled: bool = Field(default=False, description="When True, the agent runs an extract → validate → filter pass.")
    llm: _LLMConfig | None = Field(
        default=None,
        description="LLM used for the self-review (validate) call. Falls back to the stage's primary LLM when None.",
    )


class ResolutionLLMConfig(_LLMConfig):
    provider: LLMProvider = LLMProvider.ANTHROPIC
    model: str = "claude-sonnet-4-6"
    max_concurrent_calls: int = Field(
        default=10, gt=0, description="Maximum simultaneous LLM calls per resolution agent."
    )
    max_global_calls: int = Field(
        default=30, gt=0, description="Global cap on simultaneous LLM calls across all resolution agents."
    )


class ExtractionConfig(BaseConfig):
    concept_types: list[ConceptType] = Field(
        default_factory=lambda: list(ConceptType),
        description="Concept types that the extraction pipeline will attempt to extract.",
    )
    identification: IdentificationLLMConfig = Field(
        default_factory=IdentificationLLMConfig, description="LLM settings used for the identification step."
    )
    identification_self_review: ReflectiveLLMConfig = Field(
        default_factory=ReflectiveLLMConfig,
        description=(
            "Self-review loop settings for identification agents; "
            "set identification_self_review.enabled=True to activate."
        ),
    )
    resolution: ResolutionLLMConfig = Field(
        default_factory=ResolutionLLMConfig, description="LLM and concurrency settings for the resolution step."
    )
    resolution_self_review: ReflectiveLLMConfig = Field(
        default_factory=ReflectiveLLMConfig,
        description=(
            "Self-review loop settings for resolution agents; set resolution_self_review.enabled=True to activate."
        ),
    )
    grouped_identification_types: set[ConceptType] = Field(
        default_factory=lambda: {ConceptType.DECISION},
        description="Concept types for which identified items are passed through the grouping step.",
    )
    grounding: GroundingLLMConfig | None = Field(
        default=None, description="Optional second LLM used to ground extraction results; None disables grounding."
    )
    confidence_threshold: float = Field(
        default=0.7, ge=0, le=1, description="Minimum heuristics score required to retain an identified node."
    )
    per_type_thresholds: dict[ConceptType, float] | None = Field(
        default=None, description="Optional per-concept-type confidence thresholds that override the global threshold."
    )
    auto_mode: bool = Field(
        default=False, description="When True, auto-approve extraction results without manual review."
    )
    max_total_input_tokens: int = Field(
        default=2_000_000, gt=0, description="Hard cap on total input tokens consumed in one extraction run."
    )
    max_total_output_tokens: int = Field(
        default=500_000, gt=0, description="Hard cap on total output tokens generated in one extraction run."
    )
    max_total_embedding_tokens: int = Field(
        default=10_000_000, gt=0, description="Hard cap on total embedding input tokens in one extraction run."
    )
    max_hint_nodes: int = Field(
        default=20, gt=0, description="Maximum number of KB hint nodes injected into the extraction prompt."
    )
    max_hint_tokens: int = Field(
        default=1000, gt=0, description="Maximum tokens consumed by hint nodes injected into the extraction prompt."
    )
    identification_timeout_seconds: float | None = Field(
        default=None, gt=0, description="Optional wall-clock timeout for a full extraction run; None means no limit."
    )
    resolution_timeout_seconds: float | None = Field(
        default=None, gt=0, description="Optional wall-clock timeout for a full resolution run; None means no limit."
    )

    @model_validator(mode="after")
    def check_grounding_provider(self) -> "ExtractionConfig":
        if self.grounding is not None and self.grounding.provider == self.identification.provider:
            raise ValueError(
                "`grounding.provider` must differ from `identification.provider`"
                f" (both are '{self.identification.provider}')"
            )

        return self


class VectorIndexConfig(BaseConfig):
    collection: str = Field(
        default="seshat-docs", description="Name of the collection (table or index) within the vector store."
    )
    embedding_provider: EmbeddingProvider = EmbeddingProvider.OPENAI
    embedding_model: str = "text-embedding-3-small"
    api_key_secret_key: str | None = Field(
        default=None,
        description="Secrets key for the embedding API key. Defaults to '<provider>_api_key' if not set.",
    )
    max_indexing_tokens: int = Field(
        default=500_000, gt=0, description="Maximum total tokens that may be embedded in a single RAG indexing run."
    )

    @model_validator(mode="after")
    def _default_api_key_secret_key(self) -> "VectorIndexConfig":
        if self.api_key_secret_key is None:
            self._set_on_frozen_model("api_key_secret_key", f"{self.embedding_provider}_api_key")
        return self


class RAGConfig(BaseConfig):
    enabled: bool = True
    top_k: int = Field(default=5, gt=0)
    # TODO: calibrate against labeled retrieval corpus; 0.5 is a placeholder for text-embedding-3-small cosine scores
    min_similarity_score: float = Field(
        default=0.5, ge=0, le=1, description="Minimum similarity score [0, 1] to retain a retrieved result."
    )
    max_context_tokens: int = Field(
        default=4000, gt=0, description="Maximum tokens the retrieved context may occupy in the prompt."
    )
    traversal_max_depth: int = Field(
        default=1, ge=0, description="Maximum graph-traversal depth when expanding retrieved nodes via relationships."
    )
    traversal_rel_types: list[RelationshipType] | None = Field(
        default=None, description="Relationship types to follow during traversal; None means all."
    )
    max_concurrent_retrievals: int = Field(
        default=20, gt=0, description="Maximum number of simultaneous RAG retrieval calls."
    )


class VectorStoreConfig(BaseConfig):
    provider: VectorStoreProvider = VectorStoreProvider.PGVECTOR
    connection_secret_key: str = Field(
        default="postgres_url", description="Secrets key for the vector store connection string."
    )


class KBStoreConfig(BaseConfig):
    schema_name: str = Field(
        default="ops",
        pattern=r"^[a-z_][a-z0-9_]*$",
        description="PostgreSQL schema name used by the KB store.",
    )
    pool_min_size: int = Field(default=2, gt=0)
    pool_max_size: int = Field(default=10, gt=0)
    connection_secret_key: str = Field(
        default="postgres_url", description="Secrets key for the KB store connection string."
    )


class BlobStoreConfig(BaseConfig):
    bucket: str = Field(default="seshat-mvp")
    region: str = Field(default="eu-west-1")
    endpoint_url: str | None = Field(
        default=None, description="Optional custom endpoint URL; set for LocalStack or other S3-compatible stores."
    )


class TranscriptionConfig(BaseConfig):
    provider: TranscriptionProvider = TranscriptionProvider.ASSEMBLYAI
    model: str | None = None
    language: str = Field(default="en", description="BCP-47 language code for the audio being transcribed.")
    max_file_bytes: int = Field(default=500 * 1024 * 1024, gt=0)
    max_audio_seconds: int = Field(default=7200, gt=0)
    max_retries: int = Field(default=3, ge=0)
    timeout_seconds: float | None = Field(
        default=None, gt=0, description="Per-request timeout for transcription calls in seconds; None means no limit."
    )
    api_key_secret_key: str | None = Field(
        default=None,
        description="Secrets key for the transcription provider API key. Defaults to '<provider>_api_key' if not set.",
    )

    @model_validator(mode="after")
    def _default_api_key_secret_key(self) -> "TranscriptionConfig":
        if self.api_key_secret_key is None:
            self._set_on_frozen_model("api_key_secret_key", f"{self.provider}_api_key")
        return self


class ObservabilityConfig(BaseConfig):
    mlflow_tracking_uri: str = "http://mlflow:5000"
    mlflow_experiment_name: str = "seshat"


class SecretsConfig(BaseConfig):
    provider: SecretsProvider = SecretsProvider.AWS
    region: str = "eu-west-1"
    secret_path_prefix: str = "seshat"
    endpoint_url: str | None = Field(
        default=None, description="Optional custom endpoint URL; set for LocalStack or VPC endpoints."
    )


class DocumentLoaderConfig(BaseConfig):
    provider: DocumentLoaderProvider = Field(
        default=DocumentLoaderProvider.MARKDOWN, description="Document loader backend used to ingest source documents."
    )
    source_path: str = "./init-docs"


class SeshatConfig(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_nested_delimiter="__", extra="ignore")

    transcription: TranscriptionConfig = Field(default_factory=TranscriptionConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    vector_index: VectorIndexConfig = Field(default_factory=VectorIndexConfig)
    kb_store: KBStoreConfig = Field(default_factory=KBStoreConfig)
    blob_store: BlobStoreConfig = Field(default_factory=BlobStoreConfig)
    extraction: ExtractionConfig = Field(default_factory=ExtractionConfig)
    rag: RAGConfig = Field(default_factory=RAGConfig)
    secrets: SecretsConfig = Field(default_factory=SecretsConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)

    # only used for `seshat init`
    document_loader: DocumentLoaderConfig | None = None

    max_jobs_per_user_per_hour: int = Field(default=10, gt=0)
    max_concurrent_jobs: int = Field(default=1, gt=0)
    max_concurrent_init_runs: int = Field(default=1, gt=0)

    @model_validator(mode="after")
    def _warn_grounding_disabled(self) -> "SeshatConfig":
        if self.extraction.grounding is None:
            logger.warning("grounding=None: heuristics-only confidence scoring.")
        return self


class SeshatConfigOverride(BaseConfig):
    transcription: TranscriptionConfig | None = None
    extraction: ExtractionConfig | None = None
    rag: RAGConfig | None = None


_config: SeshatConfig | None = None


def get_config() -> SeshatConfig:
    global _config
    if _config is None:
        logger.info("Loading configuration...")
        _config = SeshatConfig()
    return _config


def get_request_settings(overrides: SeshatConfigOverride | None) -> SeshatConfig:
    base = get_config()
    if overrides is None:
        return base

    logger.info("Overriding configuration...")
    update = {}
    for field in overrides.model_fields_set:
        base_section = getattr(base, field)
        override_section = getattr(overrides, field)
        update[field] = base_section.model_copy(update=override_section.model_dump(exclude_unset=True))

    return base.model_copy(update=update)
