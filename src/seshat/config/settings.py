import logging
import math

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

logger = logging.getLogger(__name__)


class BaseConfig(BaseModel):
    model_config = ConfigDict(frozen=True)


class LLMConfig(BaseConfig):
    provider: LLMProvider = LLMProvider.ANTHROPIC
    model: str = "claude-sonnet-4-6"
    temperature: float = Field(default=0.0, ge=0)


class VerificationConfig(BaseConfig):
    provider: LLMProvider
    model: str


class ConfidenceWeights(BaseConfig):
    logprobs: float = Field(default=0.5, ge=0, lt=1)
    verification: float = Field(default=0.35, ge=0, lt=1)
    heuristics: float = Field(default=0.15, gt=0, le=1)

    @model_validator(mode="after")
    def _weights_sum_to_one(self) -> "ConfidenceWeights":
        total = self.logprobs + self.verification + self.heuristics
        if not math.isclose(total, 1.0, abs_tol=1e-6):
            raise ValueError(f"ConfidenceWeights must sum to 1.0, got {total:.6f}")
        return self

    _DISABLEABLE_SIGNALS: frozenset[str] = frozenset({"logprobs", "verification"})

    def redistribute(self, disabled_signals: set[str]) -> "ConfidenceWeights":
        """Return new weights with disabled signals zeroed and remaining weights scaled to sum to 1.0."""
        unknown = disabled_signals - self._DISABLEABLE_SIGNALS
        if unknown:
            disabled_signals_str = sorted(self._DISABLEABLE_SIGNALS)
            raise ValueError(
                f"Unknown or non-disableable signals: {sorted(unknown)}. Must be one of {disabled_signals_str}"
            )
        active = {k: v for k, v in self.model_dump().items() if k not in disabled_signals}
        total = sum(active.values())
        scaled = {k: v / total for k, v in active.items()} | dict.fromkeys(disabled_signals, 0.0)
        return ConfidenceWeights.model_construct(**scaled)


class ExtractionConfig(BaseConfig):
    llm: LLMConfig = Field(default_factory=LLMConfig, description="LLM settings used for the extraction step.")
    concept_types: list[ConceptType] = Field(
        default_factory=lambda: list(ConceptType),
        description="Concept types that the extraction pipeline will attempt to extract.",
    )
    confidence_threshold: float = Field(
        default=0.7, ge=0, le=1, description="Minimum composite confidence score required to retain an extracted node."
    )
    per_type_thresholds: dict[ConceptType, float] | None = Field(
        default=None, description="Optional per-concept-type confidence thresholds that override the global threshold."
    )
    auto_mode: bool = Field(
        default=False, description="When True, auto-approve extraction results without manual review."
    )
    max_chunk_count: int = Field(
        default=50, gt=0, description="Maximum number of transcript chunks processed in a single extraction pass."
    )
    max_output_tokens: int = Field(
        default=2048, gt=0, description="Maximum number of tokens the LLM may generate per extraction request."
    )
    max_total_input_tokens: int = Field(
        default=2_000_000, gt=0, description="Hard cap on total input tokens consumed across all chunks in one run."
    )
    max_total_output_tokens: int = Field(
        default=500_000, gt=0, description="Hard cap on total output tokens generated across all chunks in one run."
    )
    max_transcript_chunk_tokens: int = Field(
        default=8000, gt=0, description="Maximum token length for a single transcript chunk sent to the extraction LLM."
    )
    chunk_overlap_tokens: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Token overlap between consecutive chunks in the fixed-size fallback chunker; "
            "None defaults to ~20% of max_transcript_chunk_tokens."
        ),
    )
    max_hint_nodes: int = Field(
        default=20, gt=0, description="Maximum number of KB hint nodes injected into the extraction prompt."
    )
    max_hint_tokens: int = Field(
        default=1000, gt=0, description="Maximum tokens consumed by hint nodes injected into the extraction prompt."
    )
    merge_similarity_threshold: float = Field(
        default=0.85,
        ge=0,
        le=1,
        description="Minimum embedding similarity required to merge two candidate nodes during deduplication.",
    )
    max_retries: int = Field(default=3, ge=0)
    verification: VerificationConfig | None = Field(
        default=None, description="Optional second LLM used to verify extraction results; None disables verification."
    )
    confidence_weights: ConfidenceWeights = Field(
        default_factory=ConfidenceWeights, description="Weights used to compute the composite confidence score."
    )
    result_cache_enabled: bool = Field(
        default=False, description="When True, extraction results are cached to avoid redundant LLM calls."
    )

    @model_validator(mode="after")
    def check_verification_provider(self) -> "ExtractionConfig":
        if self.verification is not None and self.verification.provider == self.llm.provider:
            raise ValueError(
                f"`verification.provider` must differ from `llm.provider` (both are '{self.llm.provider}')"
            )

        if self.verification is None:
            logger.warning("verification=None: heuristics-only confidence scoring.")

        return self


class VectorIndexConfig(BaseConfig):
    collection: str = Field(
        default="seshat-docs", description="Name of the collection (table or index) within the vector store."
    )
    embedding_provider: EmbeddingProvider = EmbeddingProvider.OPENAI
    embedding_model: str = "text-embedding-3-small"
    max_indexing_tokens: int = Field(
        default=500_000, gt=0, description="Maximum total tokens that may be embedded in a single RAG indexing run."
    )


class RAGConfig(BaseConfig):
    enabled: bool = True
    top_k: int = Field(default=5, gt=0)
    max_context_tokens: int = Field(
        default=4000, gt=0, description="Maximum tokens the retrieved context may occupy in the prompt."
    )
    traversal_max_depth: int = Field(
        default=1, ge=0, description="Maximum graph-traversal depth when expanding retrieved nodes via relationships."
    )
    traversal_rel_types: list[RelationshipType] | None = Field(
        default=None, description="Relationship types to follow during traversal; None means all."
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
    model_config = SettingsConfigDict(env_file=".env", env_nested_delimiter="__")

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
