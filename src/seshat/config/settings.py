import math
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator, model_validator
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
            object.__setattr__(self, "api_key_secret_key", f"{self.provider}_api_key")
        return self


class IdentificationLLMConfig(_LLMConfig):
    provider: LLMProvider = LLMProvider.ANTHROPIC
    model: str = "claude-sonnet-4-6"


class VerificationLLMConfig(_LLMConfig):
    provider: LLMProvider = LLMProvider.OPENAI
    model: str = "gpt-5.4-nano"
    use_full_transcript: bool = Field(
        default=True,
        description="When False, verification uses only the identified quote instead of the full transcript.",
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


class ConfidenceWeights(BaseConfig):
    # Default weights are hand-tuned placeholders; calibrate against a labeled corpus before
    # enabling verification in production. With verification=None the verification weight is
    # redistributed to heuristics automatically, but the heuristics weight itself is also uncalibrated.
    verification: float = Field(default=0.70, ge=0, lt=1)
    heuristics: float = Field(default=0.30, gt=0, le=1)

    @model_validator(mode="after")
    def _weights_sum_to_one(self) -> "ConfidenceWeights":
        total = self.verification + self.heuristics
        if not math.isclose(total, 1.0, abs_tol=1e-6):
            raise ValueError(f"ConfidenceWeights must sum to 1.0, got {total:.6f}")
        return self

    _DISABLEABLE_SIGNALS: frozenset[str] = frozenset({"verification"})

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
    identification: IdentificationLLMConfig = Field(
        default_factory=IdentificationLLMConfig, description="LLM settings used for the identification step."
    )
    resolution: ResolutionLLMConfig = Field(
        default_factory=ResolutionLLMConfig, description="LLM and concurrency settings for the resolution step."
    )
    concept_types: list[ConceptType] = Field(
        default_factory=lambda: list(ConceptType),
        description="Concept types that the extraction pipeline will attempt to extract.",
    )
    # TODO: calibrate against a labeled corpus before use
    confidence_threshold: float = Field(
        default=0.7, ge=0, le=1, description="Minimum composite confidence score required to retain an identified node."
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
    max_hint_nodes: int = Field(
        default=20, gt=0, description="Maximum number of KB hint nodes injected into the extraction prompt."
    )
    max_hint_tokens: int = Field(
        default=1000, gt=0, description="Maximum tokens consumed by hint nodes injected into the extraction prompt."
    )
    verification: VerificationLLMConfig | None = Field(
        default=None, description="Optional second LLM used to verify extraction results; None disables verification."
    )
    confidence_weights: ConfidenceWeights = Field(
        default_factory=ConfidenceWeights, description="Weights used to compute the composite confidence score."
    )
    identification_timeout_seconds: float | None = Field(
        default=None, gt=0, description="Optional wall-clock timeout for a full extraction run; None means no limit."
    )
    resolution_timeout_seconds: float | None = Field(
        default=None, gt=0, description="Optional wall-clock timeout for a full resolution run; None means no limit."
    )
    grouped_identification_types: set[ConceptType] = Field(
        default_factory=lambda: {ConceptType.DECISION},
        description="Concept types for which identified items are passed through the grouping step.",
    )

    @model_validator(mode="after")
    def check_verification_provider(self) -> "ExtractionConfig":
        if self.verification is not None and self.verification.provider == self.identification.provider:
            raise ValueError(
                "`verification.provider` must differ from `identification.provider`"
                f" (both are '{self.identification.provider}')"
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
            object.__setattr__(self, "api_key_secret_key", f"{self.embedding_provider}_api_key")
        return self


class RAGConfig(BaseConfig):
    enabled: bool = True
    top_k: int = Field(default=5, gt=0)
    # TODO: calibrate against labeled retrieval corpus; 0.5 is a placeholder for text-embedding-3-small cosine scores
    min_score: float = Field(
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


class ObservabilityConfig(BaseConfig):
    mlflow_tracking_uri: str = "http://mlflow:5000"
    mlflow_experiment_name: str = "seshat"


class EvalConfig(BaseConfig):
    corpus_base_dir: Path = Field(
        description=("Root directory for eval corpora. Expected subdirs: identification/, resolution/, retrieval/.")
    )
    gate_path: Path = Field(description="Full path (including filename) for the GateResult JSON output.")
    observability: ObservabilityConfig = Field(
        default_factory=lambda: ObservabilityConfig(mlflow_experiment_name="seshat-eval")
    )
    run_identification: bool = Field(
        default=True,
        description=(
            "Run the identification eval pass, i.e., "
            "check if the pipeline extracted the right nodes from the transcript."
        ),
    )
    run_resolution: bool = Field(
        default=True,
        description=(
            "Run the resolution eval pass, i.e., "
            "check if the pipeline inferred the correct relationships between nodes."
        ),
    )
    run_retrieval: bool = Field(
        default=True,
        description=(
            "Run the retrieval eval pass, i.e., "
            "check if vector search surfaces the right nodes (similar and related neighbors)."
        ),
    )
    run_verification: bool = Field(
        default=True,
        description=(
            "Run the verification eval pass, i.e., "
            "check if the verification agent correctly identifies grounded vs. hallucinated descriptions."
        ),
    )
    run_grouping: bool = Field(
        default=True,
        description=(
            "Run the grouping eval pass, i.e., "
            "check if the grouping agent correctly clusters extracted items into thematic groups."
        ),
    )
    nli_scorer_enabled: bool = Field(
        default=False,
        description=(
            "Enable NLI-based faithfulness scoring, i.e., check if the extracted information is faithful to the source."
        ),
    )
    max_concurrent_predictions: int = Field(
        default=10,
        gt=0,
        description="Maximum number of prediction coroutines that may run in parallel during eval.",
    )
    # 0.0 disables score filtering so all candidates rank — calibrate before tightening.
    retrieval_score_threshold: float = Field(
        default=0.0,
        ge=0,
        le=1,
        description="Minimum similarity score [0, 1] forwarded to the vector store during retrieval eval.",
    )

    _identification_subdir: ClassVar[str] = "identification"
    _resolution_subdir: ClassVar[str] = "resolution"
    _retrieval_subdir: ClassVar[str] = "retrieval"
    _verification_subdir: ClassVar[str] = "verification"
    _grouping_subdir: ClassVar[str] = "grouping"
    # a hidden folder in the project root for caching intermediate results during eval runs; not intended for manual use
    _cache_dir: ClassVar[Path] = Path(__file__).resolve().parent.parent.parent.parent / ".seshat" / "eval_cache"

    @computed_field  # type: ignore[misc]
    @property
    def identification_corpus_dir(self) -> Path:
        return self.corpus_base_dir / self._identification_subdir

    @computed_field  # type: ignore[misc]
    @property
    def resolution_corpus_dir(self) -> Path:
        return self.corpus_base_dir / self._resolution_subdir

    @computed_field  # type: ignore[misc]
    @property
    def retrieval_corpus_dir(self) -> Path:
        return self.corpus_base_dir / self._retrieval_subdir

    @computed_field  # type: ignore[misc]
    @property
    def verification_corpus_dir(self) -> Path:
        return self.corpus_base_dir / self._verification_subdir

    @computed_field  # type: ignore[misc]
    @property
    def identification_cache_dir(self) -> Path:
        return self._cache_dir / self._identification_subdir

    @computed_field  # type: ignore[misc]
    @property
    def resolution_cache_dir(self) -> Path:
        return self._cache_dir / self._resolution_subdir

    @computed_field  # type: ignore[misc]
    @property
    def retrieval_cache_dir(self) -> Path:
        return self._cache_dir / self._retrieval_subdir

    @computed_field  # type: ignore[misc]
    @property
    def verification_cache_dir(self) -> Path:
        return self._cache_dir / self._verification_subdir

    @computed_field  # type: ignore[misc]
    @property
    def grouping_corpus_dir(self) -> Path:
        return self.corpus_base_dir / self._grouping_subdir

    @computed_field  # type: ignore[misc]
    @property
    def grouping_cache_dir(self) -> Path:
        return self._cache_dir / self._grouping_subdir

    @field_validator("gate_path")
    @classmethod
    def _validate_gate_path(cls, v: Path) -> Path:
        if v.suffix != ".json":
            raise ValueError(f"gate_path must be a .json file, got: {v}")
        v.parent.mkdir(parents=True, exist_ok=True)
        return v

    @model_validator(mode="after")
    def _validate_corpus_dirs(self) -> "EvalConfig":
        checks = [
            (self.run_identification, self.identification_corpus_dir),
            (self.run_resolution, self.resolution_corpus_dir),
            (self.run_retrieval, self.retrieval_corpus_dir),
            (self.run_verification, self.verification_corpus_dir),
            (self.run_grouping, self.grouping_corpus_dir),
        ]
        for enabled, path in checks:
            if enabled and not path.is_dir():
                raise ValueError(f"corpus dir does not exist: {path}")
        return self

    @model_validator(mode="after")
    def _create_cache_dirs(self) -> "EvalConfig":
        for path in (
            self.identification_cache_dir,
            self.resolution_cache_dir,
            self.retrieval_cache_dir,
            self.verification_cache_dir,
            self.grouping_cache_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)
        return self


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
