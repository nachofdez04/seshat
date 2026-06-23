from pathlib import Path
from typing import ClassVar

from pydantic import Field, computed_field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from seshat.config.settings import ObservabilityConfig

_ROOT_DIR: Path = Path(__file__).resolve().parent.parent.parent.parent
_DEFAULT_CORPUS_BASE_DIR: Path = _ROOT_DIR / "data" / "eval" / "corpora"
_DEFAULT_GATE_PATH: Path = _ROOT_DIR / "eval_gate.json"


class EvalConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="EVAL__",
        env_nested_delimiter="__",
        env_file=".env",
        extra="ignore",
        frozen=True,
    )

    corpus_base_dir: Path = Field(
        default=_DEFAULT_CORPUS_BASE_DIR,
        description="Root directory for eval corpora. Expected subdirs: one per eval harness.",
    )
    gate_path: Path = Field(
        default=_DEFAULT_GATE_PATH,
        description="Full path (including filename) for the GateResult JSON output.",
    )
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
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
    run_grounding: bool = Field(
        default=True,
        description=(
            "Run the grounding eval pass, i.e., "
            "check if the grounding agent correctly identifies grounded vs. hallucinated descriptions."
        ),
    )
    run_grouping: bool = Field(
        default=True,
        description=(
            "Run the grouping eval pass, i.e., "
            "check if the grouping agent correctly clusters extracted items into thematic groups."
        ),
    )
    max_concurrent_predictions: int = Field(
        default=10,
        gt=0,
        description="Maximum number of prediction coroutines that may run in parallel during eval.",
    )
    # 0.0 virtually disables score filtering, so all candidates rank.
    # we recommend to calibrate it using the retrieval meta-scorer and set via EVAL__RETRIEVAL_SCORE_THRESHOLD
    retrieval_score_threshold: float = Field(
        default=0.0,
        ge=0,
        le=1,
        description="Minimum similarity score [0, 1] forwarded to the vector store during retrieval eval.",
    )

    _identification_subdir: ClassVar[str] = "identification"
    _resolution_subdir: ClassVar[str] = "resolution"
    _retrieval_subdir: ClassVar[str] = "retrieval"
    _grounding_subdir: ClassVar[str] = "grounding"
    _grouping_subdir: ClassVar[str] = "grouping"
    # a hidden folder in the project root for caching intermediate results during eval runs; not intended for manual use
    _cache_dir: ClassVar[Path] = _ROOT_DIR / ".seshat" / "eval_cache"

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
    def grounding_corpus_dir(self) -> Path:
        return self.corpus_base_dir / self._grounding_subdir

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
    def grounding_cache_dir(self) -> Path:
        return self._cache_dir / self._grounding_subdir

    @computed_field  # type: ignore[misc]
    @property
    def grouping_corpus_dir(self) -> Path:
        return self.corpus_base_dir / self._grouping_subdir

    @computed_field  # type: ignore[misc]
    @property
    def grouping_cache_dir(self) -> Path:
        return self._cache_dir / self._grouping_subdir

    @field_validator("gate_path", mode="after")
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
            (self.run_grounding, self.grounding_corpus_dir),
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
            self.grounding_cache_dir,
            self.grouping_cache_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)
        return self
