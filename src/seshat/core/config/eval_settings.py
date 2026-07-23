from pathlib import Path
from typing import ClassVar

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from seshat.core.config.settings import DEFAULT_EVAL_GATE_PATH, PROJECT_ROOT
from seshat.core.models.enums import SearchMode

_DEFAULT_CORPUS_BASE_DIR: Path = PROJECT_ROOT / "data" / "eval" / "corpora"


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
        default=DEFAULT_EVAL_GATE_PATH,
        description="Full path (including filename) for the GateResult JSON output.",
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
    run_transcription: bool = Field(
        default=True,
        description=(
            "Run the transcription eval pass, i.e., "
            "measure the Word Error Rate of the configured transcription provider against reference transcripts."
        ),
    )
    max_concurrent_predictions: int = Field(
        default=10,
        gt=0,
        description="Maximum number of prediction coroutines that may run in parallel during eval.",
    )
    # Per-mode score thresholds calibrated by the retrieval meta-scorer (argmax macro-F2).
    # Absent keys default to 0.0 (no filtering). Each mode has its own score scale
    # (cosine similarity for SEMANTIC, ts_rank_cd for KEYWORD, RRF for HYBRID), so thresholds
    # must be calibrated independently. Set via EVAL__RETRIEVAL_SCORE_THRESHOLDS__SEMANTIC=0.77 etc.
    retrieval_score_thresholds: dict[SearchMode, float] = Field(
        default_factory=dict,
        description="Per-mode minimum score thresholds [0, 1] applied during retrieval eval.",
    )

    _identification_subdir: ClassVar[str] = "identification"
    _resolution_subdir: ClassVar[str] = "resolution"
    _retrieval_subdir: ClassVar[str] = "retrieval"
    _grounding_subdir: ClassVar[str] = "grounding"
    _grouping_subdir: ClassVar[str] = "grouping"
    _transcription_subdir: ClassVar[str] = "transcription"
    # a hidden folder in the project root for caching intermediate results during eval runs; not intended for manual use
    _cache_dir: ClassVar[Path] = PROJECT_ROOT / ".seshat" / "eval_cache"

    def corpus_dir_for(self, harness: str) -> Path:
        """Return the corpus directory for a harness name (relative to the configured corpus_base_dir)."""
        subdir = getattr(self, f"_{harness}_subdir", None)
        if subdir is None:
            raise ValueError(f"unknown harness: {harness!r}")

        return self.corpus_base_dir / subdir

    @property
    def identification_corpus_dir(self) -> Path:
        return self.corpus_dir_for("identification")

    @property
    def grouping_corpus_dir(self) -> Path:
        return self.corpus_dir_for("grouping")

    @property
    def grounding_corpus_dir(self) -> Path:
        return self.corpus_dir_for("grounding")

    @property
    def resolution_corpus_dir(self) -> Path:
        return self.corpus_dir_for("resolution")

    @property
    def retrieval_corpus_dir(self) -> Path:
        return self.corpus_dir_for("retrieval")

    @property
    def transcription_corpus_dir(self) -> Path:
        return self.corpus_dir_for("transcription")

    @classmethod
    def cache_dir_for(cls, harness: str) -> Path:
        """Return the cache directory for a harness name, without constructing an instance."""
        subdir = getattr(cls, f"_{harness}_subdir", None)
        if subdir is None:
            raise ValueError(f"unknown harness: {harness!r}")

        return cls._cache_dir / subdir

    @property
    def identification_cache_dir(self) -> Path:
        return self.cache_dir_for("identification")

    @property
    def grouping_cache_dir(self) -> Path:
        return self.cache_dir_for("grouping")

    @property
    def grounding_cache_dir(self) -> Path:
        return self.cache_dir_for("grounding")

    @property
    def resolution_cache_dir(self) -> Path:
        return self.cache_dir_for("resolution")

    @property
    def retrieval_cache_dir(self) -> Path:
        return self.cache_dir_for("retrieval")

    @property
    def transcription_cache_dir(self) -> Path:
        return self.cache_dir_for("transcription")

    @property
    def enabled_harnesses(self) -> list[str]:
        """Harness names whose run_<harness> flag is enabled, in canonical order."""
        flags = [
            (self.run_identification, "identification"),
            (self.run_resolution, "resolution"),
            (self.run_retrieval, "retrieval"),
            (self.run_grounding, "grounding"),
            (self.run_grouping, "grouping"),
            (self.run_transcription, "transcription"),
        ]
        return [name for enabled, name in flags if enabled]

    @field_validator("gate_path", mode="after")
    @classmethod
    def _validate_gate_path(cls, v: Path) -> Path:
        if v.suffix != ".json":
            raise ValueError(f"gate_path must be a .json file, got: {v}")
        v.parent.mkdir(parents=True, exist_ok=True)
        return v

    @model_validator(mode="after")
    def _validate_corpus_dirs(self) -> "EvalConfig":
        for harness in self.enabled_harnesses:
            path = self.corpus_dir_for(harness)
            if not path.is_dir():
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
            self.transcription_cache_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)
        return self
