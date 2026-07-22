from __future__ import annotations

import statistics
from typing import TYPE_CHECKING

import mlflow

from seshat.core.utils.log import get_logger

if TYPE_CHECKING:
    from seshat.core.models.enums import ConceptType
    from seshat.core.models.nodes import FailedResolutionSource

logger = get_logger(__name__)


def log_identification_failures(failed_concept_types: list[ConceptType]) -> None:
    """Log identification-agent failures as an MLflow metric and tag.

    No-ops when there were no failures, so successful runs stay uncluttered.
    """
    if not failed_concept_types:
        return

    mlflow.log_metric("identification.failed_concept_types", len(failed_concept_types))
    mlflow.set_tag("identification.failed_concept_types", ",".join(ct.value for ct in failed_concept_types))


def log_latency_metrics(stage: str, durations: list[float], metrics_prefix: str = "latency.") -> None:
    """Log LLM call latency percentiles as MLflow metrics to the active run.

    No-ops when no run is active so it is safe to call from the production pipeline
    before MLflow is wired there.
    """
    if not mlflow.active_run():
        logger.debug("No active MLflow run: skipping log_latency_metrics for stage %s", stage)
        return
    if not durations:
        logger.debug("No LLM calls duration: skipping log_latency_metrics for stage %s", stage)
        return

    sorted_d = sorted(durations)
    p95_idx = min(int(0.95 * len(sorted_d)), len(sorted_d) - 1)

    if stage:
        stage = stage.replace(".", "_").replace(" ", "_").replace("-", "_")
        metrics_prefix = f"{metrics_prefix}{stage}."

    mlflow.log_metrics(
        {
            f"{metrics_prefix}min_ms": round(sorted_d[0]),
            f"{metrics_prefix}max_ms": round(sorted_d[-1]),
            f"{metrics_prefix}mean_ms": round(statistics.mean(sorted_d)),
            f"{metrics_prefix}median_ms": round(statistics.median(sorted_d)),
            f"{metrics_prefix}p95_ms": round(sorted_d[p95_idx]),
            f"{metrics_prefix}call_count": float(len(sorted_d)),
        }
    )


def log_resolution_failures(failed_sources: list[FailedResolutionSource]) -> None:
    """Log resolution-agent failures as an MLflow metric and tag.

    No-ops when there were no failures, so successful runs stay uncluttered.
    """
    if not failed_sources:
        return

    mlflow.log_metric("resolution.failed_sources", len(failed_sources))
    mlflow.set_tag("resolution.failed_sources", ",".join(str(s.node_id) for s in failed_sources))


def log_token_metrics(
    stage: str,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    cache_creation_tokens: int = 0,
    embedding_input_tokens: int = 0,
    audio_seconds: int = 0,
    metrics_prefix: str = "usage.",
) -> None:
    """Log LLM and embedding token counts as MLflow metrics to the active run.

    No-ops when no run is active so it is safe to call from the production pipeline
    before MLflow is wired there.
    """
    if not mlflow.active_run():
        logger.debug("No active MLflow run: skipping log_token_metrics for stage %s", stage)
        return

    if stage:
        stage = stage.replace(".", "_").replace(" ", "_").replace("-", "_")
        metrics_prefix = f"{metrics_prefix}{stage}."

    metrics = {
        f"{metrics_prefix}llm_input": float(input_tokens),
        f"{metrics_prefix}llm_output": float(output_tokens),
        f"{metrics_prefix}cache_read_input_tokens": float(cache_read_tokens),
        f"{metrics_prefix}cache_creation_input_tokens": float(cache_creation_tokens),
        f"{metrics_prefix}embedding_input": float(embedding_input_tokens),
        f"{metrics_prefix}audio_seconds": float(audio_seconds),
    }
    mlflow.log_metrics({k: v for k, v in metrics.items() if v != 0.0})


def set_error_tag(exc: Exception) -> None:
    """Set the `error` tag on the active MLflow run, truncated to fit MLflow's tag length limit."""
    mlflow.set_tag("error", str(exc)[:250])


def set_phase_tag(phase: str) -> None:
    """Set the `phase` tag on the active MLflow run."""
    mlflow.set_tag("phase", phase)
