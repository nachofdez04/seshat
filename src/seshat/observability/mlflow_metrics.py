import statistics

import mlflow

from seshat.utils.log import get_logger

logger = get_logger(__name__)


def log_token_metrics(
    stage: str,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    cache_creation_tokens: int = 0,
    embedding_input_tokens: int = 0,
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

    mlflow.log_metrics(
        {
            f"{metrics_prefix}llm_input": float(input_tokens),
            f"{metrics_prefix}llm_output": float(output_tokens),
            f"{metrics_prefix}cache_read_input_tokens": float(cache_read_tokens),
            f"{metrics_prefix}cache_creation_input_tokens": float(cache_creation_tokens),
            f"{metrics_prefix}embedding_input": float(embedding_input_tokens),
        }
    )


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
