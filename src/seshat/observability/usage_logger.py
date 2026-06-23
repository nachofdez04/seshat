import mlflow

from seshat.models.jobs import UsageRecord


def log_usage(run_id: str, stage: str, records: list[UsageRecord]) -> None:
    """Log UsageRecord list as MLflow metrics for the given run."""
    metrics: dict[str, float] = {}
    for record in records:
        key = f"{stage}.{record.call_type.value}"
        metrics[key] = metrics.get(key, 0.0) + record.units

    with mlflow.start_run(run_id=run_id, nested=True):
        mlflow.log_metrics(metrics)


def log_cache_metrics(
    run_id: str,
    stage: str,
    cache_read_tokens: int,
    cache_write_tokens: int,
    cached_tokens: int | None,
) -> None:
    """Log prompt cache hit/miss metrics per agent call."""
    metrics = {
        f"{stage}.cache_read_input_tokens": float(cache_read_tokens),
        f"{stage}.cache_creation_input_tokens": float(cache_write_tokens),
    }
    if cached_tokens is not None:
        metrics[f"{stage}.cached_tokens"] = float(cached_tokens)

    with mlflow.start_run(run_id=run_id, nested=True):
        mlflow.log_metrics(metrics)
