from __future__ import annotations

import asyncio
import subprocess
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Annotated, Any

if TYPE_CHECKING:
    from seshat.eval.corpus_tags import CorpusTagFilter

import typer
from dotenv import load_dotenv

from seshat.app.platform.observability.mlflow_setup import setup_mlflow
from seshat.core.config.eval_settings import EvalConfig
from seshat.core.config.settings import GroundingLLMConfig, ObservabilityConfig, SeshatConfig
from seshat.core.utils.log import configure_logging, get_logger, set_job_id

logger = get_logger(__name__)

app = typer.Typer(name="seshat", help="Seshat — meeting knowledge base CLI", no_args_is_help=True)
eval_app = typer.Typer(help="Eval harnesses, calibration, and tooling", no_args_is_help=True)
app.add_typer(eval_app, name="eval")

_HARNESS_TYPES = ["grounding", "grouping", "identification", "resolution", "retrieval"]
_CALIBRATION_TYPES = ["retrieval", "identification"]


def _patch_httpx_ssl() -> None:
    import httpx

    _orig = httpx.Client.__init__

    def _no_verify(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        kwargs.setdefault("verify", False)
        _orig(self, *args, **kwargs)

    httpx.Client.__init__ = _no_verify  # type: ignore[method-assign]


@eval_app.command("harness")
def eval_cmd(
    harness: Annotated[str, typer.Argument(help=f"Harness to run: {' | '.join(_HARNESS_TYPES)}")],
    tags: Annotated[
        list[str] | None,
        typer.Option("--tag", help="Filter corpus by tag in `key=value` format. Repeatable."),
    ] = None,
) -> None:
    """Run an evaluation harness against the labelled corpus."""
    import mlflow

    async def _run() -> None:
        eval_config, seshat_config, run_name = _bootstrap_eval(harness)

        match harness:
            case "grouping":
                from seshat.eval.grouping.entrypoint import run
            case "identification":
                from seshat.eval.identification.entrypoint import run
            case "resolution":
                from seshat.eval.resolution.entrypoint import run
            case "retrieval":
                from seshat.eval.retrieval.entrypoint import run
            case "grounding":
                from seshat.eval.grounding.entrypoint import run
            case _:
                typer.echo(f"Unknown harness '{harness}'. Choose from: {', '.join(_HARNESS_TYPES)}", err=True)
                raise typer.Exit(code=1)

        tag_filter = _parse_tags(tags) if tags is not None else None
        with mlflow.start_run(run_name=run_name):
            await run(eval_config, seshat_config, tag_filter=tag_filter)

    asyncio.run(_run())


@eval_app.command("calibrate")
def calibrate_cmd(
    component: Annotated[str, typer.Argument(help=f"Component to calibrate: {' | '.join(_CALIBRATION_TYPES)}")],
    pc_curve: bool = typer.Option(False, "--pc-curve", help="Plot precision-coverage curve (identification only)"),
    p_target: float = typer.Option(0.95, "--p-target", help="Precision target for threshold sweep"),
    ignore_grounding: bool = typer.Option(False, "--ignore-grounding", help="Ignore grounding signal in calibration"),
) -> None:
    """Calibrate eval thresholds and weights for the given component."""
    import mlflow

    async def _run() -> None:
        eval_config, seshat_config, run_name = _bootstrap_eval(f"{component}-calibration")

        _kwargs: dict[str, Any] = {"eval_config": eval_config, "seshat_config": seshat_config}
        match component:
            case "retrieval":
                from seshat.eval.calibration.retrieval_entrypoint import run

            case "identification":
                from seshat.eval.calibration.identification_entrypoint import run

                mode = "precision_coverage_curve" if pc_curve else "sweep_threshold"
                _kwargs.update({"mode": mode, "p_target": p_target, "ignore_grounding": ignore_grounding})

            case _:
                typer.echo(f"Unknown component '{component}'. Choose from: {', '.join(_CALIBRATION_TYPES)}", err=True)
                raise typer.Exit(code=1)

        with mlflow.start_run(run_name=run_name):
            await run(**_kwargs)

    asyncio.run(_run())


@eval_app.command("mlflow")
def mlflow_cmd(
    port: int = typer.Option(5000, "--port", help="Port to serve MLflow UI on"),
) -> None:
    """Start the MLflow tracking server."""
    result = subprocess.run(
        ["uv", "run", "--no-sync", "mlflow", "server", "--port", str(port)],
        check=False,
    )
    raise typer.Exit(code=result.returncode)


@app.command("api")
def api_cmd(
    host: str = typer.Option("0.0.0.0", "--host", help="Bind host"),
    port: int = typer.Option(8000, "--port", help="Bind port"),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload (development only)"),
    no_access_log: bool = typer.Option(True, "--no-access-log/--access-log", help="Suppress uvicorn access log"),
) -> None:
    """Start the Seshat API server."""
    import uvicorn

    uvicorn.run(
        "seshat.app.platform.api.app:create_app",
        factory=True,
        host=host,
        port=port,
        reload=reload,
        access_log=(not no_access_log),
    )


@app.command("worker")
def worker_cmd() -> None:
    """Start the Seshat background worker (standalone mode)."""
    typer.echo("The worker is currently embedded in the API process.", err=True)
    typer.echo("Standalone worker support is not yet implemented.", err=True)
    raise typer.Exit(code=1)


@app.command("migrate")
def migrate_cmd(
    revision: str = typer.Argument(default="head", help="Alembic revision target (default: head)"),
) -> None:
    """Run Alembic database migrations."""
    result = subprocess.run(
        ["uv", "run", "--no-sync", "alembic", "upgrade", revision],
        check=False,
    )
    raise typer.Exit(code=result.returncode)


def _parse_tags(tags: list[str]) -> CorpusTagFilter:
    """Parse ``key=value`` tag strings into a dict, erroring on malformed entries."""
    result: CorpusTagFilter = {}
    for tag in tags:
        if "=" not in tag:
            typer.echo(f"Invalid tag format '{tag}': expected key=value", err=True)
            raise typer.Exit(code=1)
        k, _, v = tag.partition("=")
        result[k] = v
    return result


def _assert_reachable(uri: str, *, label: str, timeout: float = 2.0) -> None:
    import socket
    from urllib.parse import urlparse

    parsed = urlparse(uri)
    host = parsed.hostname or "localhost"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        with socket.create_connection((host, port), timeout=timeout):
            pass
    except OSError as exc:
        typer.echo(f"Cannot reach {label} at {uri} — is the stack up? ({exc})", err=True)
        raise typer.Exit(code=1) from exc


def _bootstrap_eval(harness_type: str) -> tuple[EvalConfig, SeshatConfig, str]:
    """Set up MLflow and configs for an eval or calibration run."""
    load_dotenv()
    _patch_httpx_ssl()

    job_id = f"seshat-eval-{harness_type}"
    run_name = f"seshat-eval-{harness_type}-{datetime.now(tz=UTC).isoformat(timespec='minutes')}"

    set_job_id(job_id)
    eval_config = EvalConfig(
        observability=ObservabilityConfig(mlflow_tracking_uri="http://localhost:5000", mlflow_experiment_name=job_id)
    )

    _assert_reachable(eval_config.observability.mlflow_tracking_uri, label="MLflow")
    setup_mlflow(eval_config.observability)

    seshat_config = SeshatConfig()
    configure_logging(seshat_config.logging)

    if harness_type == "grounding" and seshat_config.extraction.grounding is None:
        seshat_config = seshat_config._with(extraction=seshat_config.extraction._with(grounding=GroundingLLMConfig()))
        logger.warning("grounding LLM config not found in SeshatConfig, using default grounding config")

    return eval_config, seshat_config, run_name


if __name__ == "__main__":
    app()
