from __future__ import annotations

import asyncio
import selectors
import subprocess
import sys
from typing import TYPE_CHECKING, Annotated, Any

if TYPE_CHECKING:
    from collections.abc import Coroutine

import typer

from seshat.cli._eval_support import (
    CALIBRATION_TYPES,
    HARNESS_TYPES,
    bootstrap_eval,
    parse_tags,
)
from seshat.core.config.eval_settings import EvalConfig
from seshat.core.models.enums import TranscriptionProvider
from seshat.core.utils.log import get_logger

logger = get_logger(__name__)

_SUPPORTED_TRANSCRIPTION_PROVIDERS = (
    TranscriptionProvider.ASSEMBLYAI,
    TranscriptionProvider.OPENAI,
)


def _run_async(coro: Coroutine) -> None:
    # psycopg (asyncpg-backed PGVector) is incompatible with Windows ProactorEventLoop
    if sys.platform == "win32":
        asyncio.run(coro, loop_factory=lambda: asyncio.SelectorEventLoop(selectors.SelectSelector()))
    else:
        asyncio.run(coro)


app = typer.Typer(name="seshat", help="Seshat — meeting knowledge base CLI", no_args_is_help=True)
eval_app = typer.Typer(help="Eval harnesses, calibration, and tooling", no_args_is_help=True)
app.add_typer(eval_app, name="eval")


@eval_app.command("harness")
def eval_cmd(
    harness: Annotated[
        str | None,
        typer.Argument(help=f"Harness to run: {' | '.join(HARNESS_TYPES)}. Omit with --all to run all enabled."),
    ] = None,
    tags: Annotated[
        list[str] | None,
        typer.Option("--tag", help="Filter corpus by tag in `key=value` format. Repeatable."),
    ] = None,
    clear_cache: Annotated[
        bool,
        typer.Option("--clear-cache", help="Clear the prediction cache of each harness that runs, before running."),
    ] = False,
    run_all: Annotated[
        bool,
        typer.Option("--all", help="Run every harness whose EVAL__RUN_<harness> flag is enabled."),
    ] = False,
    providers: Annotated[
        list[str] | None,
        typer.Option(
            "--provider",
            help=(
                "Transcription harness only: compare these providers side by side, one MLflow run each. "
                "Repeatable. Only the configured default provider updates the gate file."
            ),
        ),
    ] = None,
) -> None:
    """Run one evaluation harness, or every enabled harness with --all."""
    if harness is not None and run_all:
        typer.echo("Pass either a harness name or --all, not both.", err=True)
        raise typer.Exit(code=1)

    if providers and harness != "transcription":
        typer.echo("--provider is only supported by the transcription harness.", err=True)
        raise typer.Exit(code=1)

    # Parsed here rather than in the harness so a typo fails before MLflow is contacted.
    parsed_providers = _parse_providers(providers)

    # Single named harness: the simple case — run it, and let any failure propagate (fail-hard).
    if harness is not None:
        if clear_cache:
            _clear_cache(harness)

        _run_single_harness(harness, tags, parsed_providers)
        return

    if not run_all:
        typer.echo("Provide a harness name or --all.", err=True)
        raise typer.Exit(code=1)

    harnesses = EvalConfig().enabled_harnesses
    if not harnesses:
        typer.echo("No harnesses enabled: every EVAL__RUN_<harness> flag is false.", err=True)
        raise typer.Exit(code=1)

    # A single harness failing (transient provider error, a bad fixture) should not throw away
    # the spend on the others — run them all, collect failures, and report at the end.
    failed: list[str] = []
    for h in harnesses:
        if clear_cache:
            _clear_cache(h)

        try:
            _run_single_harness(h, tags)
        except Exception as exc:  # report and continue across the suite
            logger.exception("Harness %r failed", h)
            typer.echo(f"Harness '{h}' failed: {exc}", err=True)
            failed.append(h)

    if failed:
        typer.echo(f"\n{len(failed)}/{len(harnesses)} harness(es) failed: {', '.join(failed)}", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"\nAll {len(harnesses)} harnesses completed: {', '.join(harnesses)}")


def _run_single_harness(
    harness: str, tags: list[str] | None, providers: list[TranscriptionProvider] | None = None
) -> None:
    """Bootstrap MLflow and run a single named harness against the labelled corpus."""
    import mlflow

    async def _run() -> None:
        eval_config, seshat_config, run_name = bootstrap_eval(harness)

        extra_kwargs: dict[str, Any] = {}
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
            case "transcription":
                from seshat.eval.transcription.entrypoint import run

                extra_kwargs["providers"] = providers
            case _:
                typer.echo(f"Unknown harness '{harness}'. Choose from: {', '.join(HARNESS_TYPES)}", err=True)
                raise typer.Exit(code=1)

        tag_filter = parse_tags(tags) if tags is not None else None
        with mlflow.start_run(run_name=run_name):
            await run(eval_config, seshat_config, tag_filter=tag_filter, **extra_kwargs)

    _run_async(_run())


def _parse_providers(providers: list[str] | None) -> list[TranscriptionProvider] | None:
    if not providers:
        return None

    valid = ", ".join(p.value for p in _SUPPORTED_TRANSCRIPTION_PROVIDERS)
    try:
        parsed = [TranscriptionProvider(p) for p in providers]
    except ValueError as exc:
        typer.echo(f"{exc}. Choose from: {valid}", err=True)
        raise typer.Exit(code=1) from exc

    unsupported = [p.value for p in parsed if p not in _SUPPORTED_TRANSCRIPTION_PROVIDERS]
    if unsupported:
        typer.echo(f"Transcription provider(s) not supported: {', '.join(unsupported)}. Choose from: {valid}", err=True)
        raise typer.Exit(code=1)

    return parsed


@eval_app.command("clear-cache")
def clear_cache_cmd(
    harness: Annotated[
        str | None,
        typer.Argument(help=f"Harness cache to clear: {' | '.join(HARNESS_TYPES)}. Omit to clear all."),
    ] = None,
) -> None:
    """Clear cached eval predictions for one harness, or all harnesses when none is given."""
    if harness is None:
        for h in HARNESS_TYPES:
            _clear_cache(h)
    else:
        _clear_cache(harness)


@eval_app.command("calibrate")
def calibrate_cmd(
    component: Annotated[str, typer.Argument(help=f"Component to calibrate: {' | '.join(CALIBRATION_TYPES)}")],
    pc_curve: bool = typer.Option(False, "--pc-curve", help="Plot precision-coverage curve (identification only)"),
    p_target: float = typer.Option(0.95, "--p-target", help="Precision target for threshold sweep"),
    ignore_grounding: bool = typer.Option(False, "--ignore-grounding", help="Ignore grounding signal in calibration"),
    clear_cache: Annotated[
        bool,
        typer.Option("--clear-cache", help="Clear this component's prediction cache before calibrating."),
    ] = False,
) -> None:
    """Calibrate eval thresholds and weights for the given component."""
    import mlflow

    if clear_cache:
        _clear_cache(component)

    async def _run() -> None:
        eval_config, seshat_config, run_name = bootstrap_eval(f"{component}-calibration")

        _kwargs: dict[str, Any] = {"eval_config": eval_config, "seshat_config": seshat_config}
        match component:
            case "retrieval":
                from seshat.eval.calibration.retrieval_entrypoint import run

            case "identification":
                from seshat.eval.calibration.identification_entrypoint import run

                mode = "precision_coverage_curve" if pc_curve else "sweep_threshold"
                _kwargs.update({"mode": mode, "p_target": p_target, "ignore_grounding": ignore_grounding})

            case _:
                typer.echo(f"Unknown component '{component}'. Choose from: {', '.join(CALIBRATION_TYPES)}", err=True)
                raise typer.Exit(code=1)

        with mlflow.start_run(run_name=run_name):
            await run(**_kwargs)

    _run_async(_run())


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


def _clear_cache(harness: str) -> None:
    """Clear the prediction cache directory for a single harness."""
    from seshat.eval.cache import clear_cache_dir

    if harness not in HARNESS_TYPES:
        typer.echo(f"Unknown harness '{harness}'. Choose from: {', '.join(HARNESS_TYPES)}", err=True)
        raise typer.Exit(code=1)

    cache_dir = EvalConfig.cache_dir_for(harness)
    clear_cache_dir(cache_dir)
    typer.echo(f"Cleared eval cache for '{harness}': {cache_dir}")


if __name__ == "__main__":
    app()
