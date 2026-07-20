"""Plumbing for the `seshat eval` CLI commands.

Kept separate from `app.py` so that module holds only the Typer command surface.
Import direction is one-way: `app.py` imports from here, never the reverse.
"""

from __future__ import annotations

import os
import sys
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import typer
from dotenv import load_dotenv

from seshat.app.platform.observability.mlflow_setup import setup_mlflow
from seshat.core.config.eval_settings import EvalConfig
from seshat.core.config.settings import GroundingLLMConfig, ObservabilityConfig, SeshatConfig
from seshat.core.utils.http_patch import disable_httpx_ssl_verification
from seshat.core.utils.log import configure_logging, get_logger, set_job_id
from seshat.eval.mlflow_logging import configure_trace_processors

if TYPE_CHECKING:
    from seshat.eval.corpus_tags import CorpusTagFilter

logger = get_logger(__name__)

HARNESS_TYPES = ["grounding", "grouping", "identification", "resolution", "retrieval"]
CALIBRATION_TYPES = ["retrieval", "identification"]


def bootstrap_eval(harness_type: str) -> tuple[EvalConfig, SeshatConfig, str]:
    """Set up MLflow and configs for an eval or calibration run."""
    load_dotenv()

    # Build config before anything makes an httpx call: the SSL opt-out must be applied first.
    seshat_config = SeshatConfig()
    configure_logging(seshat_config.logging)

    if seshat_config.disable_ssl_verification:
        disable_httpx_ssl_verification()

    job_id = f"seshat-eval-{harness_type}"
    run_name = f"seshat-eval-{harness_type}-{datetime.now(tz=UTC).isoformat(timespec='minutes')}"

    set_job_id(job_id)

    observability = ObservabilityConfig(mlflow_tracking_uri="http://localhost:5000", mlflow_experiment_name=job_id)
    _assert_reachable(observability.mlflow_tracking_uri, label="MLflow")
    setup_mlflow(observability)
    ensure_utf8_streams()
    bound_mlflow_retries()

    # Clear any span processor a prior harness registered globally (e.g. identification's
    # node slimmer) so it cannot fire on this harness's differently-shaped prediction spans.
    configure_trace_processors()

    if harness_type == "grounding" and seshat_config.extraction.grounding is None:
        seshat_config = seshat_config._with(extraction=seshat_config.extraction._with(grounding=GroundingLLMConfig()))
        logger.warning("grounding LLM config not found in SeshatConfig, using default grounding config")

    eval_config = EvalConfig()
    return eval_config, seshat_config, run_name


def ensure_utf8_streams() -> None:
    """Make stdout/stderr tolerate non-ASCII so a stray char cannot crash the CLI.

    MLflow logs a runner emoji at end_run; on a cp1252 Windows console that raised
    UnicodeEncodeError at shutdown. Reconfiguring to utf-8 with backslashreplace degrades
    an unencodable char instead of crashing. No-op on streams lacking reconfigure().
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="backslashreplace")


def bound_mlflow_retries() -> None:
    """Cap MLflow's retry/timeout budgets so a slow tracking server fails fast.

    LangChain autolog is intentionally on during eval (agent traces show in the MLflow UI),
    so a large cold harness queues many trace exports. MLflow has TWO independent paths that
    must both be bounded:
      * sync API calls — MLFLOW_HTTP_REQUEST_MAX_RETRIES (default 7) / _TIMEOUT (default 120)
      * async trace export — MLFLOW_ASYNC_TRACE_LOGGING_RETRY_TIMEOUT (default 500), drained
        at process exit; this is what backed up after a cold 138-call harness and hung ~15min.
    Bounding only the HTTP path is insufficient. setdefault leaves explicit user overrides intact.
    """
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "1")
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "15")
    os.environ.setdefault("MLFLOW_ASYNC_TRACE_LOGGING_RETRY_TIMEOUT", "20")


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


def parse_tags(tags: list[str]) -> CorpusTagFilter:
    """Parse ``key=value`` tag strings into a dict, erroring on malformed entries."""
    result: CorpusTagFilter = {}
    for tag in tags:
        if "=" not in tag:
            typer.echo(f"Invalid tag format '{tag}': expected key=value", err=True)
            raise typer.Exit(code=1)
        k, _, v = tag.partition("=")
        result[k] = v
    return result
