from __future__ import annotations

import json
import os
import pathlib
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from fastapi import APIRouter, FastAPI

from seshat.api.routers import graph, health, jobs
from seshat.api.state import AppState
from seshat.config.settings import SeshatConfig
from seshat.observability.mlflow_setup import setup_mlflow
from seshat.utils.log import get_logger
from seshat.worker.bootstrap import build_worker_context
from seshat.worker.pipeline_runner import PipelineRunner
from seshat.worker.queue import AsyncioTaskQueue

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from seshat.models.nodes import ExtractionResult

logger = get_logger(__name__)


def create_app() -> FastAPI:
    """Create and return a FastAPI app instance."""
    v1_router = APIRouter(prefix="/v1")
    v1_router.include_router(health.router)
    v1_router.include_router(jobs.router)
    v1_router.include_router(graph.router)

    app = FastAPI(title="Seshat API", version="0.1.0", lifespan=_lifespan)
    app.include_router(v1_router)
    return app


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncGenerator[None]:
    _check_eval_gate()

    config = SeshatConfig()
    setup_mlflow(config.observability)

    async with build_worker_context(config) as ctx:
        stranded = await ctx.ops.get_stranded_writing_jobs()
        for job_id in stranded:
            await ctx.ops.fail_job(job_id, "writing", "Server crash during write", recoverable=True)
            logger.warning("Startup recovery: marked stranded job %s as FAILED", job_id)

        result_store: dict[str, ExtractionResult] = {}
        runner = PipelineRunner(
            ctx.ingestion_orch, ctx.extraction_orch, ctx.writing_stage, ctx.ops, result_store, ctx.blob_store
        )
        queue = AsyncioTaskQueue()

        app.state.app_state = AppState(
            config=config,
            kb_store=ctx.kb_store,
            manual_ingestion=ctx.manual_ingestion,
            ops=ctx.ops,
            queue=queue,
            results=result_store,
            runner=runner,
            blob_store=ctx.blob_store,
        )

        yield


def _check_eval_gate() -> None:
    if os.environ.get("SESHAT_SKIP_EVAL_GATE", "").lower() in ("1", "true"):
        logger.warning("Eval gate check bypassed — do not use in production")
        return
    gate_path = pathlib.Path("eval_gate.json")
    if not gate_path.exists():
        logger.error("FATAL: eval_gate.json not found. Run 'seshat eval' first.")
        raise SystemExit(1)
    gate = json.loads(gate_path.read_text())
    if not gate.get("passed"):
        logger.error("FATAL: eval gate not passed. Run 'seshat eval' first.")
        raise SystemExit(1)


if __name__ == "__main__":
    import asyncio

    import uvicorn

    from seshat.utils.log import configure_logging

    async def _serve() -> None:
        configure_logging()
        config = uvicorn.Config(create_app(), host="0.0.0.0", port=8000, log_config=None)
        await uvicorn.Server(config).serve()

    asyncio.run(_serve())
