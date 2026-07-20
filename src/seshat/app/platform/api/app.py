from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from fastapi import APIRouter, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from langchain_core.messages import HumanMessage

from seshat.app.pipeline.bootstrap import _get_reranker
from seshat.app.pipeline.llm_factory import _build_llm
from seshat.app.platform.api.routers import admin, graph, health, identity, jobs
from seshat.app.platform.api.state import build_app_state
from seshat.app.platform.observability.mlflow_setup import setup_mlflow
from seshat.app.transcription.factory import get_transcriber
from seshat.core.config.settings import SeshatConfig, get_config
from seshat.core.utils.http_patch import disable_httpx_ssl_verification
from seshat.core.utils.log import configure_logging, get_logger, set_job_id
from seshat.infra.vector_store.factory import _build_embeddings

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from seshat.core.config.settings import APIConfig, _LLMConfig


logger = get_logger(__name__)


def create_app() -> FastAPI:
    """Create and return a FastAPI app instance."""
    v1_router = APIRouter(prefix="/v1")
    v1_router.include_router(health.router)
    v1_router.include_router(identity.router)
    v1_router.include_router(jobs.router)
    v1_router.include_router(graph.router)
    v1_router.include_router(admin.router)

    app = FastAPI(title="Seshat API", version="0.1.0", lifespan=_lifespan)
    app.include_router(v1_router)

    @app.exception_handler(RequestValidationError)
    async def _validation_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": exc.errors()})

    return app


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncGenerator[None]:
    set_job_id("api")

    config = get_config()
    configure_logging(config.logging)

    if config.disable_ssl_verification:
        disable_httpx_ssl_verification()

    _emit_config_warnings(config)
    _check_eval_gate(config.api)

    setup_mlflow(config.observability)
    await _ping_external_model_providers(config)

    async with build_app_state(config) as app_state:
        await app_state.job_service.recover_stranded()
        app.state.app_state = app_state
        yield


def _emit_config_warnings(config: SeshatConfig) -> None:
    if config.extraction.grounding is None:
        logger.warning("`grounding=None`: heuristics-only confidence scoring for identified nodes.")


def _check_eval_gate(config: APIConfig) -> None:
    if config.skip_eval_gate:
        logger.warning("`skip_eval_gate=True`: eval gate check bypassed")
        return

    gate_path = config.eval_gate_path
    if not gate_path.exists():
        logger.critical("%s not found. Run 'seshat eval' first.", gate_path)
        raise SystemExit(1)

    gate = json.loads(gate_path.read_text())
    if not gate.get("passed"):
        logger.critical("eval gate not passed. Run 'seshat eval' first.")
        raise SystemExit(1)


async def _ping_external_model_providers(config: SeshatConfig) -> None:
    """Verify connectivity to all configured LLM providers. Raises SystemExit(1) on failure."""
    if config.api.skip_external_provider_ping:
        logger.warning("`skip_external_provider_ping=True`: external model provider ping check bypassed")
        return

    faulty_providers: dict[str, list[str]] = {
        "chat": await _ping_llm_providers(config),
        "embedding": await _ping_embedding_providers(config),
        "transcription": await _ping_transcription_providers(config),
        "reranking": await _ping_reranking_providers(config),
    }

    if any(faulty_providers.values()):
        logger.critical("LLM connectivity check failed: %s", json.dumps(faulty_providers, indent=2))
        raise SystemExit(1)


async def _ping_llm_providers(config: SeshatConfig) -> list[str]:
    llm_configs: list[_LLMConfig | None] = [
        config.extraction.identification,
        config.extraction.identification_self_review.llm,
        config.extraction.grounding,
        config.rag.multi_query.llm,
        config.rag.keyword_extraction_llm,
        config.extraction.resolution,
        config.extraction.resolution_self_review.llm,
    ]

    seen: set[tuple[str, str | None]] = set()
    faulty_providers: list[str] = []
    for llm_cfg in llm_configs:
        if llm_cfg is None:
            continue

        key = (llm_cfg.provider, llm_cfg.api_key_secret_key)
        if key in seen:
            continue

        seen.add(key)

        try:
            llm = _build_llm(llm_cfg, config)
            await llm.ainvoke([HumanMessage(content="ping")], max_tokens=1)
            logger.debug("LLM reachable: provider=%s model=%s", llm_cfg.provider, llm_cfg.model)
        except Exception as exc:
            logger.warning(
                "LLM provider unreachable at startup: provider=%s model=%s — %s: %s",
                llm_cfg.provider,
                llm_cfg.model,
                type(exc).__name__,
                exc,
            )
            faulty_providers.append(llm_cfg.provider)

    return faulty_providers


async def _ping_embedding_providers(config: SeshatConfig) -> list[str]:
    try:
        embeddings = _build_embeddings(config.vector_index, config)
        await embeddings.aembed_query("ping")
        logger.debug("Embedding provider reachable: provider=%s", config.vector_index.embedding_provider)
    except Exception as exc:
        logger.warning(
            "Embedding provider unreachable at startup: provider=%s — %s: %s",
            config.vector_index.embedding_provider,
            type(exc).__name__,
            exc,
        )
        return [config.vector_index.embedding_provider]

    return []


async def _ping_transcription_providers(config: SeshatConfig) -> list[str]:
    try:
        transcriber = get_transcriber(config)
        await transcriber.ping()
        logger.debug("Transcription provider reachable: provider=%s", config.transcription.provider)
    except Exception as exc:
        logger.warning(
            "Transcription provider unreachable at startup: provider=%s — %s: %s",
            config.transcription.provider,
            type(exc).__name__,
            exc,
        )
        return [config.transcription.provider]

    return []


async def _ping_reranking_providers(config: SeshatConfig) -> list[str]:
    reranker = _get_reranker(config)
    if reranker is None:
        return []

    reranker_cfg = config.rag.reranker
    assert reranker_cfg is not None

    try:
        await reranker.ping()
        logger.debug("Reranking provider reachable: provider=%s", reranker_cfg.provider)
    except Exception as exc:
        logger.warning(
            "Reranking provider unreachable at startup: provider=%s — %s: %s",
            reranker_cfg.provider,
            type(exc).__name__,
            exc,
        )
        return [reranker_cfg.provider]

    return []
