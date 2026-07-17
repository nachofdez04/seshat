from __future__ import annotations

from typing import TYPE_CHECKING

import mlflow

from seshat.app.pipeline.bootstrap import get_search_engine
from seshat.core.utils.log import get_logger
from seshat.eval.calibration.retrieval_meta_scorer import RetrievalMetaScorer
from seshat.eval.mlflow_logging import log_retrieval_model
from seshat.infra.vector_store.factory import get_vector_store

if TYPE_CHECKING:
    from seshat.core.config.eval_settings import EvalConfig
    from seshat.core.config.settings import SeshatConfig

logger = get_logger(__name__)


async def run(eval_config: EvalConfig, seshat_config: SeshatConfig) -> None:
    search_mode = seshat_config.rag.search_mode
    log_retrieval_model("seshat-retrieval", seshat_config.vector_index)

    vector_store = get_vector_store(seshat_config)
    search_engine = get_search_engine(seshat_config, vector_store)
    scorer = RetrievalMetaScorer(
        search_engine=search_engine,
        vector_store=vector_store,
        config=eval_config,
        rag_config=seshat_config.rag,
    )

    logger.info("Sweeping thresholds for search_mode=%r...", search_mode.value)
    result = await scorer.sweep_threshold()

    suggested = result.suggested_threshold
    logger.info("Suggested threshold: %.2f (search_mode=%r)", suggested, search_mode.value)
    logger.info(
        "Set EVAL__RETRIEVAL_SCORE_THRESHOLDS__%s=%.2f in .env",
        search_mode.value.upper(),
        suggested,
    )

    llm_cfg = seshat_config.rag.keyword_extraction_llm
    metrics = next(p for p in result.points if p.threshold == suggested)
    mlflow.log_metrics(metrics.model_dump())
    mlflow.log_param("retrieval.search_mode", search_mode.value)
    mlflow.log_param("retrieval.keyword_extraction_llm", llm_cfg.model if llm_cfg is not None else "none")
