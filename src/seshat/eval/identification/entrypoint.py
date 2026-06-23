from __future__ import annotations

from typing import TYPE_CHECKING

from seshat.eval.identification.runner import IdentificationEvalRunner
from seshat.eval.mlflow_logging import log_eval_model
from seshat.pipeline.bootstrap import build_orchestrator
from seshat.utils.log import get_logger

if TYPE_CHECKING:
    from seshat.config.eval_settings import EvalConfig
    from seshat.config.settings import SeshatConfig
    from seshat.eval.corpus_tags import CorpusTagFilter

logger = get_logger(__name__)


async def run(eval_config: EvalConfig, seshat_config: SeshatConfig, tag_filter: CorpusTagFilter | None = None) -> None:
    async with build_orchestrator(seshat_config) as orchestrator:
        llm_cfg = seshat_config.extraction.identification
        self_review_cfg = seshat_config.extraction.identification_self_review
        logger.info(
            "LLM provider=%r model=%r temperature=%s self_review=%s",
            llm_cfg.provider.value,
            llm_cfg.model,
            llm_cfg.temperature,
            self_review_cfg.enabled,
        )
        model_id = log_eval_model(
            "seshat-identification-agent",
            inference_component=orchestrator._identification_registry,
            llm_config=llm_cfg,
            self_review_config=self_review_cfg,
        )

        runner = IdentificationEvalRunner(orchestrator=orchestrator, config=eval_config)
        gate = await runner.run(tag_filter=tag_filter, model_id=model_id)

        logger.info("identification eval: passed=%s", gate.passed)
