from __future__ import annotations

from typing import TYPE_CHECKING

from seshat.core.utils.log import get_logger
from seshat.eval.bootstrap import build_extraction_orchestrator
from seshat.eval.identification.runner import IdentificationEvalRunner
from seshat.eval.mlflow_logging import log_eval_model

if TYPE_CHECKING:
    from seshat.core.config.eval_settings import EvalConfig
    from seshat.core.config.settings import SeshatConfig
    from seshat.eval.corpus_tags import CorpusTagFilter

logger = get_logger(__name__)


async def run(eval_config: EvalConfig, seshat_config: SeshatConfig, tag_filter: CorpusTagFilter | None = None) -> None:
    # Grounding is a production scoring gate — it is not what identification eval measures.
    # Disable it so the harness only tests the identification agent itself.
    seshat_config = seshat_config._with(extraction=seshat_config.extraction._with(grounding=None))
    async with build_extraction_orchestrator(seshat_config) as orchestrator:
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
