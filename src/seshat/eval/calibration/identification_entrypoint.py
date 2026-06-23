from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import mlflow

from seshat.eval.calibration.identification_meta_scorer import IdentificationMetaScorer
from seshat.eval.mlflow_logging import log_eval_model
from seshat.pipeline.bootstrap import build_orchestrator
from seshat.utils.log import get_logger

if TYPE_CHECKING:
    from seshat.config.eval_settings import EvalConfig
    from seshat.config.settings import SeshatConfig

logger = get_logger(__name__)

CalibrationMode = Literal["sweep_threshold", "precision_coverage_curve"]


async def run(
    eval_config: EvalConfig,
    seshat_config: SeshatConfig,
    *,
    mode: CalibrationMode = "sweep_threshold",
    p_target: float = 0.95,
    ignore_grounding: bool = False,
) -> None:
    async with build_orchestrator(seshat_config) as orchestrator:
        llm_cfg = seshat_config.extraction.identification
        logger.info(
            "LLM provider=%r model=%r temperature=%s", llm_cfg.provider.value, llm_cfg.model, llm_cfg.temperature
        )
        log_eval_model(
            "seshat-identification-agent", inference_component=orchestrator._identification_registry, llm_config=llm_cfg
        )
        scorer = IdentificationMetaScorer(orchestrator=orchestrator, config=eval_config)

        match mode:
            case "sweep_threshold":
                await _run_sweep(scorer, p_target=p_target, ignore_grounding=ignore_grounding)
            case "precision_coverage_curve":
                await _run_pc_curve(scorer, ignore_grounding=ignore_grounding)


async def _run_sweep(
    scorer: IdentificationMetaScorer, *, p_target: float = 0.95, ignore_grounding: bool = False
) -> None:
    logger.info("Sweeping thresholds (p_target=%.3f, ignore_grounding=%s)...", p_target, ignore_grounding)
    result = await scorer.sweep_threshold(p_target=p_target, ignore_grounding=ignore_grounding)

    suggested = result.suggested_threshold
    pt = next(p for p in result.points if p.threshold == suggested)
    logger.info(
        "Suggested threshold: %.4f (precision=%.3f, coverage=%.3f)", suggested, pt.precision_approved, pt.coverage
    )
    logger.info("To use: set EXTRACTION__CONFIDENCE_THRESHOLD=%.2f in .env", suggested)

    mlflow.log_metric("suggested_threshold", suggested)
    mlflow.log_metric("precision_approved_at_suggested", pt.precision_approved)
    mlflow.log_metric("coverage_at_suggested", pt.coverage)


async def _run_pc_curve(scorer: IdentificationMetaScorer, *, ignore_grounding: bool = False) -> None:
    # the PC curve is an exploratory tool; excluding it from MLflow is intentional to reduce noise
    logger.info("Building precision-coverage curve...")
    points = await scorer.precision_coverage_curve(ignore_grounding=ignore_grounding)

    print(f"\n{'threshold':>10}  {'precision':>10}  {'coverage':>10}")  # noqa: T201
    print("-" * 36)  # noqa: T201
    for pt in points:
        print(f"{pt.threshold:>10.2f}  {pt.precision_approved:>10.4f}  {pt.coverage:>10.4f}")  # noqa: T201
