from __future__ import annotations

from pathlib import Path

from seshat.app.agents.grounding import GroundingAgent
from seshat.app.agents.identification.grouping import GroupingAgent
from seshat.app.agents.identification.registry import IdentificationRegistry
from seshat.app.agents.resolution.registry import ResolutionRegistry
from seshat.app.pipeline.extraction.orchestrator import ExtractionOrchestrator
from seshat.core.config.eval_settings import EvalConfig
from seshat.core.config.settings import ExtractionConfig
from seshat.eval.grounding.runner import GroundingEvalRunner
from seshat.eval.grouping.runner import GroupingEvalRunner
from seshat.eval.identification.runner import IdentificationEvalRunner
from seshat.eval.resolution.runner import ResolutionEvalRunner
from tests.integration.helpers import (
    cheap_grounding_config,
    cheap_identification_config,
    cheap_resolution_config,
    make_cheap_llm,
)

CORPUS_BASE_DIR = Path(__file__).parent.parent.parent.parent / "data" / "eval" / "test_corpora"


def _make_eval_orchestrator(extraction_config: ExtractionConfig) -> ExtractionOrchestrator:
    llm = make_cheap_llm()
    return ExtractionOrchestrator(
        config=extraction_config,
        identification_registry=IdentificationRegistry(llm, extraction_config),
        resolution_registry=ResolutionRegistry(llm, extraction_config),
        node_retriever=None,  # type: ignore[arg-type]
        node_repo=None,  # type: ignore[arg-type]
        blob_repo=None,  # type: ignore[arg-type]
    )


def make_identification_runner(config: EvalConfig) -> IdentificationEvalRunner:
    id_config = cheap_identification_config()
    res_config = cheap_resolution_config()
    # Grouping is disabled: identification eval measures extraction only, not the downstream grouping step.
    extraction_config = ExtractionConfig(
        identification=id_config, resolution=res_config, grouped_identification_types=set()
    )
    return IdentificationEvalRunner(orchestrator=_make_eval_orchestrator(extraction_config), config=config)


def make_resolution_runner(config: EvalConfig) -> ResolutionEvalRunner:
    id_config = cheap_identification_config()
    res_config = cheap_resolution_config()
    extraction_config = ExtractionConfig(identification=id_config, resolution=res_config)
    return ResolutionEvalRunner(orchestrator=_make_eval_orchestrator(extraction_config), config=config)


def make_grounding_runner(config: EvalConfig) -> GroundingEvalRunner:
    grounding_config = cheap_grounding_config()
    agent = GroundingAgent(llm=make_cheap_llm(), config=grounding_config)
    return GroundingEvalRunner(agent=agent, config=config)


def make_grouping_runner(config: EvalConfig) -> GroupingEvalRunner:
    id_config = cheap_identification_config()
    agent = GroupingAgent(llm=make_cheap_llm(), config=id_config)
    return GroupingEvalRunner(agent=agent, config=config)


def make_identification_meta_scorer(config: EvalConfig):
    from seshat.eval.calibration.identification_meta_scorer import IdentificationMetaScorer

    id_config = cheap_identification_config()
    res_config = cheap_resolution_config()
    extraction_config = ExtractionConfig(
        identification=id_config, resolution=res_config, grouped_identification_types=set()
    )
    return IdentificationMetaScorer(orchestrator=_make_eval_orchestrator(extraction_config), config=config, step=0.1)


def make_eval_config(tmp_path: Path) -> EvalConfig:
    return EvalConfig(
        corpus_base_dir=CORPUS_BASE_DIR,
        gate_path=tmp_path / "eval_gate.json",
    )
