from __future__ import annotations

from pathlib import Path

from seshat.agents.grounding import GroundingAgent
from seshat.agents.identification.grouping import GroupingAgent
from seshat.agents.identification.registry import IdentificationAgentRegistry
from seshat.agents.resolution.registry import ResolutionRegistry
from seshat.config.eval_settings import EvalConfig
from seshat.config.settings import ExtractionConfig, ObservabilityConfig
from seshat.eval.grounding.runner import GroundingEvalRunner
from seshat.eval.grouping.runner import GroupingEvalRunner
from seshat.eval.identification.runner import IdentificationEvalRunner
from seshat.eval.resolution.runner import ResolutionEvalRunner
from seshat.pipeline.extraction.orchestrator import ExtractionOrchestrator
from tests.integration.helpers import (
    cheap_grounding_config,
    cheap_identification_config,
    cheap_resolution_config,
    make_cheap_llm,
)

CORPUS_BASE_DIR = Path(__file__).parent.parent.parent.parent / "data" / "eval" / "test_corpora"


class _NoopBlobStore:
    async def put(self, key: str, data: bytes) -> None:
        raise NotImplementedError("BlobStore.put called during eval")

    async def get(self, key: str) -> bytes:
        raise NotImplementedError("BlobStore.get called during eval")

    async def exists(self, key: str) -> bool:
        raise NotImplementedError("BlobStore.exists called during eval")


class _NoopKBStore:
    async def query(self, *args, **kwargs):
        raise NotImplementedError("KBStore.query called during eval")

    async def write_node(self, *args, **kwargs):
        raise NotImplementedError("KBStore.write_node called during eval")

    async def close(self) -> None:
        pass


def _make_eval_orchestrator(extraction_config: ExtractionConfig) -> ExtractionOrchestrator:
    llm = make_cheap_llm()
    return ExtractionOrchestrator(
        config=extraction_config,
        identification_registry=IdentificationAgentRegistry(llm, extraction_config),
        resolution_registry=ResolutionRegistry(llm, extraction_config.resolution),
        node_retriever=None,  # type: ignore[arg-type]
        kb_store=_NoopKBStore(),  # type: ignore[arg-type]
        blob_store=_NoopBlobStore(),  # type: ignore[arg-type]
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


def make_eval_config(tmp_path: Path, experiment_name: str = "seshat-eval-test") -> EvalConfig:
    return EvalConfig(
        corpus_base_dir=CORPUS_BASE_DIR,
        gate_path=tmp_path / "eval_gate.json",
        observability=ObservabilityConfig(
            mlflow_tracking_uri="sqlite:///" + str(tmp_path / "mlflow.db"),
            mlflow_experiment_name=experiment_name,
        ),
    )
