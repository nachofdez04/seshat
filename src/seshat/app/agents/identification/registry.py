from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from seshat.app.agents.identification.action_item import ActionItemIdentificationAgent
from seshat.app.agents.identification.decision import DecisionIdentificationAgent
from seshat.app.agents.identification.open_question import OpenQuestionIdentificationAgent
from seshat.app.agents.identification.reflective import ReflectiveIdentificationAgent
from seshat.app.agents.identification.risk import RiskIdentificationAgent
from seshat.core.models.enums import ConceptType
from seshat.core.utils.hashing import fingerprint
from seshat.core.utils.log import get_logger

if TYPE_CHECKING:
    from collections.abc import Iterable

    from langchain_core.language_models import BaseChatModel

    from seshat.app.agents.identification.base import AnchoredConcept, _BaseIdentificationAgent
    from seshat.app.agents.identification.grouping import ConceptGroup
    from seshat.core.config.settings import ExtractionConfig

    # Heterogeneous across concept types (each agent yields a different concrete model)
    RawConcepts = list[AnchoredConcept[Any]] | list[ConceptGroup[Any]]

logger = get_logger(__name__)


class IdentificationRegistry:
    def __init__(self, llm: BaseChatModel, config: ExtractionConfig, review_llm: BaseChatModel | None = None) -> None:
        self._agents: dict[ConceptType, _BaseIdentificationAgent] = {
            concept_type: _make_agent(agent_cls, llm, config, review_llm)
            for concept_type, agent_cls in (
                (ConceptType.ACTION_ITEM, ActionItemIdentificationAgent),
                (ConceptType.DECISION, DecisionIdentificationAgent),
                (ConceptType.OPEN_QUESTION, OpenQuestionIdentificationAgent),
                (ConceptType.RISK, RiskIdentificationAgent),
            )
        }

    def get(self, concept_type: ConceptType) -> _BaseIdentificationAgent:
        agent = self._agents.get(concept_type)
        if agent is None:
            raise KeyError(f"No agent registered for {concept_type}")
        return agent

    async def run_all(
        self,
        transcript: str,
        transcript_file: str,
        hints: dict[ConceptType, str],
        concept_types: Iterable[ConceptType] | None = None,
    ) -> tuple[dict[ConceptType, RawConcepts], list[ConceptType]]:
        """Fan out identification across the given concept types concurrently.

        Defaults to every registered type. return_exceptions=True gives partial results:
        one type failing doesn't abort the others. Returns the raw per-type concepts keyed
        by type plus the list of types whose agent raised.
        """
        types = list(concept_types) if concept_types is not None else list(self._agents)
        outcomes = await asyncio.gather(
            *(self.get(ct).identify(transcript, hints.get(ct, ""), transcript_file) for ct in types),
            return_exceptions=True,
        )

        results: dict[ConceptType, RawConcepts] = {}
        failed: list[ConceptType] = []
        for ct, outcome in zip(types, outcomes, strict=True):
            if isinstance(outcome, Exception):
                logger.error("Identification failed for %s: %s", ct, outcome)
                failed.append(ct)
                continue

            assert isinstance(outcome, list)
            results[ct] = outcome
        return results, failed

    def fingerprint(self) -> str:
        """8-char hex digest of all agents' fingerprints concatenated.

        All four concept types always fire per example (parallel fan-out in the orchestrator),
        so any prompt change (including the validate prompt) busts the full identification cache.
        """
        combined = "".join(agent.fingerprint() for agent in self._agents.values())
        return fingerprint(combined)

    def prompt_texts(self) -> dict[str, str]:
        texts = {}
        for concept_type, agent in self._agents.items():
            for prompt_type, prompt in agent.prompt_texts().items():
                texts[f"{concept_type}_{prompt_type}"] = prompt
        return texts


def _make_agent(
    agent_cls: type[_BaseIdentificationAgent],
    llm: BaseChatModel,
    config: ExtractionConfig,
    review_llm: BaseChatModel | None,
) -> _BaseIdentificationAgent:
    inner = agent_cls(
        llm=llm, config=config.identification, grouped_identification_types=config.grouped_identification_types
    )
    if not config.identification_self_review.enabled:
        return inner

    logger.debug("Using Reflective%s", agent_cls.__name__)
    return ReflectiveIdentificationAgent(inner=inner, review_llm=review_llm or llm)
