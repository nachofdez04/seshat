from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from seshat.agents.resolution.same_type.action_item import ActionItemResolutionAgent
from seshat.agents.resolution.same_type.decision import DecisionResolutionAgent
from seshat.agents.resolution.same_type.open_question import OpenQuestionResolutionAgent
from seshat.agents.resolution.same_type.reflective import ReflectiveResolutionAgent
from seshat.agents.resolution.same_type.risk import RiskResolutionAgent
from seshat.models.enums import ConceptType
from seshat.utils.log import get_logger

logger = get_logger(__name__)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from uuid import UUID

    from langchain_core.language_models import BaseChatModel

    from seshat.agents.resolution.base import BaseSameTypeResolutionAgent, ResolvedRelationship, _BaseResolutionAgent
    from seshat.config.settings import ExtractionConfig
    from seshat.models.nodes import FailedResolutionSource, KBNode


class SameTypeResolutionRegistry:
    def __init__(self, llm: BaseChatModel, config: ExtractionConfig, review_llm: BaseChatModel | None = None) -> None:
        self._agents: dict[ConceptType, _BaseResolutionAgent] = {
            concept_type: _make_agent(agent_cls, llm, config, review_llm)
            for concept_type, agent_cls in (
                (ConceptType.ACTION_ITEM, ActionItemResolutionAgent),
                (ConceptType.DECISION, DecisionResolutionAgent),
                (ConceptType.OPEN_QUESTION, OpenQuestionResolutionAgent),
                (ConceptType.RISK, RiskResolutionAgent),
            )
        }

    def get(self, concept_type: ConceptType) -> _BaseResolutionAgent:
        agent = self._agents.get(concept_type)
        if agent is None:
            raise KeyError(f"No resolution agent registered for {concept_type}")
        return agent

    async def resolve_all(
        self,
        source_nodes: list[KBNode],
        per_source_targets: dict[UUID, list[KBNode]],
        global_sem: asyncio.Semaphore | None = None,
    ) -> tuple[list[ResolvedRelationship], list[FailedResolutionSource]]:
        """Run one agent per active concept type concurrently.

        return_exceptions=True gives partial results: one type failing doesn't abort the others.
        """
        sources_by_type: dict[ConceptType, list[KBNode]] = {}
        for node in source_nodes:
            sources_by_type.setdefault(node.type, []).append(node)

        concept_types, tasks = [], []
        for ct, sources, scoped in self._iter_active_types(sources_by_type, per_source_targets):
            concept_types.append(ct)
            tasks.append(self._agents[ct].resolve(sources, scoped, global_sem))

        if not tasks:
            return [], []

        results = await asyncio.gather(*tasks, return_exceptions=True)
        resolved: list[ResolvedRelationship] = []
        failed: list[FailedResolutionSource] = []
        for ct, result in zip(concept_types, results, strict=True):
            if isinstance(result, Exception):
                logger.error("Same-type resolution failed for %s: %s", ct, result)
                continue

            assert isinstance(result, tuple)
            rels, fails = result
            resolved.extend(rels)
            failed.extend(fails)
        return resolved, failed

    def _iter_active_types(
        self,
        sources_by_type: dict[ConceptType, list[KBNode]],
        per_source_targets: dict[UUID, list[KBNode]],
    ) -> Iterator[tuple[ConceptType, list[KBNode], dict[UUID, list[KBNode]]]]:
        """Skip input types with no registered agent —
        defensive guard against nodes whose type predates or outlives a registered agent.
        """
        for ct, sources in sources_by_type.items():
            if ct not in self._agents:
                continue
            yield ct, sources, _scope_targets(sources, per_source_targets, ct)


def _make_agent(
    agent_cls: type[BaseSameTypeResolutionAgent],
    llm: BaseChatModel,
    config: ExtractionConfig,
    review_llm: BaseChatModel | None,
) -> _BaseResolutionAgent:
    inner = agent_cls(llm=llm, config=config.resolution)
    if not config.resolution_self_review.enabled:
        return inner

    logger.debug("Using Reflective%s", agent_cls.__name__)
    return ReflectiveResolutionAgent(inner=inner, review_llm=review_llm or llm)


def _scope_targets(
    sources: list[KBNode],
    per_source_targets: dict[UUID, list[KBNode]],
    target_type: ConceptType,
) -> dict[UUID, list[KBNode]]:
    """Filter per_source_targets down to the single type this agent handles.

    per_source_targets is built from the full KB and contains all target types mixed together;
    each agent only reasons about one type at a time, so passing the unfiltered map would expose
    it to nodes it has no prompt for.
    """
    targets = {}
    for src in sources:
        scoped = [t for t in per_source_targets.get(src.id, []) if t.type == target_type]
        targets[src.id] = scoped
    return targets
