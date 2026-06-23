from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from seshat.agents.resolution.cross_type.action_item import ActionItemCrossTypeResolutionAgent
from seshat.agents.resolution.cross_type.decision import DecisionCrossTypeResolutionAgent
from seshat.agents.resolution.cross_type.open_question import OpenQuestionCrossTypeResolutionAgent
from seshat.agents.resolution.cross_type.risk import RiskCrossTypeResolutionAgent
from seshat.agents.resolution.same_type.registry import _scope_targets
from seshat.models.enums import ConceptType
from seshat.utils.log import get_logger

logger = get_logger(__name__)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from uuid import UUID

    from langchain_core.language_models import BaseChatModel

    from seshat.agents.resolution.base import BaseCrossTypeResolutionAgent, ResolvedRelationship, _BaseResolutionAgent
    from seshat.config.settings import ExtractionConfig
    from seshat.models.nodes import FailedResolutionSource, KBNode


class CrossTypeResolutionRegistry:
    def __init__(self, llm: BaseChatModel, config: ExtractionConfig) -> None:
        self._agents_mapping: dict[tuple[ConceptType, ConceptType], _BaseResolutionAgent] = {
            (source_type, target_type): _make_agent(agent_cls, target_type, llm, config)
            for (source_type, target_type), agent_cls in (
                ((ConceptType.DECISION, ConceptType.RISK), DecisionCrossTypeResolutionAgent),
                ((ConceptType.DECISION, ConceptType.OPEN_QUESTION), DecisionCrossTypeResolutionAgent),
                ((ConceptType.DECISION, ConceptType.ACTION_ITEM), DecisionCrossTypeResolutionAgent),
                ((ConceptType.RISK, ConceptType.DECISION), RiskCrossTypeResolutionAgent),
                ((ConceptType.RISK, ConceptType.OPEN_QUESTION), RiskCrossTypeResolutionAgent),
                ((ConceptType.RISK, ConceptType.ACTION_ITEM), RiskCrossTypeResolutionAgent),
                ((ConceptType.OPEN_QUESTION, ConceptType.DECISION), OpenQuestionCrossTypeResolutionAgent),
                ((ConceptType.OPEN_QUESTION, ConceptType.ACTION_ITEM), OpenQuestionCrossTypeResolutionAgent),
                ((ConceptType.ACTION_ITEM, ConceptType.RISK), ActionItemCrossTypeResolutionAgent),
            )
        }

    def get(self, src_type: ConceptType, tgt_type: ConceptType) -> _BaseResolutionAgent:
        agent = self._agents_mapping.get((src_type, tgt_type))
        if agent is None:
            raise KeyError(f"No cross-type agent registered for ({src_type}, {tgt_type})")
        return agent

    async def resolve_all(
        self,
        source_nodes: list[KBNode],
        per_source_targets: dict[UUID, list[KBNode]],
        global_sem: asyncio.Semaphore | None = None,
    ) -> tuple[list[ResolvedRelationship], list[FailedResolutionSource]]:
        """Fan-out: one concurrent task per (source type, target type) combination."""
        sources_by_type: dict[ConceptType, list[KBNode]] = {}
        for node in source_nodes:
            sources_by_type.setdefault(node.type, []).append(node)

        pairs, tasks = [], []
        for pair, sources, scoped in self._iter_active_pairs(sources_by_type, per_source_targets):
            pairs.append(pair)
            tasks.append(self._agents_mapping[pair].resolve(sources, scoped, global_sem))

        if not tasks:
            return [], []

        results = await asyncio.gather(*tasks, return_exceptions=True)
        resolved: list[ResolvedRelationship] = []
        failed: list[FailedResolutionSource] = []
        for (src_type, tgt_type), result in zip(pairs, results, strict=True):
            if isinstance(result, Exception):
                logger.error("Cross-type resolution failed for (%s, %s): %s", src_type, tgt_type, result)
                continue

            assert isinstance(result, tuple)
            rels, fails = result
            resolved.extend(rels)
            failed.extend(fails)
        return resolved, failed

    def _iter_active_pairs(
        self,
        sources_by_type: dict[ConceptType, list[KBNode]],
        per_source_targets: dict[UUID, list[KBNode]],
    ) -> Iterator[tuple[tuple[ConceptType, ConceptType], list[KBNode], dict[UUID, list[KBNode]]]]:
        """Skip pairs where no source has any target of the right type —
        avoids spawning an LLM task with an empty candidate list.
        """
        for src_type, tgt_type in self._agents_mapping:
            sources = sources_by_type.get(src_type, [])
            if not sources:
                continue
            scoped = _scope_targets(sources, per_source_targets, tgt_type)
            if any(scoped.values()):
                yield (src_type, tgt_type), sources, scoped


def _make_agent(
    agent_cls: type[BaseCrossTypeResolutionAgent],
    target_type: ConceptType,
    llm: BaseChatModel,
    config: ExtractionConfig,
) -> _BaseResolutionAgent:
    return agent_cls(llm=llm, config=config.resolution, target_type=target_type)
