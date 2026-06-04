from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import TYPE_CHECKING

from seshat.agents.resolution.cross_type.registry import CrossTypeResolutionRegistry
from seshat.agents.resolution.same_type.registry import SameTypeResolutionRegistry
from seshat.models.enums import RelationshipType
from seshat.utils.log import get_logger

if TYPE_CHECKING:
    from uuid import UUID

    from langchain_core.language_models import BaseChatModel

    from seshat.agents.resolution.base import ResolvedRelationship
    from seshat.config.settings import ResolutionLLMConfig
    from seshat.models.nodes import FailedResolutionSource, KBNode

logger = get_logger(__name__)

# Relationships that are never valid when the target node has been superseded in the same run.
# BLOCKS is safe to strip here because it flows from a structurally different agent type (risk/action_item)
# than the SUPERSEDES that triggers the strip (decision). Independent agents can't produce a spurious
# SUPERSEDES on the same target, so the strip is reliable.
# CONFLICTS_WITH and AMENDS are NOT included: they are same-type relationships, and a spurious SUPERSEDES
# from one source can cascade into stripping a valid CONFLICTS_WITH/AMENDS from another source on the
# same target. Add them here only if same-type spurious-supersedes false positives become negligible.
_INVALID_ON_SUPERSEDED = {
    RelationshipType.BLOCKS,
}


class ResolutionRegistry:
    def __init__(self, llm: BaseChatModel, config: ResolutionLLMConfig) -> None:
        self._same_type = SameTypeResolutionRegistry(llm, config)
        self._cross_type = CrossTypeResolutionRegistry(llm, config)

    async def resolve_all(
        self,
        source_nodes: list[KBNode],
        per_source_targets: dict[UUID, list[KBNode]],
        semaphore: asyncio.Semaphore | None = None,
    ) -> tuple[list[ResolvedRelationship], list[FailedResolutionSource]]:
        (same_rels, same_failed), (cross_rels, cross_failed) = await asyncio.gather(
            self._same_type.resolve_all(source_nodes, per_source_targets, semaphore),
            self._cross_type.resolve_all(source_nodes, per_source_targets, semaphore),
        )
        all_rels = self._strip_invalid_on_superseded(same_rels + cross_rels)
        return all_rels, same_failed + cross_failed

    @staticmethod
    def _strip_invalid_on_superseded(
        relationships: list[ResolvedRelationship],
    ) -> list[ResolvedRelationship]:
        superseded_ids = {rel.target_id for rel in relationships if rel.rel_type == RelationshipType.SUPERSEDES}
        if not superseded_ids:
            return relationships

        stripped_counts: defaultdict[str, int] = defaultdict(int)
        filtered = []
        for rel in relationships:
            if rel.rel_type in _INVALID_ON_SUPERSEDED and rel.target_id in superseded_ids:
                stripped_counts[rel.rel_type] += 1
            else:
                filtered.append(rel)

        for rel_type, count in stripped_counts.items():
            logger.info("Stripped %d spurious %r relationship(s) targeting superseded nodes", count, rel_type)

        return filtered
