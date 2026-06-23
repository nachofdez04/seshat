from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import TYPE_CHECKING

from seshat.agents.resolution.cross_type.registry import CrossTypeResolutionRegistry
from seshat.agents.resolution.same_type.registry import SameTypeResolutionRegistry
from seshat.models.enums import ConceptType, RelationshipType
from seshat.utils.hashing import fingerprint
from seshat.utils.log import get_logger

if TYPE_CHECKING:
    from uuid import UUID

    from langchain_core.language_models import BaseChatModel

    from seshat.agents.resolution.base import ResolvedRelationship
    from seshat.config.settings import ExtractionConfig
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
    def __init__(self, llm: BaseChatModel, config: ExtractionConfig, review_llm: BaseChatModel | None = None) -> None:
        self._same_type = SameTypeResolutionRegistry(llm, config, review_llm=review_llm)
        self._cross_type = CrossTypeResolutionRegistry(llm, config)

    def fingerprint(self) -> str:
        """8-char hex digest of all same-type and cross-type agent prompts.

        Uses agent.fingerprint() so the validate prompt is included when agents are
        wrapped in ReflectiveResolutionAgent, giving shallow and reflective runs distinct cache keys.
        """
        combined = "".join(agent.fingerprint() for agent in self._same_type._agents.values())
        combined += "".join(agent.fingerprint() for agent in self._cross_type._agents_mapping.values())
        return fingerprint(combined)

    def prompt_texts(self) -> dict[str, str]:
        texts = {}
        for concept_type, agent in self._same_type._agents.items():
            for prompt_key, prompt_text in agent.prompt_texts().items():
                texts[f"same_type-{concept_type}-{prompt_key}"] = prompt_text
        texts.update(
            {
                f"cross_type-{src}-to-{tgt}-{prompt_key}": prompt_text
                for (src, tgt), agent in self._cross_type._agents_mapping.items()
                for prompt_key, prompt_text in agent.prompt_texts().items()
            }
        )
        return texts

    def fingerprint_for_types(self, source_types: set[ConceptType], target_types: set[ConceptType]) -> str:
        """8-char hex digest of only the agents that fire for the given source/target type sets.

        Same-type agents fire when a type appears in both sets; cross-type agents fire for each
        (src, tgt) pair where src is in source_types and tgt is in target_types.
        """
        prompts: list[str] = []
        # sorted for deterministic hash — set iteration order is not guaranteed
        for ct in sorted(source_types & target_types, key=lambda c: c.value):
            if ct in self._same_type._agents:
                prompts.append(self._same_type._agents[ct].fingerprint())
        for (src, tgt), agent in self._cross_type._agents_mapping.items():
            if src in source_types and tgt in target_types:
                prompts.append(agent.fingerprint())
        return fingerprint("".join(prompts))

    async def resolve_all(
        self,
        source_nodes: list[KBNode],
        per_source_targets: dict[UUID, list[KBNode]],
        semaphore: asyncio.Semaphore | None = None,
    ) -> tuple[list[ResolvedRelationship], list[FailedResolutionSource]]:
        """Run same-type and cross-type resolution in parallel and merge results.

        Same-type agents fire when a type appears in both sets; cross-type agents fire for each
        (src, tgt) pair where src is in source_types and tgt is in target_types.
        Relationships targeting superseded nodes are stripped before returning.
        """
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
