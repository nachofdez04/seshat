from __future__ import annotations

from typing import TYPE_CHECKING

from seshat.core.models.api_graph import NodeFilter
from seshat.core.models.enums import GraphDirection, NodeState
from seshat.core.utils.log import get_logger
from seshat.core.utils.tokens import count_tokens

if TYPE_CHECKING:
    from uuid import UUID

    from seshat.app.pipeline.extraction.reranker import AbstractReranker
    from seshat.app.pipeline.extraction.search_engine import SearchEngine
    from seshat.app.repositories.node_repository import NodeRepository
    from seshat.core.config.settings import RAGConfig
    from seshat.core.models.api_graph import SearchResult
    from seshat.core.models.nodes import KBNode

logger = get_logger(__name__)


class NodeRetriever:
    def __init__(
        self,
        rag_config: RAGConfig,
        node_repo: NodeRepository,
        search_engine: SearchEngine,
        reranker: AbstractReranker | None = None,
    ) -> None:
        self._config = rag_config
        self._repo = node_repo
        self._search_engine = search_engine
        self._reranker = reranker

    @property
    def max_concurrent_retrievals(self) -> int:
        return self._config.max_concurrent_retrievals

    @property
    def node_retrieval_cap(self) -> int:
        return self._config.top_k * 2  # doubled to leave room for neighbour-expansion below

    async def retrieve(
        self,
        node: KBNode,
        *,
        node_filter: NodeFilter | None = None,
        exclude_job_id: str | None = None,
    ) -> list[KBNode]:
        filter_kwargs: dict = {"node_type": node.type, "state": NodeState.CURRENT}
        if node_filter is not None:
            filter_kwargs.update(node_filter.model_dump(exclude_unset=True))

        query = node.vector_store_text
        logger.debug("Retrieving targets for node id=%s type=%s", node.id, node.type.value)

        results = await self._search_engine.search(
            query,
            node_filter=NodeFilter(**filter_kwargs),
            exclude_job_id=exclude_job_id,
            score_threshold=self._config.min_similarity_score,
        )
        logger.info("Search returned %d raw results for node id=%s", len(results), node.id)

        budget = _ContextBudget(self._config.max_context_tokens)
        seen: dict[UUID, KBNode] = {}

        # TOCONSIDER: retrieve direct hits in parallel: faster but potential wasted KB calls on nodes we'd discard
        await self._fetch_direct_hits(seen, results, node.id, budget)
        # TOCONSIDER: retrieved neighbours in parallel, re-rerank them and take top-k.
        await self._expand_with_neighbours(seen, results, node.id, budget)

        if self._reranker is not None and seen:
            logger.debug("reranking %d nodes for node id=%s", len(seen), node.id)
            reranked = await self._reranker.rerank(query, list(seen.values()))
            seen = {n.id: n for n in reranked}
            logger.debug("after rerank: %d nodes for node id=%s", len(seen), node.id)

        targets = list(seen.values())
        logger.info("target retrieval done: %d targets for node id=%s", len(targets), node.id)
        return targets

    async def _fetch_direct_hits(
        self,
        seen: dict[UUID, KBNode],
        results: list[SearchResult],
        node_id: UUID,
        budget: _ContextBudget,
    ) -> None:
        # Sequential fetch to allow early exit on node cap or budget;
        # parallel gather would waste KB calls on nodes we'd discard.
        for result in results:
            if result.node_id in seen:
                logger.warning("Duplicate node id=%s found in vector search results; skipping", result.node_id)
                continue

            if result.node_id == node_id:
                continue

            if len(seen) >= self.node_retrieval_cap or budget.exhausted:
                break

            kb_node = await self._repo.get_node(result.node_id)
            if kb_node is None:
                logger.warning("Node id=%s found in vector search but missing from KB store", result.node_id)
                continue

            if not budget.consume(kb_node):
                logger.debug("Context budget exhausted; stopping at %d direct hits", len(seen))
                break

            seen[result.node_id] = kb_node

    async def _expand_with_neighbours(
        self,
        seen: dict[UUID, KBNode],
        results: list[SearchResult],
        node_id: UUID,
        budget: _ContextBudget,
    ) -> None:
        for result in results:
            if len(seen) >= self.node_retrieval_cap or budget.exhausted:
                logger.debug(
                    "Skipping neighbour expansion for node id=%s: cap=%d seen=%d budget_exhausted=%s",
                    result.node_id,
                    self.node_retrieval_cap,
                    len(seen),
                    budget.exhausted,
                )
                break

            if result.node_id not in seen:
                continue

            neighbours = await self._repo.get_neighbours(
                result.node_id, rel_types=self._config.traversal_rel_types, direction=GraphDirection.BOTH
            )
            for neighbour in neighbours:
                # we are not checking if budget is already exhausted, since the neighbours
                # are already in memory and `consume` allows slight overage before rejecting
                if len(seen) >= self.node_retrieval_cap:
                    break

                if neighbour.id == node_id or neighbour.id in seen:
                    continue

                if not budget.consume(neighbour):
                    # check if there is any "cheaper" neighbour that would fit in the remaining budget,
                    # instead of just skipping all remaining neighbours once we hit the first expensive one
                    continue

                seen[neighbour.id] = neighbour


class _ContextBudget:
    _OVERAGE = 1.1  # allow up to 10% over the soft cap before rejecting a node

    def __init__(self, max_tokens: int) -> None:
        self._soft_limit = max_tokens
        self._hard_limit = int(max_tokens * self._OVERAGE)
        self._used = 0

    @property
    def exhausted(self) -> bool:
        """True once the soft cap is reached — used to skip further KB fetches."""
        return self._used >= self._soft_limit

    def consume(self, node: KBNode) -> bool:
        """Deduct token cost of node. Returns False (and does not deduct) if it would exceed the hard limit."""
        cost = count_tokens(node.vector_store_text)
        if self._used + cost > self._hard_limit:
            return False
        self._used += cost
        return True
