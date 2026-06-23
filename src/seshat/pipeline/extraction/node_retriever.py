from __future__ import annotations

from typing import TYPE_CHECKING

from seshat.models.api import NodeFilter
from seshat.models.enums import GraphDirection
from seshat.pipeline.extraction.pending_node import _quote_text
from seshat.utils.log import get_logger
from seshat.utils.retry import async_retry

if TYPE_CHECKING:
    from seshat.config.settings import RAGConfig
    from seshat.knowledge_store.pg_store import PostgresKBStore
    from seshat.models.api import SearchResult
    from seshat.models.nodes import KBNode
    from seshat.vector_store.base_store import AbstractVectorStore

logger = get_logger(__name__)


class NodeRetriever:
    def __init__(
        self,
        rag_config: RAGConfig,
        kb_store: PostgresKBStore,
        vector_store: AbstractVectorStore,
    ) -> None:
        self._config = rag_config
        self._kb = kb_store
        self._vs = vector_store

    @property
    def max_concurrent_retrievals(self) -> int:
        return self._config.max_concurrent_retrievals

    async def retrieve(
        self,
        node: KBNode,
        transcript: str,
        *,
        node_filter: NodeFilter | None = None,
        exclude_job_id: str | None = None,
    ) -> list[KBNode]:
        # Vector store only holds approved+current nodes — status/state are enforced
        # by only upserting on approval and deleting on archival (TODO: implement delete hook).
        filter_kwargs: dict = {"node_type": node.type}
        if node_filter is not None:
            filter_kwargs.update(node_filter.model_dump(exclude_unset=True))

        source_quote = _quote_text(node.quote_anchors, transcript)
        # Truncate source_quote to avoid it dominating the embedding centroid: title+description
        # carry the semantic signal; the quote adds speaker context but must not outweigh them.
        query = f"{node.title} {node.description} {source_quote[:80]}".strip()
        logger.debug("Retrieving targets for node id=%s type=%s", node.id, node.type.value)

        results = await self._vector_search(
            query, node_filter=NodeFilter(**filter_kwargs), exclude_job_id=exclude_job_id
        )
        logger.debug("Vector search returned %d results for node id=%s", len(results), node.id)

        cap = self._config.top_k * 2
        token_budget = self._config.max_context_tokens
        node_id = str(node.id)
        seen: dict[str, KBNode] = {}
        tokens_used = 0

        # fetch actual nodes from vector search results.
        # we fetch sequentially to allow early exit once cap or token budget is reached;
        # parallel gather would fetch all unconditionally and waste KB calls
        for result in results:
            if len(seen) >= cap or tokens_used >= token_budget:
                break

            if result.node_id == node_id:
                continue

            kb_node = await self._get_node(result.node_id)
            if kb_node is not None:
                seen[result.node_id] = kb_node
                tokens_used += (len(kb_node.title) + len(kb_node.description)) // 4

        # if we have fewer than top_k results, traverse neighbours of retrieved nodes to fill up targets (up to cap)
        for result in results:
            if len(seen) >= cap:
                break

            if result.node_id not in seen:
                continue

            for neighbour in await self._get_neighbours(result.node_id):
                if len(seen) >= cap:
                    break

                neighbour_id = str(neighbour.id)
                if neighbour_id != node_id and neighbour_id not in seen:
                    seen[neighbour_id] = neighbour

        targets = list(seen.values())
        logger.debug("target retrieval done: %d targets for node id=%s", len(targets), node.id)
        return targets

    @async_retry()
    async def _vector_search(
        self, query: str, node_filter: NodeFilter, *, exclude_job_id: str | None
    ) -> list[SearchResult]:
        return await self._vs.search(
            query,
            top_k=self._config.top_k,
            node_filter=node_filter,
            exclude_job_id=exclude_job_id,
            score_threshold=self._config.min_score,
        )

    @async_retry()
    async def _get_node(self, node_id: str) -> KBNode | None:
        return await self._kb.get_node(node_id)

    @async_retry()
    async def _get_neighbours(self, node_id: str) -> list[KBNode]:
        return await self._kb.get_neighbours(
            node_id,
            rel_types=self._config.traversal_rel_types,
            direction=GraphDirection.BOTH,
        )
