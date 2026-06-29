from __future__ import annotations

from typing import TYPE_CHECKING

from seshat.models.enums import NodeState, NodeStatus, RelationshipType
from seshat.utils.log import get_logger

if TYPE_CHECKING:
    from uuid import UUID

    from seshat.knowledge_store.pg_store import PostgresKBStore, _Conn
    from seshat.models.nodes import ExtractionResult, KBNode, KBRelationship
    from seshat.vector_store.base_store import AbstractVectorStore

logger = get_logger(__name__)

_STATE_TRANSITIONS = {
    RelationshipType.SUPERSEDES: NodeState.SUPERSEDED,
    RelationshipType.AMENDS: NodeState.AMENDED,
}


class WritingStage:
    def __init__(self, kb_store: PostgresKBStore, vector_store: AbstractVectorStore) -> None:
        self._kb = kb_store
        self._vs = vector_store

    async def write(self, result: ExtractionResult) -> int:
        async with self._kb.transaction() as conn:
            await self._update_existing_nodes(result.relationships, conn)
            approved_ids = await self._write_nodes_and_relationships(result.nodes, result.relationships, conn)

        await self._upsert_vectors(result.nodes, approved_ids)

        return len(approved_ids)

    async def _update_existing_nodes(self, relationships: list[KBRelationship], conn: _Conn) -> None:
        for rel in relationships:
            new_state = _STATE_TRANSITIONS.get(rel.rel_type)
            if not new_state:
                continue

            try:
                await self._kb.update_node_state(str(rel.target_id), new_state, conn=conn)
                logger.info(
                    "Node %s state → %s (due to %s from %s)",
                    rel.target_id,
                    new_state,
                    rel.rel_type,
                    rel.source_id,
                )
            except KeyError:
                logger.warning(
                    "update_node_state: target node %s not found (rel_type=%s, source=%s) — skipping",
                    rel.target_id,
                    rel.rel_type,
                    rel.source_id,
                )

    async def _write_nodes_and_relationships(
        self, nodes: list[KBNode], relationships: list[KBRelationship], conn: _Conn
    ) -> set[UUID]:
        all_ids: set[UUID] = {n.id for n in nodes}
        approved_ids: set[UUID] = set()

        for node in nodes:
            if node.status != NodeStatus.APPROVED:
                continue

            await self._kb.write_node(node, conn=conn)
            approved_ids.add(node.id)

        for rel in relationships:
            target_ok = rel.target_id in approved_ids or rel.target_id not in all_ids
            if rel.source_id in approved_ids and target_ok:
                await self._kb.write_relationship(rel, conn=conn)

        return approved_ids

    async def _upsert_vectors(self, nodes: list[KBNode], approved_ids: set[UUID]) -> None:
        for node in nodes:
            if node.id not in approved_ids:
                continue

            text = f"{node.title} {node.description}"
            await self._vs.upsert(str(node.id), text, node.metadata.model_dump(mode="json"))
