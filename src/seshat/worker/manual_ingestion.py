from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from seshat.models.api_graph import BulkFailure, BulkNodeCreate, BulkNodeDelete, BulkResult
from seshat.models.enums import ApprovalMethod, IngestionSource, NodeState, NodeStatus
from seshat.models.nodes import KBRelationship
from seshat.utils.log import get_logger

if TYPE_CHECKING:
    import asyncpg

    from seshat.knowledge_store.pg_store import PostgresKBStore
    from seshat.models.api_graph import ManualNodeCreate, ManualNodeUpdate, NodeOverride, RelationshipInput
    from seshat.models.nodes import KBNode
    from seshat.pipeline.extraction.orchestrator import ExtractionOrchestrator
    from seshat.vector_store.base_store import AbstractVectorStore

logger = get_logger(__name__)


class NodeNotFoundError(Exception):
    pass


class NodePreconditionError(Exception):
    pass


class ManualIngestionService:
    def __init__(
        self,
        kb_store: PostgresKBStore,
        vector_store: AbstractVectorStore,
        extraction_orch: ExtractionOrchestrator,
    ) -> None:
        self._kb = kb_store
        self._vs = vector_store
        self._extraction_orch = extraction_orch

    async def create(self, payload: ManualNodeCreate, user_id: str) -> KBNode:
        from seshat.models.nodes import KBNode, NodeMetadata

        now = datetime.now(UTC)
        job_id = f"manual_{uuid4()}"

        if payload.source_quote is not None:
            logger.warning("blob-based quote anchors are not yet implemented — ignoring source_quote")

        node = KBNode(
            id=uuid4(),
            schema_version="1.0",
            type=payload.type,
            title=payload.title,
            description=payload.description,
            confidence=1.0,
            quote_anchors=[],
            status=NodeStatus.APPROVED,
            state=NodeState.CURRENT,
            metadata=NodeMetadata(
                job_id=job_id,
                ingestion_source=IngestionSource.MANUAL,
                approval_method=ApprovalMethod.MANUAL,
                approved_by=user_id,
                approved_at=now,
                meeting_date=payload.meeting_date,
                participants=payload.participants,
                team=payload.team,
                project=payload.project,
                domain=payload.domain,
                concept_fields=payload.concept_fields,
            ),
        )

        async with self._kb.transaction() as conn:
            await self._kb.write_node(node, conn=conn)
            if payload.relationships is not None:
                await self._write_relationships(node.id, payload.relationships, now, job_id=job_id, conn=conn)

        await self._vs.upsert(str(node.id), f"{node.title} {node.description}", node.metadata.model_dump(mode="json"))

        return node

    async def bulk_create(self, payload: BulkNodeCreate, user_id: str) -> BulkResult:
        succeeded: list[str] = []
        failed: list[BulkFailure] = []

        for item in payload.nodes:
            try:
                node = await self.create(item, user_id)
                succeeded.append(str(node.id))
            except Exception as exc:
                if payload.on_error == "stop":
                    raise
                failed.append(BulkFailure(node_id=f"<{item.type}:{item.title}>", error=str(exc)))

        return BulkResult(succeeded=succeeded, failed=failed)

    async def delete(self, node_id: str, *, cascade: bool = True) -> None:
        if not cascade:
            n = await self._kb.count_inbound_relationships(node_id)
            if n > 0:
                raise NodePreconditionError(
                    f"Node is referenced as a target by {n} relationships — delete them first or use cascade=true"
                )

        async with self._kb.transaction() as conn:
            await self._kb.delete_relationships_for_node(node_id, cascade=cascade, conn=conn)
            await self._kb.delete_node(node_id, conn=conn)

        await self._vs.delete(node_id)

    async def bulk_delete(self, payload: BulkNodeDelete, *, cascade: bool = True) -> BulkResult:
        succeeded: list[str] = []
        failed: list[BulkFailure] = []

        for node_id in payload.node_ids:
            try:
                await self.delete(node_id, cascade=cascade)
                succeeded.append(node_id)
            except Exception as exc:
                if payload.on_error == "stop":
                    raise
                failed.append(BulkFailure(node_id=node_id, error=str(exc)))

        return BulkResult(succeeded=succeeded, failed=failed)

    async def update(self, node_id: str, payload: ManualNodeUpdate, user_id: str) -> KBNode:
        node = await self._kb.get_node(node_id)
        if node is None:
            raise NodeNotFoundError(node_id)

        if node.metadata.ingestion_source != IngestionSource.MANUAL:
            raise NodePreconditionError(
                "Only manually-created nodes can be edited — use the override endpoint for pipeline nodes"
            )

        return await self._edit(node, payload, user_id)

    async def override(
        self,
        node_id: str,
        payload: NodeOverride,
        user_id: str,
        minimum_method: ApprovalMethod | None,
    ) -> KBNode:
        node = await self._kb.get_node(node_id)
        if node is None:
            raise NodeNotFoundError(node_id)

        if minimum_method is not None and node.metadata.approval_method != minimum_method:
            raise NodePreconditionError("Insufficient role to override this node")

        return await self._edit(node, payload, user_id)

    async def _edit(self, node: KBNode, payload: ManualNodeUpdate, user_id: str) -> KBNode:
        now = datetime.now(UTC)
        job_id = f"manual_{uuid4()}"
        meta_updates: dict = {
            "meeting_date": payload.meeting_date,
            "participants": payload.participants,
            "team": payload.team,
            "project": payload.project,
            "domain": payload.domain,
            "concept_fields": payload.concept_fields,
            "corrected_by": user_id,
            "corrected_at": now,
            "correction_reason": payload.reason,
        }

        updated_node = node._with(
            title=payload.title,
            description=payload.description,
            metadata=node.metadata._with(**meta_updates),
        )

        async with self._kb.transaction() as conn:
            await self._kb.update_node(updated_node, conn=conn)
            if payload.relationships is not None:
                await self._kb.delete_relationships_for_node(str(node.id), cascade=False, conn=conn)
                await self._write_relationships(node.id, payload.relationships, now, job_id=job_id, conn=conn)

        await self._vs.upsert(
            str(updated_node.id),
            f"{updated_node.title} {updated_node.description}",
            updated_node.metadata.model_dump(mode="json"),
        )

        return updated_node

    async def resolve(self, nodes: list[KBNode], job_id: str) -> list[KBRelationship]:
        """Run resolution for the given approved nodes and persist the resulting relationships."""
        result = await self._extraction_orch.run_resolution(job_id=job_id, approved=nodes)

        async with self._kb.transaction() as conn:
            for rel in result.relationships:
                await self._kb.write_relationship(rel, conn=conn)

        return result.relationships

    async def _write_relationships(
        self,
        source_id: UUID,
        relationships: list[RelationshipInput],
        now: datetime,
        *,
        job_id: str,
        conn: asyncpg.Connection | asyncpg.pool.PoolConnectionProxy,
    ) -> None:
        for r in relationships:
            rel = KBRelationship(
                source_id=source_id,
                target_id=UUID(r.target_id),
                rel_type=r.rel_type,
                job_id=job_id,
                created_at=now,
            )
            await self._kb.write_relationship(rel, conn=conn)
