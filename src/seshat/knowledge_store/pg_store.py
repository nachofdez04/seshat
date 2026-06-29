"""PostgreSQL-backed knowledge store."""

from __future__ import annotations

import json
import re
import textwrap
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import asyncpg

from seshat.models.enums import GraphDirection, NodeState, RelationshipType
from seshat.models.nodes import KBNode, KBRelationship
from seshat.utils.log import get_logger
from seshat.utils.retry import async_retry

type _Conn = asyncpg.Connection | asyncpg.pool.PoolConnectionProxy

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from seshat.config.settings import KBStoreConfig
    from seshat.models.api_graph import NodeFilter

logger = get_logger(__name__)


def _pg_should_retry(exc: Exception) -> bool:
    if isinstance(exc, asyncpg.PostgresError):
        return isinstance(
            exc, (asyncpg.PostgresConnectionError, asyncpg.TooManyConnectionsError, asyncpg.DeadlockDetectedError)
        )
    return True


_PG_ASYNC_RETRY = async_retry(retryable_exceptions=(Exception,), should_retry=_pg_should_retry)


class PostgresKBStore:
    def __init__(self, config: KBStoreConfig, connection_string: str) -> None:
        self._connection_string = self._validate_connection_string(connection_string)
        self._schema = config.schema_name
        self._pool_min = config.pool_min_size
        self._pool_max = config.pool_max_size
        self._pool: asyncpg.Pool | None = None

    @staticmethod
    def _validate_connection_string(connection_string: str) -> str:
        match = re.match(r"^postgresql(\+\w+)?://", connection_string)
        if match is None:
            raise ValueError("Invalid connection string: must start with 'postgresql://' or 'postgresql+<driver>://'")

        if match.group(1) is not None:
            logger.warning(
                "Connection string contains driver qualifier %r; removing for asyncpg compatibility",
                match.group(1),
            )
            return re.sub(r"postgresql\+\w+://", "postgresql://", connection_string)

        return connection_string

    @_PG_ASYNC_RETRY
    async def connect(self) -> None:
        self._pool = await asyncpg.create_pool(
            self._connection_string,
            min_size=self._pool_min,
            max_size=self._pool_max,
        )
        logger.info("PostgresKBStore pool created (min=%d max=%d)", self._pool_min, self._pool_max)

    async def close(self) -> None:
        if self._pool:
            await self._pool.close()
            logger.debug("PostgresKBStore pool closed")

    @property
    def pool(self) -> asyncpg.Pool:
        if self._pool is None:
            raise RuntimeError("PostgresKBStore.connect() has not been called")
        return self._pool

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[_Conn]:
        """Async context manager that acquires a connection and starts a transaction.

        PGVectorStore uses its own connection pool (langchain-postgres) and cannot join this
        transaction. To keep both stores in sync, perform the vector upsert *inside* the block
        before the KB transaction commits. If the vector upsert raises, the KB write rolls back:

            async with kb_store.transaction() as conn:
                await kb_store.write_node(node, conn=conn)
                await vector_store.upsert(node.id, text, metadata)  # raises → KB rolls back

        The transaction is committed on clean exit and rolled back on exception.
        """
        async with self.pool.acquire() as conn:  # noqa: SIM117
            async with conn.transaction():
                yield conn

    async def write_node(self, node: KBNode, *, conn: _Conn | None = None) -> str:
        """Write a node to the KB.

        Pass conn (from transaction()) when the write must be atomic with a vector upsert.
        Omit it only for standalone writes (tests, seed scripts) where no vector store is involved.

        Relationships must be written separately via write_relationship() once all
        referenced node IDs exist, because kb_relationships has FK constraints on
        both source_id and target_id.

        No ON CONFLICT clause: node_id is a uuid4 generated fresh at extraction time,
        so a PK collision is structurally impossible. Retry paths (POST /jobs re-submit
        or POST /jobs/{id}/retry) both re-run extraction and produce new UUIDs.
        Postgres transaction atomicity ensures no partial state is committed on failure,
        so there is nothing to overwrite. If a future checkpoint-resume feature re-uses
        node IDs from a saved IdentificationResult, revisit this.
        """
        logger.debug("Inserting node with node_id=%s", node.id)

        executor = conn or self.pool
        await executor.execute(
            f"""
            INSERT INTO {self._schema}.kb_nodes
                (node_id, schema_version, type, title, description,
                 confidence, quote_anchors, status, state, metadata, created_at)
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11)
            """,
            *self._node_to_row_args(node, created_at=datetime.now(UTC)),
        )
        return str(node.id)

    async def write_relationship(self, rel: KBRelationship, *, conn: _Conn | None = None) -> None:
        """Write a relationship to the KB.

        Pass conn (from transaction()) when the write must be atomic with other KB writes.
        Omit it only for standalone writes where atomicity is not required.

        No ON CONFLICT clause: composite PK (source_id, target_id, rel_type) inherits
        the same no-collision guarantee as write_node — source_id and target_id are fresh
        UUIDs generated at extraction time and never reused across retry runs.
        """
        logger.debug(
            "Inserting relationship with source_id=%s target_id=%s rel_type=%s",
            rel.source_id,
            rel.target_id,
            rel.rel_type,
        )

        executor = conn or self.pool
        await executor.execute(
            f"""
            INSERT INTO {self._schema}.kb_relationships
                (source_id, target_id, rel_type, job_id, created_at)
            VALUES ($1,$2,$3,$4,$5)
            """,
            str(rel.source_id),
            str(rel.target_id),
            rel.rel_type.value,
            rel.job_id,
            rel.created_at,
        )

    @_PG_ASYNC_RETRY
    async def update_node_state(self, node_id: str, new_state: NodeState, *, conn: _Conn | None = None) -> None:
        logger.debug("Updating node state with node_id=%s to new_state=%s", node_id, new_state)

        executor = conn or self.pool
        result = await executor.execute(
            f"UPDATE {self._schema}.kb_nodes SET state=$1 WHERE node_id=$2",
            new_state.value,
            node_id,
        )
        if result == "UPDATE 0":
            raise KeyError(f"update_node_state: node_id={node_id!r} not found")

    @_PG_ASYNC_RETRY
    async def update_node(self, node: KBNode, *, conn: _Conn | None = None) -> None:
        logger.debug("Updating node with node_id=%s", node.id)

        executor = conn or self.pool
        result = await executor.execute(
            f"""
            UPDATE {self._schema}.kb_nodes
            SET title=$1, description=$2, metadata=$3, schema_version=$4
            WHERE node_id=$5
            """,
            node.title,
            node.description,
            json.dumps(node.metadata.model_dump(mode="json")),
            node.schema_version,
            str(node.id),
        )
        if result == "UPDATE 0":
            raise KeyError(f"update_node: node_id={node.id!r} not found")

    @_PG_ASYNC_RETRY
    async def delete_node(self, node_id: str, *, conn: _Conn | None = None) -> None:
        logger.debug("Deleting node with node_id=%s", node_id)

        executor = conn or self.pool
        await executor.execute(
            f"DELETE FROM {self._schema}.kb_nodes WHERE node_id=$1",
            node_id,
        )

    @_PG_ASYNC_RETRY
    async def count_inbound_relationships(self, node_id: str) -> int:
        """Return the number of relationships where target_id = node_id."""
        row = await self.pool.fetchrow(
            f"SELECT COUNT(*) FROM {self._schema}.kb_relationships WHERE target_id=$1",
            node_id,
        )
        return row[0] if row else 0

    @_PG_ASYNC_RETRY
    async def delete_relationships_for_node(
        self, node_id: str, *, cascade: bool = True, conn: _Conn | None = None
    ) -> None:
        """Delete relationships for node_id.

        cascade=True (default): delete both outbound (source_id) and inbound (target_id) edges.
        cascade=False: delete only outbound (source_id) edges — caller must ensure no inbound
        edges remain before calling delete_node, otherwise the FK constraint will raise.
        """
        logger.debug("Deleting relationships for node_id=%s cascade=%s", node_id, cascade)

        query = f"DELETE FROM {self._schema}.kb_relationships WHERE source_id=$1"
        if cascade:
            query += " OR target_id=$1"

        executor = conn or self.pool
        await executor.execute(query, node_id)

    @_PG_ASYNC_RETRY
    async def get_node(self, node_id: str) -> KBNode | None:
        logger.debug("Fetching node with node_id=%s", node_id)

        row = await self.pool.fetchrow(f"SELECT * FROM {self._schema}.kb_nodes WHERE node_id=$1", node_id)
        if row is None:
            return None

        return self._row_to_node(row)

    @_PG_ASYNC_RETRY
    async def get_neighbours(
        self,
        node_id: str,
        rel_types: list[RelationshipType] | None = None,
        direction: GraphDirection = GraphDirection.BOTH,
    ) -> list[KBNode]:
        if direction == GraphDirection.OUTBOUND:
            join_on = "r.source_id = $1"
            neighbour_col = "r.target_id"
        elif direction == GraphDirection.INBOUND:
            join_on = "r.target_id = $1"
            neighbour_col = "r.source_id"
        else:
            # One query covers both directions: OR matches either end, CASE picks the other end as neighbour.
            # $1 appears three times; PostgreSQL binds the same value each time.
            join_on = "(r.source_id = $1 OR r.target_id = $1)"
            neighbour_col = "CASE WHEN r.source_id = $1 THEN r.target_id ELSE r.source_id END"

        params: list[object] = [node_id]
        rel_filter = ""
        if rel_types:
            placeholders = ", ".join(f"${i + 2}" for i in range(len(rel_types)))
            rel_filter = f"AND r.rel_type IN ({placeholders})"
            params.extend(rt.value for rt in rel_types)

        query = textwrap.dedent(
            f"""
            SELECT DISTINCT n.* FROM {self._schema}.kb_nodes n
            JOIN {self._schema}.kb_relationships r ON {join_on} {rel_filter}
            WHERE n.node_id = {neighbour_col}
            """
        )
        logger.debug("Fetching neighbours with query=%s params=%s", query, params)

        rows = await self.pool.fetch(query, *params)
        return [self._row_to_node(r) for r in rows]

    @_PG_ASYNC_RETRY
    async def query(self, node_filter: NodeFilter) -> list[KBNode]:
        conditions = ["1=1"]
        params: list[object] = []

        def add(col: str, val: object, op: str = "=") -> None:
            # `asyncpg` uses positional placeholders ($1, $2, …) instead of named ones. Thus, we use
            # one append per call; placeholder index is derived from list length at append time.
            params.append(val)
            conditions.append(f"{col} {op} ${len(params)}")

        if node_filter.node_type:
            add("type", node_filter.node_type.value)
        if node_filter.state:
            add("state", node_filter.state.value)
        if node_filter.status:
            add("status", node_filter.status.value)
        if node_filter.min_confidence is not None:
            add("confidence", node_filter.min_confidence, ">=")
        if node_filter.job_id:
            add("metadata->>'job_id'", node_filter.job_id)
        if node_filter.meeting_date_from:
            add("metadata->>'meeting_date'", node_filter.meeting_date_from.isoformat(), ">=")
        if node_filter.meeting_date_to:
            add("metadata->>'meeting_date'", node_filter.meeting_date_to.isoformat(), "<=")
        if node_filter.team:
            add("metadata->>'team'", node_filter.team)
        if node_filter.project:
            add("metadata->>'project'", node_filter.project)
        if node_filter.domain:
            add("metadata->>'domain'", node_filter.domain)
        if node_filter.ingestion_source:
            add("metadata->>'ingestion_source'", node_filter.ingestion_source.value)

        params.append(node_filter.limit)
        params.append(node_filter.offset)
        limit_clause = f"LIMIT ${len(params) - 1} OFFSET ${len(params)}"
        query = textwrap.dedent(
            f"SELECT * FROM {self._schema}.kb_nodes WHERE {' AND '.join(conditions)} {limit_clause}"
        )
        logger.debug("Fetching nodes with query=%s params=%s", query, params)

        rows = await self.pool.fetch(query, *params)
        return [self._row_to_node(r) for r in rows]

    async def paginated_query(self, node_filter: NodeFilter) -> list[KBNode]:
        """Fetch all matching nodes across pages, using node_filter.limit as the page size."""
        results: list[KBNode] = []
        page_size = node_filter.limit
        offset = node_filter.offset
        while True:
            page = await self.query(node_filter.model_copy(update={"offset": offset}))
            results.extend(page)
            if len(page) < page_size:
                break
            offset += page_size
        return results

    @staticmethod
    def _node_to_row_args(node: KBNode, created_at: datetime) -> tuple:
        return (
            str(node.id),
            node.schema_version,
            node.type.value,
            node.title,
            node.description,
            node.confidence,
            json.dumps([anchor.model_dump() for anchor in node.quote_anchors]),
            node.status.value,
            node.state.value,
            json.dumps(node.metadata.model_dump(mode="json")),
            created_at,
        )

    @staticmethod
    def _row_to_node(row: asyncpg.Record) -> KBNode:
        d = dict(row)
        d["id"] = d.pop("node_id")
        d["metadata"] = json.loads(d.pop("metadata"))
        d["quote_anchors"] = json.loads(d.pop("quote_anchors"))
        d.pop("created_at")
        return KBNode.model_validate(d)
