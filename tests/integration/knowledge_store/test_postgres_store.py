from datetime import UTC, date, datetime
from uuid import uuid4

import asyncpg
import pytest

from seshat.config.settings import KBStoreConfig
from seshat.knowledge_store.pg_store import PostgresKBStore
from seshat.models.api import NodeFilter
from seshat.models.enums import (
    ConceptType,
    GraphDirection,
    IngestionSource,
    NodeState,
    RelationshipType,
)
from seshat.models.nodes import KBRelationship
from tests.helpers import make_node as _make_node
from tests.integration.conftest import SKIP_IF_NO_POSTGRES

pytestmark = [pytest.mark.integration, SKIP_IF_NO_POSTGRES]


@pytest.fixture
async def store(pg_test_url):
    config = KBStoreConfig()
    s = PostgresKBStore(config, pg_test_url)
    await s.connect()
    yield s
    await s.pool.execute(f"TRUNCATE {s._schema}.kb_relationships, {s._schema}.kb_nodes CASCADE")
    await s.close()


def _rel(src, tgt, rel_type=RelationshipType.SUPERSEDES, job_id="job-1"):
    return KBRelationship(
        source_id=src.id,
        target_id=tgt.id,
        rel_type=rel_type,
        job_id=job_id,
        created_at=datetime.now(UTC),
    )


class TestWriteAndGet:
    async def test_write_then_get(self, store: PostgresKBStore):
        node = _make_node("n1")
        await store.write_node(node)
        fetched = await store.get_node(str(node.id))

        assert fetched is not None
        assert fetched.title == "Use PostgreSQL"

    async def test_get_nonexistent_returns_none(self, store: PostgresKBStore):
        result = await store.get_node("00000000-0000-0000-0000-000000000000")
        assert result is None

    async def test_duplicate_write_raises(self, store: PostgresKBStore):
        node = _make_node("n-dup-write")
        await store.write_node(node)
        with pytest.raises(asyncpg.UniqueViolationError):
            await store.write_node(node)


class TestUpdateNodeState:
    async def test_state_transition(self, store: PostgresKBStore):
        node = _make_node("n2")
        await store.write_node(node)
        await store.update_node_state(str(node.id), NodeState.SUPERSEDED)
        fetched = await store.get_node(str(node.id))

        assert fetched is not None
        assert fetched.state == NodeState.SUPERSEDED

    async def test_missing_node_raises(self, store: PostgresKBStore):
        with pytest.raises(KeyError, match="not found"):
            await store.update_node_state("00000000-0000-0000-0000-000000000000", NodeState.SUPERSEDED)

    async def test_update_state_inside_transaction(self, store: PostgresKBStore):
        node = _make_node("n-tx-state")
        await store.write_node(node)
        async with store.transaction() as conn:
            await store.update_node_state(str(node.id), NodeState.SUPERSEDED, conn=conn)
        fetched = await store.get_node(str(node.id))
        assert fetched is not None
        assert fetched.state == NodeState.SUPERSEDED


class TestWriteRelationship:
    async def test_write_relationship(self, store: PostgresKBStore):
        n1 = _make_node("n3")
        n2 = _make_node("n4", "Use Redis")
        await store.write_node(n1)
        await store.write_node(n2)
        await store.write_relationship(_rel(n1, n2))
        neighbours = await store.get_neighbours(str(n1.id), direction=GraphDirection.OUTBOUND)
        assert any(n.id == n2.id for n in neighbours)

    async def test_missing_source_raises_fk_error(self, store: PostgresKBStore):
        ghost = _make_node("n-ghost-src")
        real = _make_node("n-real-tgt-fk", "Real Node")
        await store.write_node(real)
        with pytest.raises(asyncpg.ForeignKeyViolationError):
            await store.write_relationship(_rel(ghost, real))

    async def test_missing_target_raises_fk_error(self, store: PostgresKBStore):
        real = _make_node("n-real-src-fk", "Real Source")
        ghost = _make_node("n-ghost-tgt")
        await store.write_node(real)
        with pytest.raises(asyncpg.ForeignKeyViolationError):
            await store.write_relationship(_rel(real, ghost))

    async def test_duplicate_rel_type_raises(self, store: PostgresKBStore):
        src = _make_node("n-dup-rel-src")
        tgt = _make_node("n-dup-rel-tgt", "Dup Rel Target")
        await store.write_node(src)
        await store.write_node(tgt)
        await store.write_relationship(_rel(src, tgt))
        with pytest.raises(asyncpg.UniqueViolationError):
            await store.write_relationship(_rel(src, tgt))


class TestGetNeighboursDirection:
    async def _seed(self, store: PostgresKBStore, src_label: str, tgt_label: str):
        n_src = _make_node(src_label)
        n_tgt = _make_node(tgt_label, "Target Node")
        await store.write_node(n_src)
        await store.write_node(n_tgt)
        await store.write_relationship(_rel(n_src, n_tgt))
        return n_src, n_tgt

    async def test_outbound_returns_target_only(self, store: PostgresKBStore):
        src, tgt = await self._seed(store, "n-dir-src-1", "n-dir-tgt-1")
        outbound = await store.get_neighbours(str(src.id), direction=GraphDirection.OUTBOUND)
        assert any(n.id == tgt.id for n in outbound)
        assert not any(n.id == src.id for n in outbound)

    async def test_inbound_returns_source_only(self, store: PostgresKBStore):
        src, tgt = await self._seed(store, "n-dir-src-2", "n-dir-tgt-2")
        inbound = await store.get_neighbours(str(tgt.id), direction=GraphDirection.INBOUND)
        assert any(n.id == src.id for n in inbound)
        assert not any(n.id == tgt.id for n in inbound)

    async def test_both_returns_all_neighbours(self, store: PostgresKBStore):
        src, tgt = await self._seed(store, "n-dir-src-3", "n-dir-tgt-3")
        both = await store.get_neighbours(str(src.id), direction=GraphDirection.BOTH)
        assert any(n.id == tgt.id for n in both)

    async def test_both_with_rel_types_matching(self, store: PostgresKBStore):
        src, tgt = await self._seed(store, "n-dir-src-4", "n-dir-tgt-4")
        both = await store.get_neighbours(
            str(src.id), rel_types=[RelationshipType.SUPERSEDES], direction=GraphDirection.BOTH
        )
        assert any(n.id == tgt.id for n in both)

    async def test_both_with_rel_types_nonmatching(self, store: PostgresKBStore):
        src, _ = await self._seed(store, "n-dir-src-5", "n-dir-tgt-5")
        both = await store.get_neighbours(
            str(src.id), rel_types=[RelationshipType.AMENDS], direction=GraphDirection.BOTH
        )
        assert both == []

    async def test_inbound_with_rel_type_filter_matching(self, store: PostgresKBStore):
        src, tgt = await self._seed(store, "n-dir-src-6", "n-dir-tgt-6")
        inbound = await store.get_neighbours(
            str(tgt.id), rel_types=[RelationshipType.SUPERSEDES], direction=GraphDirection.INBOUND
        )
        assert any(n.id == src.id for n in inbound)

    async def test_inbound_with_rel_type_filter_nonmatching(self, store: PostgresKBStore):
        _, tgt = await self._seed(store, "n-dir-src-7", "n-dir-tgt-7")
        inbound = await store.get_neighbours(
            str(tgt.id), rel_types=[RelationshipType.AMENDS], direction=GraphDirection.INBOUND
        )
        assert inbound == []

    async def test_isolated_node_returns_empty(self, store: PostgresKBStore):
        node = _make_node("n-isolated")
        await store.write_node(node)
        assert await store.get_neighbours(str(node.id), direction=GraphDirection.OUTBOUND) == []
        assert await store.get_neighbours(str(node.id), direction=GraphDirection.INBOUND) == []
        assert await store.get_neighbours(str(node.id), direction=GraphDirection.BOTH) == []

    async def test_multiple_rel_types_no_duplicates(self, store: PostgresKBStore):
        src = _make_node("n-dedup-src")
        tgt = _make_node("n-dedup-tgt", "Dedup Target")
        await store.write_node(src)
        await store.write_node(tgt)
        for rel_type in (RelationshipType.SUPERSEDES, RelationshipType.AMENDS):
            await store.write_relationship(_rel(src, tgt, rel_type=rel_type))
        neighbours = await store.get_neighbours(str(src.id), direction=GraphDirection.OUTBOUND)
        assert len([n for n in neighbours if n.id == tgt.id]) == 1


class TestQuery:
    async def test_query_by_type(self, store: PostgresKBStore):
        node = _make_node("n5")
        await store.write_node(node)
        results = await store.query(NodeFilter(node_type=ConceptType.ADR))
        assert any(n.id == node.id for n in results)

    async def test_query_min_confidence(self, store: PostgresKBStore):
        node = _make_node("n6")
        await store.write_node(node)
        results = await store.query(NodeFilter(min_confidence=0.95))
        assert not any(n.id == node.id for n in results)

    async def test_query_min_confidence_exact_boundary(self, store: PostgresKBStore):
        node = _make_node("n-conf-boundary", confidence=0.9)
        await store.write_node(node)
        results = await store.query(NodeFilter(min_confidence=0.9))
        assert any(n.id == node.id for n in results)

    async def test_query_by_state(self, store: PostgresKBStore):
        node = _make_node("n-state-q")
        await store.write_node(node)
        current = await store.query(NodeFilter(state=NodeState.CURRENT))
        assert any(n.id == node.id for n in current)
        superseded = await store.query(NodeFilter(state=NodeState.SUPERSEDED))
        assert not any(n.id == node.id for n in superseded)

    async def test_query_meeting_date_range(self, store: PostgresKBStore):
        node = _make_node("n-date-q")
        await store.write_node(node)
        in_range = await store.query(NodeFilter(meeting_date_from=date(2026, 4, 1), meeting_date_to=date(2026, 4, 30)))
        assert any(n.id == node.id for n in in_range)
        out_of_range = await store.query(
            NodeFilter(meeting_date_from=date(2026, 3, 1), meeting_date_to=date(2026, 3, 31))
        )
        assert not any(n.id == node.id for n in out_of_range)

    async def test_query_by_job_id(self, store: PostgresKBStore):
        node = _make_node("n-job-q")
        await store.write_node(node)
        results = await store.query(NodeFilter(job_id="job-1"))
        assert any(n.id == node.id for n in results)
        results_miss = await store.query(NodeFilter(job_id="job-999"))
        assert not any(n.id == node.id for n in results_miss)

    async def test_query_by_ingestion_source(self, store: PostgresKBStore):
        node = _make_node("n-source-q")
        await store.write_node(node)
        results = await store.query(NodeFilter(ingestion_source=IngestionSource.JOB))
        assert any(n.id == node.id for n in results)
        results_miss = await store.query(NodeFilter(ingestion_source=IngestionSource.INIT))
        assert not any(n.id == node.id for n in results_miss)

    async def test_query_by_team(self, store: PostgresKBStore):
        node = _make_node("n-team-q", team="platform")
        await store.write_node(node)
        results = await store.query(NodeFilter(team="platform"))
        assert any(n.id == node.id for n in results)
        results_miss = await store.query(NodeFilter(team="other-team"))
        assert not any(n.id == node.id for n in results_miss)

    async def test_query_combined_filters(self, store: PostgresKBStore):
        node = _make_node("n-combined-q")
        await store.write_node(node)
        results = await store.query(NodeFilter(node_type=ConceptType.ADR, state=NodeState.CURRENT))
        assert any(n.id == node.id for n in results)
        results_miss = await store.query(NodeFilter(node_type=ConceptType.ADR, state=NodeState.SUPERSEDED))
        assert not any(n.id == node.id for n in results_miss)

    async def test_query_no_match_returns_empty(self, store: PostgresKBStore):
        results = await store.query(NodeFilter(job_id=f"nonexistent-job-{uuid4()}"))
        assert results == []


class TestTransaction:
    async def test_transaction_commit(self, store: PostgresKBStore):
        node = _make_node("n-tx-1")
        async with store.transaction() as conn:
            await store.write_node(node, conn=conn)

        fetched = await store.get_node(str(node.id))
        assert fetched is not None

    async def test_transaction_rollback(self, store: PostgresKBStore):
        node = _make_node("n-tx-2")
        try:
            async with store.transaction() as conn:
                await store.write_node(node, conn=conn)
                raise RuntimeError("forced rollback")
        except RuntimeError:
            pass

        fetched = await store.get_node(str(node.id))
        assert fetched is None

    async def test_transaction_rollback_includes_relationship(self, store: PostgresKBStore):
        src = _make_node("n-tx-rel-src")
        tgt = _make_node("n-tx-rel-tgt", "Tx Rel Target")
        await store.write_node(src)
        await store.write_node(tgt)
        try:
            async with store.transaction() as conn:
                await store.write_relationship(_rel(src, tgt), conn=conn)
                raise RuntimeError("forced rollback")
        except RuntimeError:
            pass

        neighbours = await store.get_neighbours(str(src.id), direction=GraphDirection.OUTBOUND)
        assert neighbours == []
