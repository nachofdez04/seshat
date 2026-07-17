"""Integration test: VS state metadata stays in sync with KB state transitions."""

from __future__ import annotations

import pytest
import sqlalchemy as sa
from langchain_core.embeddings.fake import DeterministicFakeEmbedding

from seshat.app.repositories.node_repository import NodeRepository
from seshat.core.config.settings import KBStoreConfig, VectorIndexConfig, VectorStoreConfig
from seshat.core.models.enums import NodeState, RelationshipType
from seshat.infra.knowledge_store.pg_store import PostgresKBStore
from seshat.infra.vector_store.pgvector_store import PGVectorStore
from tests.helpers import make_node
from tests.integration.conftest import SKIP_IF_NO_POSTGRES
from tests.integration.helpers import make_relationship

pytestmark = [pytest.mark.integration, SKIP_IF_NO_POSTGRES]

_COLLECTION = "test_state_sync"


@pytest.fixture
async def kb_store(pg_test_url):
    store = PostgresKBStore(KBStoreConfig(), pg_test_url)
    await store.connect()
    yield store
    await store.pool.execute(f"TRUNCATE {store._schema}.kb_relationships, {store._schema}.kb_nodes CASCADE")
    await store.close()


@pytest.fixture
async def vs(pg_test_url):
    index = VectorIndexConfig(collection=_COLLECTION)
    store = PGVectorStore(VectorStoreConfig(), index, DeterministicFakeEmbedding(size=1536), pg_test_url)
    await store._store.__apost_init__()
    yield store
    await store._store.adelete_collection()


@pytest.fixture
def repo(kb_store, vs):
    return NodeRepository(kb_store, vs)


async def _get_vs_state(vs: PGVectorStore, node_id: str) -> str | None:
    stmt = sa.select(vs._store.EmbeddingStore.cmetadata["state"].as_string()).where(
        vs._store.EmbeddingStore.cmetadata["node_id"].as_string() == node_id
    )
    async with vs._engine.connect() as conn:
        result = await conn.execute(stmt)
        row = result.fetchone()

    return row[0] if row else None


class TestNodeRepositoryStateSync:
    async def test_approve_writes_state_current_to_vs(self, repo, vs):
        node = make_node("sync-node-1", title="Initial decision")
        await repo.write_node(node)

        assert await _get_vs_state(vs, str(node.id)) == NodeState.CURRENT

    async def test_supersedes_relationship_sets_vs_state_superseded(self, repo, vs):
        target = make_node("sync-target", title="Old decision")
        await repo.write_node(target)

        superseder = make_node("sync-superseder", title="New decision")
        await repo.write_node(superseder)

        rel = make_relationship(superseder, target, rel_type=RelationshipType.SUPERSEDES)
        await repo.create_relationship_manual(rel)

        assert await _get_vs_state(vs, str(target.id)) == NodeState.SUPERSEDED

    async def test_delete_relationship_reverts_vs_state_to_current(self, repo, vs):
        target = make_node("sync-revert-target", title="Reverted decision")
        await repo.write_node(target)

        superseder = make_node("sync-revert-superseder", title="Superseding decision")
        await repo.write_node(superseder)

        rel = make_relationship(superseder, target, rel_type=RelationshipType.SUPERSEDES)
        rel = await repo.create_relationship_manual(rel)
        assert await _get_vs_state(vs, str(target.id)) == NodeState.SUPERSEDED

        await repo.delete_relationship(rel)
        assert await _get_vs_state(vs, str(target.id)) == NodeState.CURRENT
