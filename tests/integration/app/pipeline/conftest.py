import pytest_asyncio

from seshat.core.config.settings import KBStoreConfig
from seshat.infra.knowledge_store.pg_store import PostgresKBStore


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def kb_store(pg_test_url):
    store = PostgresKBStore(KBStoreConfig(), pg_test_url)
    await store.connect()
    yield store
    await store.close()


@pytest_asyncio.fixture(autouse=True, loop_scope="module")
async def _truncate_kb_tables(kb_store):
    yield
    await kb_store.pool.execute(f"TRUNCATE {kb_store._schema}.kb_relationships, {kb_store._schema}.kb_nodes CASCADE")
