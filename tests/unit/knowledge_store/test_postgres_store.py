from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock

import pytest

from seshat.knowledge_store.pg_store import PostgresKBStore
from seshat.models.api_graph import NodeFilter
from seshat.models.enums import (
    ConceptType,
    NodeState,
)
from tests.helpers import make_node as _make_node

if TYPE_CHECKING:
    from seshat.config.settings import SeshatConfig


@pytest.fixture
def store(minimal_config: SeshatConfig) -> PostgresKBStore:
    kb_store_config = minimal_config.kb_store
    connection_string = "postgresql+asyncpg://user:pass@host/dbname"
    return PostgresKBStore(kb_store_config, connection_string=connection_string)


class TestNodeToRowArgs:
    def test_columns(self):
        node = _make_node()
        created_at = datetime(2026, 4, 21, 12, 0, tzinfo=UTC)
        row = PostgresKBStore._node_to_row_args(node, created_at)
        assert row[0] == str(node.id)
        assert row[2] == "decision"
        assert row[7] == "approved"
        assert row[8] == "current"
        assert row[10] == created_at

    def test_metadata_is_json(self):
        node = _make_node()
        row = PostgresKBStore._node_to_row_args(node, datetime.now(UTC))
        meta = json.loads(row[9])
        assert meta["job_id"] == "job-1"


class TestRowToNode:
    def test_roundtrip(self):
        node = _make_node()
        row = {
            "node_id": node.id,
            "schema_version": node.schema_version,
            "type": node.type.value,
            "title": node.title,
            "description": node.description,
            "confidence": node.confidence,
            "quote_anchors": json.dumps([anchor.model_dump() for anchor in node.quote_anchors]),
            "status": node.status.value,
            "state": node.state.value,
            "metadata": json.dumps(node.metadata.model_dump(mode="json")),
            "created_at": datetime(2026, 4, 21, 12, 0, tzinfo=UTC),
        }
        restored = PostgresKBStore._row_to_node(row)  # type: ignore[arg-type]
        assert restored.id == node.id
        assert restored.type == node.type
        assert restored.state == NodeState.CURRENT
        assert restored.metadata.job_id == "job-1"
        assert restored.type == ConceptType.DECISION


class TestPool:
    def test_raises_before_connect(self, store: PostgresKBStore):
        with pytest.raises(RuntimeError, match="connect"):
            _ = store.pool


class TestValidateConnectionString:
    def test_plain_postgresql_scheme_accepted(self):
        result = PostgresKBStore._validate_connection_string("postgresql://user:pass@host/db")
        assert result == "postgresql://user:pass@host/db"

    def test_driver_qualifier_stripped(self):
        result = PostgresKBStore._validate_connection_string("postgresql+asyncpg://user:pass@host/db")
        assert result == "postgresql://user:pass@host/db"

    def test_driver_qualifier_stripped_logs_warning(self, caplog):
        with caplog.at_level(logging.WARNING, logger="seshat.knowledge_store.pg_store"):
            PostgresKBStore._validate_connection_string("postgresql+asyncpg://user:pass@host/db")
        assert "+asyncpg" in caplog.text

    def test_invalid_scheme_raises(self):
        with pytest.raises(ValueError, match="Invalid connection string"):
            PostgresKBStore._validate_connection_string("mysql://user:pass@host/db")

    def test_error_message_does_not_contain_credentials(self):
        with pytest.raises(ValueError, match="Invalid connection string") as exc_info:
            PostgresKBStore._validate_connection_string("mysql://secret:hunter2@host/db")
        assert "secret" not in str(exc_info.value)
        assert "hunter2" not in str(exc_info.value)


class TestPaginatedQuery:
    async def test_paginates_across_multiple_pages(self, store: PostgresKBStore):
        page_size = 10
        page_one = [_make_node(f"n{i}") for i in range(page_size)]
        page_two = [_make_node(f"n{i}") for i in range(page_size, page_size + 3)]
        store.query = AsyncMock(side_effect=[page_one, page_two])

        result = await store.paginated_query(NodeFilter(limit=page_size))

        assert len(result) == page_size + 3
        assert store.query.call_count == 2

    async def test_fetches_extra_page_when_last_page_is_full(self, store: PostgresKBStore):
        # A full last page is indistinguishable from a page with more data behind it,
        # so the loop makes one extra call (returning empty) to confirm termination.
        page_size = 5
        full_page = [_make_node(f"n{i}") for i in range(page_size)]
        store.query = AsyncMock(side_effect=[full_page, []])

        result = await store.paginated_query(NodeFilter(limit=page_size))

        assert len(result) == page_size
        assert store.query.call_count == 2
