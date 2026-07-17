from __future__ import annotations

import logging
from unittest.mock import AsyncMock

import pytest

from seshat.core.models.api_graph import NodeFilter
from seshat.core.models.enums import SearchMode
from seshat.infra.vector_store.pgvector_store import PGVectorStore
from tests.unit.infra.helpers import assert_credentials_not_in_error, assert_invalid_scheme_raises

_N1 = "00000000-0000-0000-0000-000000000001"


class TestSearchModeGuard:
    async def test_hybrid_mode_raises(self):
        store = PGVectorStore.__new__(PGVectorStore)
        with pytest.raises(ValueError, match="HYBRID"):
            await store.search("q", top_k=5, mode=SearchMode.HYBRID)

    async def test_agent_mode_raises(self):
        store = PGVectorStore.__new__(PGVectorStore)
        with pytest.raises(ValueError, match="AGENT"):
            await store.search("q", top_k=5, mode=SearchMode.AGENT)


class TestSparseSearchGuard:
    async def test_empty_query_returns_empty(self):
        store = PGVectorStore.__new__(PGVectorStore)
        result = await store._sparse_search("   ", top_k=5, node_filter=None, exclude_job_id=None)
        assert result == []

    async def test_missing_collection_propagates_from_sparse_search(self):
        store = PGVectorStore.__new__(PGVectorStore)
        store._ts_content_ready = True
        store._collection_id = None
        store._get_collection_id = AsyncMock(
            side_effect=RuntimeError("Collection 'seshat_kb' not found in langchain_pg_collection")
        )
        store._ensure_ts_content = AsyncMock()

        with pytest.raises(RuntimeError, match="seshat_kb"):
            await store._sparse_search("budget approval", top_k=5, node_filter=None, exclude_job_id=None)


class TestValidateConnectionString:
    def test_psycopg_qualifier_accepted_unchanged(self):
        result = PGVectorStore._validate_connection_string("postgresql+psycopg://user:pass@host/db")
        assert result == "postgresql+psycopg://user:pass@host/db"

    def test_plain_scheme_gets_psycopg_qualifier(self):
        result = PGVectorStore._validate_connection_string("postgresql://user:pass@host/db")
        assert result == "postgresql+psycopg://user:pass@host/db"

    def test_wrong_qualifier_replaced(self):
        result = PGVectorStore._validate_connection_string("postgresql+asyncpg://user:pass@host/db")
        assert result == "postgresql+psycopg://user:pass@host/db"

    def test_wrong_qualifier_logs_warning(self, caplog):
        with caplog.at_level(logging.WARNING, logger="seshat.infra.vector_store.pgvector_store"):
            PGVectorStore._validate_connection_string("postgresql+asyncpg://user:pass@host/db")
        assert "+asyncpg" in caplog.text

    def test_plain_scheme_no_warning(self, caplog):
        with caplog.at_level(logging.WARNING, logger="seshat.infra.vector_store.pgvector_store"):
            PGVectorStore._validate_connection_string("postgresql://user:pass@host/db")
        assert caplog.text == ""

    def test_invalid_scheme_raises(self):
        assert_invalid_scheme_raises(PGVectorStore)

    def test_error_message_does_not_contain_credentials(self):
        assert_credentials_not_in_error(PGVectorStore)

    def test_psycopg2_qualifier_replaced(self):
        result = PGVectorStore._validate_connection_string("postgresql+psycopg2://user:pass@host/db")
        assert result == "postgresql+psycopg://user:pass@host/db"


class TestBuildSemanticFilter:
    def _store(self) -> PGVectorStore:
        return PGVectorStore.__new__(PGVectorStore)

    def test_none_filter_and_no_exclude_returns_none(self):
        assert self._store()._build_semantic_filter(None) is None

    def test_supported_node_type_filter_applied(self):
        from seshat.core.models.enums import ConceptType

        nf = NodeFilter(node_type=ConceptType.DECISION)
        result = self._store()._build_semantic_filter(nf)
        assert result == {"node_type": ConceptType.DECISION.value}

    def test_supported_min_confidence_filter_applied(self):
        nf = NodeFilter(min_confidence=0.7)
        result = self._store()._build_semantic_filter(nf)
        assert result == {"confidence": {"$gte": 0.7}}

    def test_unsupported_fields_warn_and_are_ignored(self, caplog):
        from seshat.core.models.enums import NodeStatus

        nf = NodeFilter(status=NodeStatus.APPROVED)
        with caplog.at_level(logging.WARNING, logger="seshat.infra.vector_store.pgvector_store"):
            result = self._store()._build_semantic_filter(nf)

        assert "status" in caplog.text
        assert "supported" in caplog.text
        assert result == {}

    def test_unsupported_fields_do_not_prevent_supported_fields_from_applying(self, caplog):
        from seshat.core.models.enums import ConceptType, NodeStatus

        nf = NodeFilter(node_type=ConceptType.DECISION, status=NodeStatus.APPROVED)
        with caplog.at_level(logging.WARNING, logger="seshat.infra.vector_store.pgvector_store"):
            result = self._store()._build_semantic_filter(nf)

        assert result == {"node_type": ConceptType.DECISION.value}
        assert "status" in caplog.text

    def test_exclude_job_id_adds_ne_filter(self):
        result = self._store()._build_semantic_filter(None, exclude_job_id="job-123")
        assert result == {"job_id": {"$ne": "job-123"}}

    def test_no_active_filters_returns_empty_dict_not_none(self):
        # node_filter set but all supported fields are None → returns {} (not None)
        nf = NodeFilter()
        result = self._store()._build_semantic_filter(nf)
        assert result == {}


class TestUpdateMetadata:
    async def test_executes_jsonb_merge_update(self):
        from unittest.mock import AsyncMock, MagicMock, patch

        store = PGVectorStore.__new__(PGVectorStore)

        embedding_store = MagicMock()
        embedding_store.cmetadata = MagicMock()
        embedding_store.cmetadata.op = MagicMock(return_value=MagicMock(return_value=MagicMock()))

        fake_conn = AsyncMock()
        fake_conn.__aenter__ = AsyncMock(return_value=fake_conn)
        fake_conn.__aexit__ = AsyncMock(return_value=None)
        fake_conn.execute = AsyncMock()

        fake_engine = MagicMock()
        fake_engine.begin = MagicMock(return_value=fake_conn)

        inner_store = MagicMock()
        inner_store.EmbeddingStore = embedding_store
        inner_store._async_engine = fake_engine

        store._store = inner_store

        with patch("sqlalchemy.update") as mock_update:
            mock_update.return_value.where.return_value.values.return_value = MagicMock()
            await store.update_metadata("node-123", {"state": "superseded"})

        mock_update.assert_called_once_with(embedding_store)
        fake_conn.execute.assert_awaited_once()
