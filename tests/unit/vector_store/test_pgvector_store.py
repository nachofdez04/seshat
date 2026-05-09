from __future__ import annotations

import logging

import pytest

from seshat.vector_store.pgvector_store import PGVectorStore


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
        with caplog.at_level(logging.WARNING, logger="seshat.vector_store.pgvector_store"):
            PGVectorStore._validate_connection_string("postgresql+asyncpg://user:pass@host/db")
        assert "+asyncpg" in caplog.text

    def test_plain_scheme_no_warning(self, caplog):
        with caplog.at_level(logging.WARNING, logger="seshat.vector_store.pgvector_store"):
            PGVectorStore._validate_connection_string("postgresql://user:pass@host/db")
        assert caplog.text == ""

    def test_invalid_scheme_raises(self):
        with pytest.raises(ValueError, match="Invalid connection string"):
            PGVectorStore._validate_connection_string("mysql://user:pass@host/db")

    def test_error_message_does_not_contain_credentials(self):
        with pytest.raises(ValueError, match="Invalid connection string") as exc_info:
            PGVectorStore._validate_connection_string("mysql://secret:hunter2@host/db")
        assert "secret" not in str(exc_info.value)
        assert "hunter2" not in str(exc_info.value)
