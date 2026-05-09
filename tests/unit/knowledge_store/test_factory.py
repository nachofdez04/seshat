import pytest

from seshat.knowledge_store.factory import get_kb_store
from seshat.knowledge_store.pg_store import PostgresKBStore
from tests.unit.conftest import _FAKE_DB_URL


@pytest.mark.usefixtures("mocked_secrets_resolver")
class TestGetKBStore:
    def test_returns_postgres_kb_store(self, minimal_config):
        store = get_kb_store(minimal_config)
        assert isinstance(store, PostgresKBStore)

    def test_resolves_connection_string_from_secret_key(self, minimal_config, mocked_secrets_resolver):
        get_kb_store(minimal_config)
        mocked_secrets_resolver.get_secret.assert_called_once_with(minimal_config.kb_store.connection_secret_key)

    def test_propagates_connection_string(self, minimal_config):
        store = get_kb_store(minimal_config)
        assert store._connection_string == store._validate_connection_string(_FAKE_DB_URL)
