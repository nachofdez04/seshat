from unittest.mock import MagicMock, patch

import pytest

from seshat.config.settings import SeshatConfig
from seshat.secrets.factory import _cached_resolver

_FAKE_DB_URL = "postgresql://user:pass@host/dbname"


@pytest.fixture
def minimal_config() -> SeshatConfig:
    """Minimal SeshatConfig for unit tests.

    Explicitly sets SecretsProvider.ENV and seeds the postgres_url secret so
    the resolve_connection_strings validator succeeds without hitting AWS or
    a real database.
    """
    return SeshatConfig(_env_file=None)  # type: ignore[call-arg]


@pytest.fixture(autouse=True)
def clear_resolver_cache():
    yield
    _cached_resolver.cache_clear()


@pytest.fixture
def mocked_secrets_resolver():
    """Patches _cached_resolver at the source so every factory sees the same mock.

    Yields the mock resolver so tests can assert on get_secret() calls.
    """
    resolver = MagicMock()
    resolver.get_secret.return_value = _FAKE_DB_URL
    with patch("seshat.secrets.factory._cached_resolver", return_value=resolver):
        yield resolver
