from __future__ import annotations

import bcrypt
import pytest

from seshat.api.auth import AuthenticationError, verify_api_key


class TestVerifyApiKey:
    def _hash(self, key: str) -> str:
        return bcrypt.hashpw(key.encode(), bcrypt.gensalt(rounds=4)).decode()

    async def test_valid_key_returns_user(self):
        key = "test-key-abc123"
        key_hash = self._hash(key)
        user_id, role = await verify_api_key(key, [(key_hash, "alice", "reviewer")])
        assert user_id == "alice"
        assert role == "reviewer"

    async def test_invalid_key_raises(self):
        key = "test-key-abc123"
        key_hash = self._hash("different-key")
        with pytest.raises(AuthenticationError):
            await verify_api_key(key, [(key_hash, "alice", "reviewer")])

    async def test_empty_store_raises(self):
        with pytest.raises(AuthenticationError):
            await verify_api_key("any-key", [])
