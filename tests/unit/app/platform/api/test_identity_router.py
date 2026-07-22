from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import bcrypt

from seshat.app.platform.api.state import AppState
from seshat.core.models.enums import UserRole
from tests.unit.app.platform.api.conftest import make_current_user


def _make_app_state() -> AppState:
    return AppState(
        config=MagicMock(),
        admin_service=MagicMock(),
        health_service=MagicMock(),
        graph_service=MagicMock(),
        job_service=MagicMock(),
        document_service=MagicMock(),
        publishing_service=MagicMock(),
    )


def _hash(key: str) -> str:
    return bcrypt.hashpw(key.encode(), bcrypt.gensalt(rounds=4)).decode()


class TestMe:
    async def test_returns_user_identity(self, api_client):
        user = make_current_user(user_id="alice", role=UserRole.OPERATOR)
        async with api_client(_make_app_state(), user) as ac:
            resp = await ac.get("/me")
        assert resp.status_code == 200
        assert resp.json()["user_id"] == "alice"
        assert resp.json()["role"] == "operator"

    async def test_requires_auth(self, api_client):
        async with api_client(_make_app_state()) as ac:
            resp = await ac.get("/me")
        assert resp.status_code == 401

    async def test_corrupt_stored_hash_returns_401_not_500(self, api_client):
        state = _make_app_state()
        state.admin_service.get_api_keys = AsyncMock(return_value=[("not-a-valid-bcrypt-hash", "alice", "viewer")])
        async with api_client(state) as ac:
            resp = await ac.get("/me", headers={"X-API-Key": "some-key"})
        assert resp.status_code == 401

    async def test_invalid_stored_role_returns_401_not_500(self, api_client):
        key = "test-key"
        state = _make_app_state()
        state.admin_service.get_api_keys = AsyncMock(return_value=[(_hash(key), "alice", "superuser")])
        async with api_client(state) as ac:
            resp = await ac.get("/me", headers={"X-API-Key": key})
        assert resp.status_code == 401
