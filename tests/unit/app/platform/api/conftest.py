from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from seshat.app.platform.api.app import create_app
from seshat.app.platform.api.dependencies import CurrentUser, _get_current_user, get_app_state
from seshat.app.platform.api.state import AppState
from seshat.core.models.enums import UserRole

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

    from fastapi import FastAPI


def make_app_state(**overrides) -> AppState:
    fields = {
        "config": MagicMock(),
        "admin_service": MagicMock(),
        "health_service": MagicMock(),
        "graph_service": MagicMock(),
        "job_service": MagicMock(),
        "document_service": MagicMock(),
        "publishing_service": MagicMock(),
    }
    fields.update(overrides)
    return AppState(**fields)


@pytest.fixture(scope="module")
def app() -> FastAPI:
    return create_app()


def make_current_user(user_id: str = "alice", role: UserRole = UserRole.OPERATOR) -> CurrentUser:
    return CurrentUser(user_id=user_id, role=role)


@pytest.fixture
def api_client(app: FastAPI) -> Generator[Callable[..., AsyncClient]]:
    """Factory fixture that wires state/user overrides and cleans up after the test.

    Usage::

        async def test_foo(self, api_client):
            async with api_client(state, make_current_user()) as ac:
                resp = await ac.get("/some/path")
    """

    def _make(state: AppState, user: CurrentUser | None = None) -> AsyncClient:
        app.dependency_overrides[get_app_state] = lambda: state
        if user is not None:
            app.dependency_overrides[_get_current_user] = lambda: user
        return AsyncClient(transport=ASGITransport(app=app), base_url="http://test/v1")

    yield _make
    app.dependency_overrides.clear()
