from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from httpx import ASGITransport, AsyncClient

from seshat.api.app import create_app
from seshat.api.dependencies import CurrentUser, _get_current_user, get_app_state
from seshat.models.enums import UserRole

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

    from fastapi import FastAPI

    from seshat.api.state import AppState


@pytest.fixture
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
