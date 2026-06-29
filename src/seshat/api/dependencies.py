from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Annotated, Any

from fastapi import Depends, Header, HTTPException, Request, status

from seshat.api.auth import AuthenticationError, verify_api_key
from seshat.api.state import AppState
from seshat.config.settings import SeshatConfig
from seshat.models.enums import UserRole

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine


@lru_cache
def get_config() -> SeshatConfig:
    return SeshatConfig()


def get_app_state(request: Request) -> AppState:
    return request.app.state.app_state


class CurrentUser:
    def __init__(self, user_id: str, role: UserRole) -> None:
        self.user_id = user_id
        self.role = role


async def _get_current_user(
    state: Annotated[AppState, Depends(get_app_state)],
    x_api_key: Annotated[str | None, Header()] = None,
) -> CurrentUser:
    if not x_api_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="X-API-Key required")
    stored_keys = await state.ops.get_api_keys()
    try:
        user_id, role = await verify_api_key(x_api_key, stored_keys)
    except AuthenticationError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")

    return CurrentUser(user_id=user_id, role=UserRole(role))


def require_role(minimum: UserRole) -> Callable[..., Coroutine[Any, Any, CurrentUser]]:
    async def _check(user: Annotated[CurrentUser, Depends(_get_current_user)]) -> CurrentUser:
        if not user.role.is_at_least(minimum):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient role")
        return user

    return _check
