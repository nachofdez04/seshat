from __future__ import annotations

import asyncio

import bcrypt


class AuthenticationError(Exception):
    pass


async def verify_api_key(
    key: str,
    stored_keys: list[tuple[str, str, str]],
) -> tuple[str, str]:
    """Check key against (hash, user_id, role) tuples using constant-time bcrypt.

    Returns (user_id, role) on success.
    """
    for key_hash, user_id, role in stored_keys:
        match = await asyncio.to_thread(bcrypt.checkpw, key.encode(), key_hash.encode())
        if match:
            return user_id, role
    raise AuthenticationError("Invalid API key")
