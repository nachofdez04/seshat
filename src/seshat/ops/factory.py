from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

import asyncpg

from seshat.ops.ledger import OpsLedger
from seshat.secrets.factory import get_secrets_resolver
from seshat.utils.log import get_logger

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from seshat.config.settings import SeshatConfig

logger = get_logger(__name__)


@asynccontextmanager
async def get_ops_ledger(config: SeshatConfig) -> AsyncIterator[OpsLedger]:
    """Async context manager — yields a ready OpsLedger and closes the pool on exit."""
    pool = await _get_pool(config)
    ledger = OpsLedger(pool)
    try:
        yield ledger
    finally:
        await ledger.close()


async def _get_pool(config: SeshatConfig) -> asyncpg.Pool:
    """Get an asyncpg pool for the ops ledger."""
    secrets = get_secrets_resolver(config)
    pg_url = secrets.get_secret(config.kb_store.connection_secret_key)
    logger.debug("Creating asyncpg pool")
    return await asyncpg.create_pool(pg_url)
