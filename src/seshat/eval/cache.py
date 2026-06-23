from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

from pydantic import BaseModel

if TYPE_CHECKING:
    from collections.abc import Coroutine
    from pathlib import Path

M = TypeVar("M", bound=BaseModel)


def clear_cache_dir(cache_dir: Path) -> None:
    for f in cache_dir.glob("*.json"):
        f.unlink(missing_ok=True)


async def read_or_run(
    cache_file: Path,
    model_cls: type[M],
    coro: Coroutine[Any, Any, M],
) -> M:
    """Return a cached result if available, otherwise await the coroutine and persist the result.

    The cache directory must exist before calling this function.
    """
    # Cache files are small local JSON blobs; the blocking time is negligible compared to
    # the LLM calls this function wraps, so asyncio.to_thread is not worth the added noise.
    if cache_file.exists():  # noqa: ASYNC240
        coro.close()
        return model_cls.model_validate_json(cache_file.read_text())  # noqa: ASYNC240
    result = await coro
    cache_file.write_text(result.model_dump_json())  # noqa: ASYNC240
    return result
