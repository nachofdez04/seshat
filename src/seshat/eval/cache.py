from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, TypeVar

from pydantic import BaseModel

from seshat.core.utils.hashing import fingerprint
from seshat.core.utils.log import get_logger

if TYPE_CHECKING:
    from collections.abc import Coroutine
    from pathlib import Path

logger = get_logger(__name__)


M = TypeVar("M", bound=BaseModel)


class _CorpusExample(Protocol):
    corpus_id: str

    def model_dump_json(self) -> str: ...


def build_cache_fp(
    cache_dir: Path,
    example: _CorpusExample,
    *,
    agent_hash: str | None = None,
    index: int | None = None,
) -> Path:
    parts = [example.corpus_id]
    if index is not None:
        parts.append(str(index))
    if agent_hash is not None:
        parts.append(agent_hash)
    parts.append(fingerprint(example.model_dump_json()))
    return cache_dir / f"{'_'.join(parts)}.json"


async def read_or_run(
    cache_fp: Path,
    model_cls: type[M],
    coro: Coroutine[Any, Any, M],
) -> tuple[M, Path, bool]:
    """Return a cached result if available, otherwise await the coroutine and persist the result.

    Returns (result, cache_path, was_cached). cache_path is for mark-and-sweep tracking;
    was_cached is True when the result came from disk rather than a live LLM call.
    The cache directory must exist before calling this function.
    """
    # Cache files are small local JSON blobs; the blocking time is negligible compared to
    # the LLM calls this function wraps, so `utils.concurrency.run_in_thread` is not worth the overhead here.
    if cache_fp.exists():  # noqa: ASYNC240
        coro.close()
        logger.debug("Cache hit in %r call: using result from %s", coro.__name__, cache_fp)
        return model_cls.model_validate_json(cache_fp.read_text()), cache_fp, True  # noqa: ASYNC240

    logger.debug("Cache miss in %r call: running coroutine and caching result to %s", coro.__name__, cache_fp)
    result = await coro
    cache_fp.write_text(result.model_dump_json())  # noqa: ASYNC240
    return result, cache_fp, False


def sweep_stale_entries(
    cache_dir: Path,
    corpus_ids: list[str],
    touched: set[Path],
    agent_hash: str | None = None,
) -> None:
    """Delete cache files for the given corpus IDs that were not touched in this run.

    Files whose corpus_id is not in `corpus_ids` (out-of-scope due to tag filtering) are
    left untouched. Only entries that were in scope but not hit (stale prompt or input hash)
    are removed. agent_hash scopes the sweep to the current agent only, so parallel cache
    entries for a different agent variant (e.g. shallow vs reflective) are preserved.
    """
    pattern_suffix = f"{agent_hash}_*.json" if agent_hash else "*.json"
    for corpus_id in corpus_ids:
        for f in cache_dir.glob(f"{corpus_id}_{pattern_suffix}"):
            if f not in touched:
                logger.debug("Removing stale cache file %s", f)
                f.unlink(missing_ok=True)


def clear_cache_dir(cache_dir: Path) -> None:
    """Delete all files in the cache directory. Use with caution."""
    logger.info("Clearing cache directory %s", cache_dir)
    for f in cache_dir.glob("*.json"):
        f.unlink(missing_ok=True)
