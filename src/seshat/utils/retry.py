import asyncio
import functools
import logging
import random
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


def async_retry(
    max_attempts: int = 3,
    base_delay: float = 0.5,
    retryable_exceptions: tuple = (Exception,),
    should_retry: Callable[[Exception], bool] | None = None,
) -> Callable:
    """Decorator: retries async methods with exponential backoff + jitter.

    ``max_attempts`` is the total number of calls (1 = no retry).
    ``should_retry(exc)`` is called for each caught exception; returning False
    re-raises immediately. If omitted, all ``retryable_exceptions`` are retried.
    """

    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            attempts = max(1, max_attempts)
            for attempt in range(attempts):
                try:
                    return await fn(*args, **kwargs)
                except retryable_exceptions as exc:
                    if attempt == attempts - 1 or (should_retry is not None and not should_retry(exc)):
                        raise
                    delay = base_delay * (2**attempt) + random.uniform(0, 0.1)
                    logger.warning(
                        "Retry %d/%d for %s (%s) — retrying in %.2fs",
                        attempt + 1,
                        attempts,
                        fn.__name__,
                        exc,
                        delay,
                    )
                    await asyncio.sleep(delay)

        return wrapper

    return decorator
