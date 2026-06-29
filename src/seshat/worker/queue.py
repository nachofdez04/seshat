from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

from seshat.models.enums import JobStatus
from seshat.utils.log import get_logger

logger = get_logger(__name__)


class AsyncioTaskQueue:
    def __init__(self) -> None:
        self._statuses: dict[str, JobStatus] = {}
        self._tasks: dict[str, asyncio.Task] = {}

    async def enqueue(self, job_id: str, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        self._statuses[job_id] = JobStatus.PENDING
        task = asyncio.create_task(self._run(job_id, fn, *args, **kwargs))
        self._tasks[job_id] = task

    async def _run(self, job_id: str, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        try:
            await fn(*args, **kwargs)
            self._statuses[job_id] = JobStatus.DONE
        except asyncio.CancelledError:
            self._statuses[job_id] = JobStatus.FAILED
            raise
        except Exception as exc:
            logger.error("Job %s failed: %s", job_id, exc)
            self._statuses[job_id] = JobStatus.FAILED

    async def get_status(self, job_id: str) -> JobStatus | None:
        return self._statuses.get(job_id)

    async def cancel(self, job_id: str) -> bool:
        task = self._tasks.get(job_id)
        if task and not task.done():
            task.cancel()
            return True
        return False
