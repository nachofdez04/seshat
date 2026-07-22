from __future__ import annotations

import asyncio
import functools
import statistics
import time
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

from langchain_core.callbacks import AsyncCallbackHandler

from seshat.app.platform.observability.mlflow_run_logging import log_latency_metrics
from seshat.core.utils.log import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable
    from uuid import UUID

logger = get_logger(__name__)

_profiling_tracker_var: ContextVar[LatencyTrackerCallback | None] = ContextVar("profiling_tracker", default=None)


class LatencyTracker:
    def __init__(self) -> None:
        self._durations: list[float] = []

    @property
    def durations(self) -> list[float]:
        return list(self._durations)

    def record(self, duration_ms: float) -> None:
        self._durations.append(duration_ms)

    def log_totals(self, label: str) -> None:
        durations = self.durations
        if not durations:
            return
        sorted_d = sorted(durations)
        p95_idx = min(int(0.95 * len(sorted_d)), len(sorted_d) - 1)
        logger.info(
            "%s latency: calls=%d min=%dms mean=%dms p95=%dms max=%dms",
            label,
            len(sorted_d),
            round(sorted_d[0]),
            round(statistics.mean(sorted_d)),
            round(sorted_d[p95_idx]),
            round(sorted_d[-1]),
        )


class LatencyTrackerCallback(AsyncCallbackHandler):
    def __init__(self, tracker: LatencyTracker) -> None:
        self._start_times: dict[UUID, float] = {}
        self._lock = asyncio.Lock()
        self.tracker = tracker

    async def on_llm_start(self, serialized: dict, prompts: list, *, run_id: UUID, **kwargs: Any) -> None:
        async with self._lock:
            self._start_times[run_id] = time.perf_counter_ns()

    async def on_llm_end(self, response: Any, *, run_id: UUID, **kwargs: Any) -> None:
        async with self._lock:
            start = self._start_times.pop(run_id, None)

        if start is None:
            logger.warning("LatencyTrackerCallback: unknown run_id %s on on_llm_end — skipping", run_id)
            return

        duration_ms = (time.perf_counter_ns() - start) / 1_000_000
        self.tracker.record(duration_ms)


def set_profiling_tracker(callback: LatencyTrackerCallback) -> None:
    _profiling_tracker_var.set(callback)


def get_profiling_tracker() -> LatencyTrackerCallback | None:
    return _profiling_tracker_var.get()


def track_latency_profile(label: str, *, metrics_label: str | None = None) -> Callable:
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        async def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            tracker = LatencyTracker()
            set_profiling_tracker(LatencyTrackerCallback(tracker))
            try:
                return await fn(self, *args, **kwargs)
            finally:
                tracker.log_totals(label)
                log_latency_metrics(metrics_label if metrics_label is not None else label, tracker.durations)

        return wrapper

    return decorator


track_eval_latency = functools.partial(track_latency_profile, metrics_label="")
