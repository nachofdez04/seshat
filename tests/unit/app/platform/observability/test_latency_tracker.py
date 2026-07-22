from __future__ import annotations

import asyncio
import logging
from unittest.mock import MagicMock, patch
from uuid import uuid4

from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from seshat.app.platform.observability.latency_tracker import (
    LatencyTracker,
    LatencyTrackerCallback,
    track_eval_latency,
    track_latency_profile,
)
from seshat.app.platform.observability.mlflow_run_logging import log_latency_metrics


def _make_llm_result() -> LLMResult:
    msg = AIMessage(content="ok")
    return LLMResult(generations=[[ChatGeneration(message=msg)]])


def _make_callback() -> LatencyTrackerCallback:
    return LatencyTrackerCallback(LatencyTracker())


class TestLatencyTracker:
    def test_durations_returns_copy(self):
        t = LatencyTracker()
        t.durations.append(99.0)
        assert len(t.durations) == 0

    def test_log_totals_emits_info(self, caplog):
        t = LatencyTracker()
        t._durations = [100.0, 200.0, 300.0]

        with caplog.at_level(logging.INFO, logger="seshat.app.platform.observability.latency_tracker"):
            t.log_totals("grounding")

        assert any("grounding latency" in r.message for r in caplog.records)

    def test_log_totals_noop_when_empty(self, caplog):
        t = LatencyTracker()

        with caplog.at_level(logging.INFO, logger="seshat.app.platform.observability.latency_tracker"):
            t.log_totals("grounding")

        assert not caplog.records


class TestLatencyTrackerCallback:
    async def test_records_duration_for_single_call(self):
        cb = _make_callback()
        run_id = uuid4()
        await cb.on_llm_start({}, [], run_id=run_id)
        await cb.on_llm_end(_make_llm_result(), run_id=run_id)

        assert len(cb.tracker.durations) == 1
        assert cb.tracker.durations[0] > 0

    async def test_records_durations_for_concurrent_calls(self):
        cb = _make_callback()
        ids = [uuid4() for _ in range(3)]
        await asyncio.gather(*[cb.on_llm_start({}, [], run_id=rid) for rid in ids])
        await asyncio.gather(*[cb.on_llm_end(_make_llm_result(), run_id=rid) for rid in ids])

        assert len(cb.tracker.durations) == 3

    async def test_unknown_run_id_on_end_logs_warning_and_skips(self, caplog):
        cb = _make_callback()

        with caplog.at_level(logging.WARNING, logger="seshat.app.platform.observability.latency_tracker"):
            await cb.on_llm_end(_make_llm_result(), run_id=uuid4())

        assert len(cb.tracker.durations) == 0
        assert any("unknown run_id" in r.message.lower() for r in caplog.records)


class TestLogLatencyMetrics:
    def test_noop_when_no_active_run(self):
        with patch("mlflow.active_run", return_value=None), patch("mlflow.log_metrics") as mock_log:
            log_latency_metrics("test", [10.0, 20.0])

        mock_log.assert_not_called()

    def test_noop_when_durations_empty(self):
        mock_run = MagicMock()
        with patch("mlflow.active_run", return_value=mock_run), patch("mlflow.log_metrics") as mock_log:
            log_latency_metrics("test", [])

        mock_log.assert_not_called()

    def test_logs_correct_keys_and_values(self):
        mock_run = MagicMock()
        durations = [10.0, 20.0, 30.0, 40.0, 50.0]

        with patch("mlflow.active_run", return_value=mock_run), patch("mlflow.log_metrics") as mock_log:
            log_latency_metrics("my.stage", durations)

        metrics = mock_log.call_args[0][0]
        assert metrics["latency.my_stage.min_ms"] == 10.0
        assert metrics["latency.my_stage.max_ms"] == 50.0
        assert metrics["latency.my_stage.call_count"] == 5.0

    def test_sanitises_stage_name(self):
        mock_run = MagicMock()

        with patch("mlflow.active_run", return_value=mock_run), patch("mlflow.log_metrics") as mock_log:
            log_latency_metrics("my-stage name", [10.0])

        metrics = mock_log.call_args[0][0]
        assert all("my_stage_name" in k for k in metrics)


class TestTrackLatencyProfile:
    async def test_decorator_calls_log_latency_metrics(self):
        with patch("seshat.app.platform.observability.latency_tracker.log_latency_metrics") as mock_log:

            class _Dummy:
                @track_latency_profile("my_stage")
                async def run(self):
                    return 42

            result = await _Dummy().run()

        assert result == 42
        mock_log.assert_called_once()
        stage, durations = mock_log.call_args[0]
        assert stage == "my_stage"
        assert isinstance(durations, list)

    async def test_metrics_label_overrides_stage(self):
        with patch("seshat.app.platform.observability.latency_tracker.log_latency_metrics") as mock_log:

            class _Dummy:
                @track_latency_profile("my_stage", metrics_label="")
                async def run(self):
                    return 42

            await _Dummy().run()

        stage, _ = mock_log.call_args[0]
        assert stage == ""

    async def test_track_eval_latency_uses_empty_metrics_label(self):
        with patch("seshat.app.platform.observability.latency_tracker.log_latency_metrics") as mock_log:

            class _Dummy:
                @track_eval_latency("identification")
                async def run(self):
                    return 42

            await _Dummy().run()

        stage, _ = mock_log.call_args[0]
        assert stage == ""
