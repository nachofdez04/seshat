"""Tests for UsageTracker, TokenBudgetCallback, and track_token_budget."""

from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from seshat.observability.usage_tracker import (
    TokenBudgetCallback,
    TokenBudgetExceededError,
    TrackingTranscriber,
    UsageTracker,
    get_run_tracker,
    set_run_tracker,
    track_token_budget,
)


def _result(
    input_tokens: int,
    output_tokens: int,
    cache_read: int = 0,
    cache_creation: int = 0,
) -> LLMResult:
    message = AIMessage(
        content="ok",
        usage_metadata={
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "input_token_details": {"cache_read": cache_read, "cache_creation": cache_creation},
        },
    )
    return LLMResult(generations=[[ChatGeneration(message=message)]])


class TestUsageTracker:
    async def test_accumulates_tokens(self):
        tracker = UsageTracker(max_input_tokens=1000, max_output_tokens=500)
        await tracker.add(input_tokens=100, output_tokens=50)
        await tracker.add(input_tokens=200, output_tokens=80)
        assert tracker.input_tokens == 300
        assert tracker.output_tokens == 130

    async def test_accumulates_cache_tokens(self):
        tracker = UsageTracker(max_input_tokens=1000, max_output_tokens=500)
        await tracker.add(input_tokens=0, output_tokens=0, cache_read_tokens=40, cache_creation_tokens=10)
        await tracker.add(input_tokens=0, output_tokens=0, cache_read_tokens=60, cache_creation_tokens=5)
        assert tracker.cache_read_tokens == 100
        assert tracker.cache_creation_tokens == 15

    def test_check_caps_passes_within_limits(self):
        tracker = UsageTracker(max_input_tokens=1000, max_output_tokens=500)
        tracker._input_tokens = 1050  # within 10% overage
        tracker._output_tokens = 525
        tracker.check_caps()  # should not raise

    def test_check_caps_raises_on_input_exceeded(self):
        tracker = UsageTracker(max_input_tokens=1000, max_output_tokens=500)
        tracker._input_tokens = 1101  # over 110%
        with pytest.raises(TokenBudgetExceededError, match="Input token cap exceeded"):
            tracker.check_caps()

    def test_check_caps_raises_on_output_exceeded(self):
        tracker = UsageTracker(max_input_tokens=1000, max_output_tokens=500)
        tracker._output_tokens = 551  # over 110%
        with pytest.raises(TokenBudgetExceededError, match="Output token cap exceeded"):
            tracker.check_caps()

    async def test_accumulates_embedding_tokens(self):
        tracker = UsageTracker(max_input_tokens=1000, max_output_tokens=500)
        await tracker.add(embedding_input_tokens=120)
        await tracker.add(embedding_input_tokens=80)
        assert tracker.embedding_input_tokens == 200
        assert tracker.input_tokens == 0

    def test_check_caps_raises_on_embedding_exceeded(self):
        tracker = UsageTracker(max_input_tokens=1000, max_output_tokens=500, max_embedding_tokens=100)
        tracker._embedding_input_tokens = 111  # over 110%
        with pytest.raises(TokenBudgetExceededError, match="Embedding token cap exceeded"):
            tracker.check_caps()


class TestTokenBudgetCallback:
    async def test_accumulates_from_llm_result(self):
        tracker = UsageTracker(max_input_tokens=10_000, max_output_tokens=10_000)
        callback = TokenBudgetCallback(tracker)

        await callback.on_llm_end(_result(42, 17), run_id=uuid4())

        assert tracker.input_tokens == 42
        assert tracker.output_tokens == 17

    async def test_accumulates_cache_tokens_from_llm_result(self):
        tracker = UsageTracker(max_input_tokens=10_000, max_output_tokens=10_000)
        callback = TokenBudgetCallback(tracker)

        await callback.on_llm_end(_result(10, 5, cache_read=30, cache_creation=20), run_id=uuid4())

        assert tracker.cache_read_tokens == 30
        assert tracker.cache_creation_tokens == 20

    async def test_skips_generation_without_usage_metadata(self):
        tracker = UsageTracker(max_input_tokens=10_000, max_output_tokens=10_000)
        callback = TokenBudgetCallback(tracker)

        result = LLMResult(generations=[[ChatGeneration(message=AIMessage(content="ok"))]])
        await callback.on_llm_end(result, run_id=uuid4())

        assert tracker.input_tokens == 0
        assert tracker.output_tokens == 0


class TestTrackTokenBudget:
    async def test_sets_run_tracker_before_fn(self):
        captured = []

        class _Obj:
            @track_token_budget(label="test", max_input_fn=lambda self: 1000, max_output_fn=lambda self: 500)
            async def run(self):
                captured.append(get_run_tracker())

        await _Obj().run()
        assert len(captured) == 1
        assert captured[0] is not None

    async def test_raises_on_cap_exceeded(self):
        class _Obj:
            @track_token_budget(label="test", max_input_fn=lambda self: 10, max_output_fn=lambda self: 10)
            async def run(self):
                cb = get_run_tracker()
                assert cb is not None
                await cb._tracker.add(input_tokens=200, output_tokens=0)  # well over 110%

        with pytest.raises(TokenBudgetExceededError):
            await _Obj().run()

    async def test_does_not_raise_within_overage_allowance(self):
        class _Obj:
            @track_token_budget(label="test", max_input_fn=lambda self: 1000, max_output_fn=lambda self: 500)
            async def run(self):
                cb = get_run_tracker()
                assert cb is not None
                await cb._tracker.add(input_tokens=1050, output_tokens=0)  # within 10% overage

        await _Obj().run()  # should not raise

    async def test_tracker_isolated_per_call(self):
        trackers = []

        class _Obj:
            @track_token_budget(label="test", max_input_fn=lambda self: 10_000, max_output_fn=lambda self: 10_000)
            async def run(self):
                trackers.append(get_run_tracker())

        obj = _Obj()
        await obj.run()
        await obj.run()

        assert len(trackers) == 2
        assert trackers[0] is not trackers[1]


class TestUsageTrackerAudioSeconds:
    async def test_accumulates_audio_seconds(self):
        tracker = UsageTracker(max_input_tokens=1000, max_output_tokens=500)
        await tracker.add(audio_seconds=30)
        await tracker.add(audio_seconds=45)
        assert tracker.audio_seconds == 75

    async def test_audio_seconds_independent_of_token_counts(self):
        tracker = UsageTracker(max_input_tokens=1000, max_output_tokens=500)
        await tracker.add(input_tokens=100, audio_seconds=60)
        assert tracker.input_tokens == 100
        assert tracker.audio_seconds == 60


class TestTrackingTranscriber:
    async def test_delegates_to_inner_transcriber(self):
        audio_bytes = b"\x00" * 16

        inner_transcriber = AsyncMock()
        inner_transcriber.transcribe.return_value = "hello world"

        with patch("seshat.utils.audio.mutagen.File") as mock_mutagen:
            mock_mutagen.return_value.info.length = 10.0
            transcriber = TrackingTranscriber(inner_transcriber)
            result = await transcriber.transcribe(audio_bytes, extension="mp3")

        assert result == "hello world"
        inner_transcriber.transcribe.assert_awaited_once_with(audio_bytes, "mp3")

    async def test_records_audio_seconds_in_tracker(self):
        audio_bytes = b"\x00" * 16

        inner_transcriber = AsyncMock()
        inner_transcriber.transcribe.return_value = ""

        tracker = UsageTracker(max_input_tokens=1000, max_output_tokens=500)
        callback = TokenBudgetCallback(tracker)
        set_run_tracker(callback)

        with patch("seshat.utils.audio.mutagen.File") as mock_mutagen:
            mock_mutagen.return_value.info.length = 7.3
            transcriber = TrackingTranscriber(inner_transcriber)
            await transcriber.transcribe(audio_bytes, extension="mp3")

        assert tracker.audio_seconds == 8  # ceil(7.3)

    async def test_skips_tracking_when_duration_unreadable(self):
        inner_transcriber = AsyncMock()
        inner_transcriber.transcribe.return_value = ""

        tracker = UsageTracker(max_input_tokens=1000, max_output_tokens=500)
        set_run_tracker(TokenBudgetCallback(tracker))

        transcriber = TrackingTranscriber(inner_transcriber)
        await transcriber.transcribe(b"\x00" * 64, extension="mp3")

        assert tracker.audio_seconds == 0
