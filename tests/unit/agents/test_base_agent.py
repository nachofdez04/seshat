"""Tests for _BaseAgent._retryable_structured_ainvoke."""

from unittest.mock import AsyncMock, patch

import pytest

from seshat.agents.base import RetryExhaustedError, _BaseAgent
from seshat.agents.grounding import GroundingResult
from seshat.config.settings import IdentificationLLMConfig
from tests.helpers import make_structured_llm


class _ConcreteAgent(_BaseAgent):
    def __init__(self, llm, max_retries: int = 3) -> None:
        super().__init__(llm=llm, config=IdentificationLLMConfig(max_retries=max_retries))

    @property
    def _system_prompt(self) -> str:
        return ""


def _make_agent(side_effect=None, return_value=None, max_retries: int = 3) -> _ConcreteAgent:
    return _ConcreteAgent(
        llm=make_structured_llm(return_value=return_value, side_effect=side_effect), max_retries=max_retries
    )


class TestRetryableStructuredAinvoke:
    async def test_returns_result_on_first_success(self):
        expected = GroundingResult(supported=True)
        agent = _make_agent(return_value=expected)

        result = await agent._retryable_structured_ainvoke(
            messages=[],
            response_model=GroundingResult,
            raise_on_exhaustion=RetryExhaustedError("exhausted"),
        )

        assert result is expected

    async def test_retries_on_failure_and_succeeds(self):
        expected = GroundingResult(supported=True)
        agent = _make_agent(side_effect=[Exception("fail"), expected])

        result = await agent._retryable_structured_ainvoke(
            messages=[],
            response_model=GroundingResult,
            raise_on_exhaustion=RetryExhaustedError("exhausted"),
        )

        assert result is expected

    async def test_raises_exhaustion_error_after_all_retries_fail(self):
        exhaustion = RetryExhaustedError("all retries exhausted")
        agent = _make_agent(side_effect=Exception("always fails"), max_retries=2)

        with pytest.raises(RetryExhaustedError, match="all retries exhausted"):
            await agent._retryable_structured_ainvoke(
                messages=[],
                response_model=GroundingResult,
                raise_on_exhaustion=exhaustion,
            )

    async def test_uses_provided_llm_override(self):
        expected = GroundingResult(supported=True)
        default_llm = make_structured_llm(return_value=GroundingResult(supported=False))
        override_llm = make_structured_llm(return_value=expected)
        agent = _ConcreteAgent(llm=default_llm)

        result = await agent._retryable_structured_ainvoke(
            messages=[],
            response_model=GroundingResult,
            raise_on_exhaustion=RetryExhaustedError("exhausted"),
            llm=override_llm,
        )

        assert result is expected
        override_llm.with_structured_output.assert_called_once()
        default_llm.with_structured_output.assert_not_called()

    async def test_attaches_profiling_callback_when_set(self):
        from unittest.mock import AsyncMock, MagicMock

        from seshat.observability.latency_tracker import LatencyTracker, LatencyTrackerCallback

        cb = LatencyTrackerCallback(LatencyTracker())
        structured_mock = MagicMock()
        configured_mock = MagicMock()
        configured_mock.ainvoke = AsyncMock(return_value=GroundingResult(supported=True))
        structured_mock.with_config = MagicMock(return_value=configured_mock)

        llm = MagicMock()
        llm.with_structured_output = MagicMock(return_value=structured_mock)
        agent = _ConcreteAgent(llm=llm)

        with patch("seshat.agents.base.get_profiling_tracker", return_value=cb):
            await agent._retryable_structured_ainvoke(
                messages=[],
                response_model=GroundingResult,
                raise_on_exhaustion=RetryExhaustedError("exhausted"),
            )

        structured_mock.with_config.assert_called_once()
        callbacks_passed = structured_mock.with_config.call_args[1]["callbacks"]
        assert cb in callbacks_passed

    async def test_sleeps_between_retry_attempts(self):
        exhaustion = RetryExhaustedError("exhausted")
        agent = _make_agent(side_effect=Exception("fail"), max_retries=3)

        with (
            patch("seshat.agents.base.asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
            pytest.raises(RetryExhaustedError),
        ):
            await agent._retryable_structured_ainvoke(
                messages=[],
                response_model=GroundingResult,
                raise_on_exhaustion=exhaustion,
            )

        assert mock_sleep.call_count == 2  # no sleep after the final failed attempt
