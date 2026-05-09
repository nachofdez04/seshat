from unittest.mock import AsyncMock

import pytest

from seshat.utils.retry import async_retry


async def test_success_on_first_attempt():
    fn = AsyncMock(return_value="ok")
    decorated = async_retry()(fn)

    result = await decorated()

    assert result == "ok"
    fn.assert_awaited_once()


async def test_retries_until_success():
    fn = AsyncMock(side_effect=[ValueError("first"), ValueError("second"), "ok"])
    decorated = async_retry(max_attempts=3, base_delay=0)(fn)

    result = await decorated()

    assert result == "ok"
    assert fn.await_count == 3


async def test_raises_after_max_attempts():
    fn = AsyncMock(side_effect=ValueError("always fails"))
    decorated = async_retry(max_attempts=3, base_delay=0)(fn)

    with pytest.raises(ValueError, match="always fails"):
        await decorated()

    assert fn.await_count == 3


async def test_non_retryable_exception_raises_immediately():
    fn = AsyncMock(side_effect=[TypeError("not retryable"), "ok"])
    decorated = async_retry(max_attempts=3, base_delay=0, retryable_exceptions=(ValueError,))(fn)

    with pytest.raises(TypeError):
        await decorated()

    fn.assert_awaited_once()


async def test_should_retry_false_stops_immediately():
    fn = AsyncMock(side_effect=[ValueError("bail"), "ok"])
    decorated = async_retry(max_attempts=3, base_delay=0, should_retry=lambda _: False)(fn)

    with pytest.raises(ValueError, match="bail"):
        await decorated()

    fn.assert_awaited_once()


async def test_should_retry_selectively_retries():
    retryable = ValueError("retry me")
    non_retryable = ValueError("stop here")
    fn = AsyncMock(side_effect=[retryable, non_retryable, "ok"])

    def should_retry(exc: Exception) -> bool:
        return str(exc) == "retry me"

    decorated = async_retry(max_attempts=3, base_delay=0, should_retry=should_retry)(fn)

    with pytest.raises(ValueError, match="stop here"):
        await decorated()

    assert fn.await_count == 2


async def test_max_attempts_one_means_no_retry():
    fn = AsyncMock(side_effect=ValueError("fail"))
    decorated = async_retry(max_attempts=1, base_delay=0)(fn)

    with pytest.raises(ValueError, match="fail"):
        await decorated()

    fn.assert_awaited_once()


async def test_max_attempts_zero_calls_function_once():
    fn = AsyncMock(return_value="ok")
    decorated = async_retry(max_attempts=0)(fn)

    result = await decorated()

    assert result == "ok"
    fn.assert_awaited_once()
