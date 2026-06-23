from __future__ import annotations

import asyncio
import random
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, TypeVar

from pydantic import BaseModel

from seshat.observability.latency_tracker import get_profiling_tracker
from seshat.observability.usage_tracker import get_run_tracker
from seshat.utils.hashing import fingerprint
from seshat.utils.log import get_logger

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

    from seshat.config.settings import _LLMConfig

M = TypeVar("M", bound=BaseModel)


logger = get_logger(__name__)


class RetryExhaustedError(Exception):
    pass


class _BaseAgent(ABC):
    """Base class for all LLM-calling agents. Provides a structured-output call with exponential backoff retry."""

    def __init__(self, llm: BaseChatModel, config: _LLMConfig) -> None:
        self._llm = llm
        self._max_retries = config.max_retries

    def __str__(self) -> str:
        return f"{self.name}(model={self._llm})"

    @property
    def name(self) -> str:
        return type(self).__name__

    def fingerprint(self) -> str:
        combined_prompts = "".join(self.prompt_texts().values())
        return fingerprint(combined_prompts)

    def prompt_texts(self) -> dict[str, str]:
        return {"system": self._system_prompt}

    @property
    @abstractmethod
    def _system_prompt(self) -> str: ...

    async def _retryable_structured_ainvoke(
        self,
        messages: list,
        response_model: type[M],
        *,
        raise_on_exhaustion: RetryExhaustedError,
        on_error_log_prefix: str | None = None,
        llm: BaseChatModel | None = None,
    ) -> M:
        llm = llm or self._llm
        structured = llm.with_structured_output(response_model)

        callbacks = self._get_callbacks()
        if callbacks:
            structured = structured.with_config(callbacks=callbacks)

        on_error_log_prefix = on_error_log_prefix or response_model.__name__
        attempts = max(1, self._max_retries)  # run at least once
        for attempt in range(attempts):
            try:
                result = await structured.ainvoke(messages)
            except Exception as exc:
                logger.warning(
                    "%s attempt %d/%d failed due to %s: %s",
                    on_error_log_prefix,
                    attempt + 1,
                    attempts,
                    type(exc).__name__,
                    exc,
                )
                if attempt < attempts - 1:
                    delay = 0.5 * (2**attempt) + random.uniform(0, 0.1)
                    await asyncio.sleep(delay)
            else:
                assert_msg = f"Expected LLM output to be {response_model.__name__}, got {type(result).__name__}"
                assert isinstance(result, response_model), assert_msg
                return result

        raise raise_on_exhaustion

    def _get_callbacks(self) -> list:
        """Return a list of callbacks for the LLM call, including profiling and usage tracking."""
        callbacks = []
        for callback in (get_profiling_tracker(), get_run_tracker()):
            if callback is not None:
                callbacks.append(callback)
        return callbacks
