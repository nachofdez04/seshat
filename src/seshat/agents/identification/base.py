from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, TypeVar

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from seshat.agents.base import RetryExhaustedError, _BaseAgent
from seshat.models.quote_anchor import QuoteAnchor
from seshat.utils.log import get_logger

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

    from seshat.agents.identification.grouping import ConceptGroup
    from seshat.config.settings import IdentificationLLMConfig
    from seshat.models.enums import ConceptType

logger = get_logger(__name__)


class IdentificationRetryExhaustedError(RetryExhaustedError):
    pass


class ConceptModel(BaseModel):
    quote: str = Field(
        description=(
            "A verbatim excerpt from the transcript covering the full exchange that informs this item — "
            "from the first passage that raises the topic through all passages needed to support it. "
            "Do not paraphrase, reconstruct, or splice passages with ellipsis. "
            "The description must not claim anything that is not present in this quote."
        )
    )
    title: str = Field(description="Short, specific title for the item.")
    description: str = Field(
        description=(
            "One to two sentences describing the item in plain language. Use different vocabulary and framing from the "
            "title to broaden semantic coverage for retrieval."
        )
    )


M = TypeVar("M", bound=ConceptModel)


class ConceptList(BaseModel, Generic[M]):
    items: list[M] = Field(default_factory=list)


class AnchoredConcept(BaseModel, Generic[M]):
    item: M
    quote_anchor: QuoteAnchor | None


class _BaseIdentificationAgent(_BaseAgent, ABC, Generic[M]):
    def __init__(
        self,
        llm: BaseChatModel,
        config: IdentificationLLMConfig,
        grouped_identification_types: set[ConceptType],
    ) -> None:
        super().__init__(llm=llm, config=config)
        self._config = config
        self._grouped_identification_types = grouped_identification_types

    @property
    @abstractmethod
    def concept_type(self) -> ConceptType: ...

    @property
    @abstractmethod
    def output_schema(self) -> type[ConceptList[M]]: ...

    @property
    @abstractmethod
    def _system_prompt(self) -> str: ...

    @property
    def grouped_identification(self) -> bool:
        return self.concept_type in self._grouped_identification_types

    async def identify(
        self, transcript: str, kb_hint: str, transcript_file: str
    ) -> list[AnchoredConcept[M]] | list[ConceptGroup[M]]:
        t0 = time.perf_counter()
        identification = await self._identify(transcript, kb_hint, transcript_file)

        result = await self._group_identification(identification) if self.grouped_identification else identification

        elapsed_ms = round((time.perf_counter() - t0) * 1000)
        logger.info(
            "%s identified %d items (grouped=%s, elapsed: %dms)",
            self.concept_type.value,
            len(result),
            self.grouped_identification,
            elapsed_ms,
            extra={"elapsed_ms": elapsed_ms, "concept_type": self.concept_type.value},
        )
        return result

    async def _identify(self, transcript: str, kb_hint: str | None, transcript_file: str) -> list[AnchoredConcept[M]]:
        result = await self._retryable_structured_ainvoke(
            messages=self._build_messages(transcript, kb_hint),
            response_model=self.output_schema,
            raise_on_exhaustion=IdentificationRetryExhaustedError(
                f"Agent {self.concept_type} exhausted {self._max_retries} retries"
            ),
            on_error_log_prefix=f"Agent({self.concept_type})",
        )
        return self._anchor_quotes(result.items, transcript, transcript_file)

    def _build_messages(self, transcript: str, kb_hint: str | None) -> list:
        content = (
            "Return all items you find using the structured output format.\n\n"
            f"<transcript>\n{transcript}\n</transcript>\n\n"
        )
        if kb_hint:
            content += f"Existing nodes from the knowledge base on this topic.\n\n<kb_hint>\n{kb_hint}\n</kb_hint>"

        # Transcript is intentionally not cached: the concept-type agents run in parallel, so all
        # calls are concurrent and would all miss the cache — paying the write premium with no hits.
        return [
            SystemMessage(
                content=[{"type": "text", "text": self._system_prompt, "cache_control": {"type": "ephemeral"}}]
            ),
            HumanMessage(content=content),
        ]

    async def _group_identification(self, items: list[AnchoredConcept[M]]) -> list[ConceptGroup[M]]:
        from seshat.agents.identification.grouping import GroupingAgent

        grouping_agent = GroupingAgent(llm=self._llm, config=self._config)
        return await grouping_agent.group(items, self.concept_type)

    def _anchor_quotes(self, items: list[M], transcript: str, transcript_file: str) -> list[AnchoredConcept[M]]:
        anchored = []
        for item in items:
            anchor = QuoteAnchor.from_transcript_quote(item.quote, transcript, transcript_file)
            if anchor is None:
                logger.warning(
                    "Agent %s: quote not anchored — '%s...'",
                    self.concept_type,
                    item.quote[:40],
                )
            anchored.append(AnchoredConcept(item=item, quote_anchor=anchor))
        return anchored
