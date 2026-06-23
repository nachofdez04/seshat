from __future__ import annotations

import json
from typing import TYPE_CHECKING, TypeVar

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from seshat.agents.base import RetryExhaustedError
from seshat.agents.identification.base import (
    AnchoredConcept,
    ConceptList,
    ConceptModel,
    _BaseIdentificationAgent,
)
from seshat.utils.log import get_logger

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

    from seshat.agents.identification.grouping import ConceptGroup
    from seshat.models.enums import ConceptType

logger = get_logger(__name__)

M = TypeVar("M", bound=ConceptModel)


_VALIDATE_PROMPT = """\
Check each extracted item on two dimensions.

**Logical compliance** — check against the system prompt above:
- Does this item satisfy the Positive criteria?
- Does it pass all Over-extraction guards?

**Semantic compliance:**
- Does the title reflect what the quote actually contains?
- Does the description match what the quote describes?

Reject an item only if it fails a logical check, or if its description contradicts
or misrepresents what the quote actually says.

Do not reject for minor phrasing, vague titles, borderline quality, or stylistic
preference. Also, quote verbatimness is out of your scope: do not reject nodes
solely because you cannot confirm the quote in the transcript, since you won't
see the transcript itself. When in doubt, pass the item.

Return exactly one review per item, in the same order as the input list.

<extracted_nodes>
{nodes_json}
</extracted_nodes>
"""


class _SelfReviewRetryExhaustedError(RetryExhaustedError):
    pass


class NodeReview(BaseModel):
    passed: bool = Field(description="True if the item passes, False if it should be discarded.")
    rationale: str | None = Field(
        default=None,
        description="Brief explanation of which rule the item violates. Required when passed=False, else None.",
    )


class SelfReviewResult(BaseModel):
    reviews: list[NodeReview] = Field(
        description="One review per extracted item, in the same order as the input list.",
    )


class ReflectiveIdentificationAgent(_BaseIdentificationAgent[M]):
    """Wraps any _BaseIdentificationAgent in an extract→validate→filter pass.

    Extracts once via the inner agent, then runs a single validation call that
    checks each item against the inner agent's own extraction rules. nodes that
    fail are discarded. If the validation call itself fails (retries exhausted or
    review count mismatch), all extracted nodes are returned as-is so the agent
    degrades gracefully to shallow behaviour.
    """

    def __init__(
        self,
        inner: _BaseIdentificationAgent[M],
        review_llm: BaseChatModel,
    ) -> None:
        super().__init__(
            llm=inner._llm,
            config=inner._config,
            grouped_identification_types=inner._grouped_identification_types,
        )
        self._inner = inner  # type: ignore[assignment]
        self._review_llm = review_llm

    @property
    def name(self) -> str:
        return f"Reflective{self._inner.name}"

    @property
    def concept_type(self) -> ConceptType:
        return self._inner.concept_type

    @property
    def output_schema(self) -> type[ConceptList[M]]:
        return self._inner.output_schema

    @property
    def _system_prompt(self) -> str:
        return self._inner._system_prompt

    def prompt_texts(self) -> dict[str, str]:
        return self._inner.prompt_texts() | {"validate": _VALIDATE_PROMPT}

    async def identify(
        self, transcript: str, kb_hint: str, transcript_file: str
    ) -> list[AnchoredConcept[M]] | list[ConceptGroup[M]]:
        nodes = await self._inner._identify(transcript, kb_hint, transcript_file)
        if not nodes:
            logger.debug("%s: identification returned no nodes", self.name)
            return []

        try:
            validation_result = await self._validate(nodes)
        except _SelfReviewRetryExhaustedError:
            logger.warning("%s: validation exhausted retries — returning all identified nodes", self.name)
            return nodes

        passing = self._filter(nodes, validation_result)

        if self._inner.grouped_identification:
            return await self._inner._group_identification(passing)

        return passing

    async def _validate(self, nodes: list[AnchoredConcept[M]]) -> SelfReviewResult:
        nodes_json = json.dumps(
            [{"title": c.item.title, "description": c.item.description, "quote": c.item.quote} for c in nodes],
            indent=2,
        )
        messages = [
            SystemMessage(
                content=[{"type": "text", "text": self._system_prompt, "cache_control": {"type": "ephemeral"}}]
            ),
            HumanMessage(content=_VALIDATE_PROMPT.format(nodes_json=nodes_json)),
        ]
        return await self._inner._retryable_structured_ainvoke(
            messages=messages,
            response_model=SelfReviewResult,
            raise_on_exhaustion=_SelfReviewRetryExhaustedError(
                f"{self.name} validate exhausted retries for {self.concept_type}"
            ),
            on_error_log_prefix=f"{self.name}.validate",
            llm=self._review_llm,
        )

    def _filter(self, nodes: list[AnchoredConcept[M]], validation: SelfReviewResult) -> list[AnchoredConcept[M]]:
        if len(validation.reviews) != len(nodes):
            logger.warning(
                "%s: review count mismatch (%d reviews / %d nodes). Returning all nodes",
                self.name,
                len(validation.reviews),
                len(nodes),
            )
            return nodes

        passing = [item for item, review in zip(nodes, validation.reviews, strict=True) if review.passed]
        failed_rationales = [
            f"item {i} ({item.item.title!r}): {review.rationale}"
            for i, (item, review) in enumerate(zip(nodes, validation.reviews, strict=True))
            if not review.passed
        ]
        if failed_rationales:
            logger.debug(
                "%s: discarded %d/%d nodes. Detail: %s",
                self.name,
                len(nodes) - len(passing),
                len(nodes),
                "; ".join(failed_rationales),
            )
        return passing
