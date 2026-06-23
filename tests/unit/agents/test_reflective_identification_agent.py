from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from seshat.agents.identification.decision import DecisionIdentificationAgent
from seshat.agents.identification.reflective import (
    NodeReview,
    ReflectiveIdentificationAgent,
    SelfReviewResult,
    _SelfReviewRetryExhaustedError,
)
from seshat.config.settings import IdentificationLLMConfig
from seshat.models.enums import ConceptType
from tests.helpers import make_anchored_concept, make_structured_llm


def _make_inner() -> DecisionIdentificationAgent:
    return DecisionIdentificationAgent(
        llm=make_structured_llm(),
        config=IdentificationLLMConfig(),
        grouped_identification_types=set(),
    )


def _all_pass(n: int) -> SelfReviewResult:
    return SelfReviewResult(reviews=[NodeReview(passed=True) for _ in range(n)])


def _mixed(passed: list[bool], rationales: list[str | None] | None = None) -> SelfReviewResult:
    rationales = rationales or [None] * len(passed)
    reviews = [NodeReview(passed=p, rationale=r) for p, r in zip(passed, rationales, strict=True)]
    return SelfReviewResult(reviews=reviews)


class TestReflectiveIdentificationAgent:
    async def test_returns_all_items_when_validation_passes(self):
        items = [make_anchored_concept("Use PostgreSQL"), make_anchored_concept("Hire contractor")]
        inner = _make_inner()
        inner._identify = AsyncMock(return_value=items)
        review_llm = make_structured_llm(return_value=_all_pass(2))

        agent = ReflectiveIdentificationAgent(inner=inner, review_llm=review_llm)
        result = await agent.identify("transcript", "hint", "file.txt")

        assert result == items
        assert inner._identify.call_count == 1

    async def test_filters_failed_items(self):
        good = make_anchored_concept("Use PostgreSQL")
        bad = make_anchored_concept("Vague preference")
        inner = _make_inner()
        inner._identify = AsyncMock(return_value=[good, bad])
        review_llm = make_structured_llm(
            return_value=_mixed([True, False], [None, "Not a decision — this is a preference."])
        )

        agent = ReflectiveIdentificationAgent(inner=inner, review_llm=review_llm)
        result = await agent.identify("transcript", "hint", "file.txt")

        assert result == [good]

    async def test_returns_empty_when_all_items_fail(self):
        items = [make_anchored_concept("Not a decision")]
        inner = _make_inner()
        inner._identify = AsyncMock(return_value=items)
        review_llm = make_structured_llm(return_value=_mixed([False], ["Wrong concept type."]))

        agent = ReflectiveIdentificationAgent(inner=inner, review_llm=review_llm)
        result = await agent.identify("transcript", "hint", "file.txt")

        assert result == []

    async def test_returns_empty_when_extraction_returns_nothing(self):
        inner = _make_inner()
        inner._identify = AsyncMock(return_value=[])
        review_llm = make_structured_llm()

        agent = ReflectiveIdentificationAgent(inner=inner, review_llm=review_llm)
        result = await agent.identify("transcript", "hint", "file.txt")

        assert result == []
        review_llm.assert_not_called()

    async def test_falls_back_to_all_items_on_validation_exhaustion(self):
        items = [make_anchored_concept("Use PostgreSQL")]
        inner = _make_inner()
        inner._identify = AsyncMock(return_value=items)
        inner._retryable_structured_ainvoke = AsyncMock(side_effect=_SelfReviewRetryExhaustedError("exhausted"))

        agent = ReflectiveIdentificationAgent(inner=inner, review_llm=MagicMock())
        result = await agent.identify("transcript", "hint", "file.txt")

        assert result == items

    async def test_falls_back_to_all_items_on_review_count_mismatch(self):
        items = [make_anchored_concept("A"), make_anchored_concept("B")]
        inner = _make_inner()
        inner._identify = AsyncMock(return_value=items)
        # Returns only 1 review for 2 items
        review_llm = make_structured_llm(return_value=_all_pass(1))

        agent = ReflectiveIdentificationAgent(inner=inner, review_llm=review_llm)
        result = await agent.identify("transcript", "hint", "file.txt")

        assert result == items

    def test_delegates_concept_type_to_inner(self):
        inner = _make_inner()
        agent = ReflectiveIdentificationAgent(inner=inner, review_llm=MagicMock())
        assert agent.concept_type == ConceptType.DECISION

    def test_delegates_system_prompt_to_inner(self):
        inner = _make_inner()
        agent = ReflectiveIdentificationAgent(inner=inner, review_llm=MagicMock())
        assert agent._system_prompt == inner._system_prompt
