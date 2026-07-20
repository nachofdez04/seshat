from unittest.mock import AsyncMock, MagicMock

import pytest

from seshat.app.agents.identification.base import IdentificationRetryExhaustedError
from seshat.app.agents.identification.decision import DecisionIdentificationAgent
from seshat.app.agents.identification.reflective import ReflectiveIdentificationAgent
from seshat.app.agents.identification.registry import IdentificationRegistry
from seshat.core.config.settings import ExtractionConfig, ReflectiveLLMConfig
from seshat.core.models.enums import ConceptType

ALL_TYPES = list(ConceptType)


class TestIdentificationRegistry:
    def _make_registry(self) -> IdentificationRegistry:
        return IdentificationRegistry(llm=MagicMock(), config=ExtractionConfig())

    def test_get_returns_correct_agent_for_each_type(self):
        registry = self._make_registry()
        assert registry.get(ConceptType.DECISION).concept_type == ConceptType.DECISION
        assert registry.get(ConceptType.RISK).concept_type == ConceptType.RISK
        assert registry.get(ConceptType.OPEN_QUESTION).concept_type == ConceptType.OPEN_QUESTION
        assert registry.get(ConceptType.ACTION_ITEM).concept_type == ConceptType.ACTION_ITEM

    def test_get_raises_for_unknown_type(self):
        registry = self._make_registry()
        registry._agents.clear()
        with pytest.raises(KeyError):
            registry.get(ConceptType.DECISION)

    def test_returns_reflective_agent_when_self_review_enabled(self):
        config = ExtractionConfig(identification_self_review=ReflectiveLLMConfig(enabled=True))
        registry = IdentificationRegistry(llm=MagicMock(), config=config, review_llm=MagicMock())

        assert isinstance(registry.get(ConceptType.DECISION), ReflectiveIdentificationAgent)

    def test_returns_shallow_agent_when_self_review_disabled(self):
        registry = self._make_registry()

        assert isinstance(registry.get(ConceptType.DECISION), DecisionIdentificationAgent)


class TestIdentificationRegistryRunAll:
    def _make_registry(self) -> IdentificationRegistry:
        return IdentificationRegistry(llm=MagicMock(), config=ExtractionConfig())

    async def test_fans_out_over_given_types_and_collects_by_type(self):
        registry = self._make_registry()
        for ct in ALL_TYPES:
            registry.get(ct).identify = AsyncMock(return_value=[f"{ct.value}-item"])

        results, failed = await registry.run_all("transcript", "blob.txt", hints={ConceptType.DECISION: "hint-d"})

        assert failed == []
        assert results == {ct: [f"{ct.value}-item"] for ct in ALL_TYPES}

    async def test_runs_only_requested_subset(self):
        registry = self._make_registry()
        for ct in ALL_TYPES:
            registry.get(ct).identify = AsyncMock(return_value=[])

        await registry.run_all("transcript", "blob.txt", hints={}, concept_types=[ConceptType.RISK])

        registry.get(ConceptType.RISK).identify.assert_awaited_once()
        registry.get(ConceptType.DECISION).identify.assert_not_called()

    async def test_passes_per_type_hint_to_each_agent(self):
        registry = self._make_registry()
        for ct in ALL_TYPES:
            registry.get(ct).identify = AsyncMock(return_value=[])

        await registry.run_all(
            "the transcript",
            "blob.txt",
            hints={ConceptType.RISK: "risk-hint"},
            concept_types=[ConceptType.RISK],
        )

        registry.get(ConceptType.RISK).identify.assert_awaited_once_with("the transcript", "risk-hint", "blob.txt")

    async def test_missing_hint_defaults_to_empty_string(self):
        registry = self._make_registry()
        registry.get(ConceptType.RISK).identify = AsyncMock(return_value=[])

        await registry.run_all("t", "b", hints={}, concept_types=[ConceptType.RISK])

        registry.get(ConceptType.RISK).identify.assert_awaited_once_with("t", "", "b")

    @pytest.mark.parametrize(
        "exc",
        [RuntimeError("boom"), IdentificationRetryExhaustedError("agent RISK exhausted 3 retries")],
    )
    async def test_one_failing_type_is_reported_and_others_survive(self, exc):
        registry = self._make_registry()
        for ct in ALL_TYPES:
            registry.get(ct).identify = AsyncMock(return_value=[f"{ct.value}-item"])
        registry.get(ConceptType.RISK).identify = AsyncMock(side_effect=exc)

        results, failed = await registry.run_all("t", "b", hints={})

        assert failed == [ConceptType.RISK]
        assert ConceptType.RISK not in results
        assert results[ConceptType.DECISION] == ["decision-item"]
