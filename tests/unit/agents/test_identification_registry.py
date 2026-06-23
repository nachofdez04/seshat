from unittest.mock import MagicMock

import pytest

from seshat.agents.identification.decision import DecisionIdentificationAgent
from seshat.agents.identification.reflective import ReflectiveIdentificationAgent
from seshat.agents.identification.registry import IdentificationAgentRegistry
from seshat.config.settings import ExtractionConfig, ReflectiveLLMConfig
from seshat.models.enums import ConceptType


class TestIdentificationAgentRegistry:
    def _make_registry(self) -> IdentificationAgentRegistry:
        return IdentificationAgentRegistry(llm=MagicMock(), config=ExtractionConfig())

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
        registry = IdentificationAgentRegistry(llm=MagicMock(), config=config, review_llm=MagicMock())

        assert isinstance(registry.get(ConceptType.DECISION), ReflectiveIdentificationAgent)

    def test_returns_shallow_agent_when_self_review_disabled(self):
        registry = self._make_registry()

        assert isinstance(registry.get(ConceptType.DECISION), DecisionIdentificationAgent)
