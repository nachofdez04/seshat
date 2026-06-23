import pytest

from seshat.agents.identification.decision import DecisionIdentificationAgent
from seshat.agents.identification.reflective import ReflectiveIdentificationAgent
from seshat.models.enums import ConceptType
from tests.integration.conftest import SKIP_IF_NO_LLM_API

pytestmark = [pytest.mark.integration, pytest.mark.llm, pytest.mark.agents, SKIP_IF_NO_LLM_API]

_TRANSCRIPT_FILE = "test_meeting.txt"


class TestReflectiveIdentificationAgent:
    def _make_agent(self, cheap_llm, extraction_config) -> ReflectiveIdentificationAgent:
        inner = DecisionIdentificationAgent(
            llm=cheap_llm,
            config=extraction_config.identification,
            grouped_identification_types=set(),
        )
        return ReflectiveIdentificationAgent(inner=inner, review_llm=cheap_llm)

    async def test_identifies_clear_decision_end_to_end(self, cheap_llm, extraction_config):
        transcript = (
            "The team reviewed the database options. PostgreSQL was proposed because of its native JSONB support, "
            "which the metadata store requires. MySQL was considered but ruled out because it lacks first-class JSON "
            "indexing. The team agreed to use PostgreSQL for the user database and closed the discussion."
        )
        agent = self._make_agent(cheap_llm, extraction_config)

        result = await agent.identify(transcript, kb_hint="", transcript_file=_TRANSCRIPT_FILE)

        assert len(result) >= 1
        assert agent.concept_type == ConceptType.DECISION

    async def test_returns_empty_for_non_extractable_transcript(self, cheap_llm, extraction_config):
        transcript = "The weather today is sunny. Everyone agrees it feels like spring."
        agent = self._make_agent(cheap_llm, extraction_config)

        result = await agent.identify(transcript, kb_hint="", transcript_file=_TRANSCRIPT_FILE)

        assert result == []
