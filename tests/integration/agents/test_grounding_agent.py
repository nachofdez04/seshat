import pytest

from seshat.agents.grounding import GroundingAgent
from tests.integration.conftest import SKIP_IF_NO_LLM_API

pytestmark = [pytest.mark.integration, pytest.mark.agents, pytest.mark.llm, SKIP_IF_NO_LLM_API]


class TestGroundingAgent:
    async def test_verify_returns_supported_true_for_matching_quote(self, cheap_llm, grounding_config):
        agent = GroundingAgent(llm=cheap_llm, config=grounding_config)

        result = await agent.verify(
            title="Use PostgreSQL",
            description="The team agreed to go with PostgreSQL.",
            quote="Agreed. Let's go with PostgreSQL.",
        )

        assert result.supported is True

    async def test_verify_returns_supported_true_for_implicit_consensus(self, cheap_llm, grounding_config):
        # Per agent rules, paraphrasing or inferring from context is supported=True even without an explicit "decided".
        # This exercises a different judgment boundary from the explicit-agreement test above.
        agent = GroundingAgent(llm=cheap_llm, config=grounding_config)

        result = await agent.verify(
            title="Use PostgreSQL for the user database",
            description="The team decided to use PostgreSQL for the user database.",
            quote="Everyone agrees PostgreSQL is the right call given our JSONB requirements.",
        )

        assert result.supported is True

    async def test_verify_returns_supported_false_for_contradicting_quote(self, cheap_llm, grounding_config):
        agent = GroundingAgent(llm=cheap_llm, config=grounding_config)

        result = await agent.verify(
            title="Use PostgreSQL",
            description="The team decided to use PostgreSQL for the user database.",
            quote="We evaluated both options and agreed MySQL is the better fit for our workload.",
        )

        assert result.supported is False
