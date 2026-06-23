import logging

import pytest

from seshat.agents.grounding import GroundingAgent, GroundingResult, GroundingRetryExhaustedError
from seshat.config.settings import GroundingLLMConfig
from tests.helpers import make_structured_llm


def _make_agent(
    return_value=None,
    side_effect=None,
    max_retries: int = 2,
    use_full_transcript: bool = True,
) -> GroundingAgent:
    return GroundingAgent(
        llm=make_structured_llm(return_value=return_value, side_effect=side_effect),
        config=GroundingLLMConfig(max_retries=max_retries, use_full_transcript=use_full_transcript),
    )


class TestGroundingAgent:
    async def test_empty_quote_returns_unsupported_without_calling_llm(self):
        llm = make_structured_llm()
        agent = GroundingAgent(llm=llm, config=GroundingLLMConfig())

        result = await agent.verify(title="T", description="D", quote="")

        assert result.supported is False
        llm.with_structured_output.assert_not_called()

    async def test_returns_llm_result_on_success(self):
        expected = GroundingResult(supported=True, rationale="Directly stated.")
        agent = _make_agent(return_value=expected)

        result = await agent.verify(title="T", description="D", quote="some quote")

        assert result.supported is True
        assert result.rationale == "Directly stated."

    async def test_raises_after_all_retries_fail(self):
        agent = _make_agent(side_effect=Exception("LLM error"), max_retries=3)

        with pytest.raises(GroundingRetryExhaustedError):
            await agent.verify(title="T", description="D", quote="some quote")

    async def test_use_full_transcript_true_includes_transcript_in_messages(self):
        agent = _make_agent(return_value=GroundingResult(supported=True, rationale=None), use_full_transcript=True)

        await agent.verify(title="T", description="D", quote="some quote", transcript="full transcript text")

        messages = agent._llm.with_structured_output.return_value.ainvoke.call_args[0][0]
        combined = " ".join(str(m.content) for m in messages)
        assert "full transcript text" in combined

    async def test_use_full_transcript_false_ignores_transcript(self):
        agent = _make_agent(return_value=GroundingResult(supported=True, rationale=None), use_full_transcript=False)

        await agent.verify(title="T", description="D", quote="some quote", transcript="full transcript text")

        messages = agent._llm.with_structured_output.return_value.ainvoke.call_args[0][0]
        combined = " ".join(str(m.content) for m in messages)
        assert "full transcript text" not in combined

    async def test_use_full_transcript_false_emits_warning(self, caplog):
        expected = GroundingResult(supported=True, rationale=None)
        agent = _make_agent(return_value=expected, use_full_transcript=False)

        with caplog.at_level(logging.WARNING, logger="seshat.agents.grounding"):
            await agent.verify(title="T", description="D", quote="q", transcript="full text")

        assert any("use_full_transcript=False" in r.message for r in caplog.records)

    async def test_use_full_transcript_false_no_warning_when_transcript_not_passed(self, caplog):
        expected = GroundingResult(supported=True, rationale=None)
        agent = _make_agent(return_value=expected, use_full_transcript=False)

        with caplog.at_level(logging.WARNING, logger="seshat.agents.grounding"):
            await agent.verify(title="T", description="D", quote="q")

        assert not any("use_full_transcript=False" in r.message for r in caplog.records)
