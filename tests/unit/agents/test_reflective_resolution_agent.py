from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from seshat.agents.resolution.base import ResolvedRelationship
from seshat.agents.resolution.same_type.decision import DecisionResolutionAgent, _DecisionEntry
from seshat.agents.resolution.same_type.reflective import (
    ReflectiveResolutionAgent,
    TiebreakerEntry,
    TiebreakerResult,
    _SelfReviewRetryExhaustedError,
)
from seshat.config.settings import ResolutionLLMConfig
from seshat.models.enums import RelationshipType
from tests.helpers import make_node, make_structured_llm


def _make_inner() -> DecisionResolutionAgent:
    return DecisionResolutionAgent(llm=make_structured_llm(), config=ResolutionLLMConfig())


def _entry(
    src_idx: str = "0",
    tgt_idx: str = "1",
    rel_type=RelationshipType.SUPERSEDES,
    alt_rel_type=None,
) -> _DecisionEntry:
    return _DecisionEntry(
        source_id=src_idx,
        target_id=tgt_idx,
        rel_type=rel_type,
        alt_rel_type=alt_rel_type,
        rationale="test",
    )


def _id_map(src_id, tgt_id) -> dict:
    return {"0": src_id, "1": tgt_id}


def _tiebreaker(choices: list[tuple[str, str]]) -> TiebreakerResult:
    return TiebreakerResult(decisions=[TiebreakerEntry(chosen=c, rationale=r) for c, r in choices])


class TestReflectiveResolutionAgentRunForSource:
    async def test_no_contested_entries_skips_tiebreaker(self):
        src = make_node("src")
        tgt = make_node("tgt")
        inner = _make_inner()
        entries = [_entry()]
        inner._run_for_source = AsyncMock(return_value=(entries, _id_map(src.id, tgt.id)))
        review_llm = make_structured_llm()

        agent = ReflectiveResolutionAgent(inner=inner, review_llm=review_llm)
        result_entries, _ = await agent._run_for_source(source=src, targets=[tgt])

        assert result_entries == entries
        review_llm.with_structured_output.assert_not_called()

    async def test_empty_inner_result_skips_tiebreaker(self):
        src = make_node("src")
        tgt = make_node("tgt")
        inner = _make_inner()
        inner._run_for_source = AsyncMock(return_value=([], {}))
        review_llm = make_structured_llm()

        agent = ReflectiveResolutionAgent(inner=inner, review_llm=review_llm)
        result_entries, _ = await agent._run_for_source(source=src, targets=[tgt])

        assert result_entries == []
        review_llm.with_structured_output.assert_not_called()

    async def test_contested_entry_tiebreaker_overwrites_rel_type(self):
        src = make_node("src")
        tgt = make_node("tgt")
        inner = _make_inner()
        contested = _entry(rel_type=RelationshipType.SUPERSEDES, alt_rel_type=RelationshipType.AMENDS)
        inner._run_for_source = AsyncMock(return_value=([contested], _id_map(src.id, tgt.id)))
        review_llm = make_structured_llm(return_value=_tiebreaker([("amends", "partial update")]))

        agent = ReflectiveResolutionAgent(inner=inner, review_llm=review_llm)
        result_entries, _ = await agent._run_for_source(source=src, targets=[tgt])

        assert len(result_entries) == 1
        assert result_entries[0].rel_type == RelationshipType.AMENDS

    async def test_uncontested_entries_unchanged_contested_entry_updated(self):
        src = make_node("src")
        tgt1 = make_node("tgt1")
        tgt2 = make_node("tgt2")
        inner = _make_inner()
        uncontested = _entry("0", "1", RelationshipType.SUPERSEDES, alt_rel_type=None)
        contested = _entry("0", "2", RelationshipType.AMENDS, alt_rel_type=RelationshipType.CONFLICTS_WITH)
        inner._run_for_source = AsyncMock(
            return_value=([uncontested, contested], {"0": src.id, "1": tgt1.id, "2": tgt2.id})
        )
        review_llm = make_structured_llm(return_value=_tiebreaker([("conflicts_with", "contradiction")]))

        agent = ReflectiveResolutionAgent(inner=inner, review_llm=review_llm)
        result_entries, _ = await agent._run_for_source(source=src, targets=[tgt1, tgt2])

        assert len(result_entries) == 2
        assert result_entries[0].rel_type == RelationshipType.SUPERSEDES
        assert result_entries[1].rel_type == RelationshipType.CONFLICTS_WITH
        # tiebreaker called once (for the single contested entry)
        review_llm.with_structured_output.assert_called_once()

    async def test_tiebreaker_invalid_chosen_keeps_original_rel_type(self):
        src = make_node("src")
        tgt = make_node("tgt")
        inner = _make_inner()
        contested = _entry(rel_type=RelationshipType.SUPERSEDES, alt_rel_type=RelationshipType.AMENDS)
        inner._run_for_source = AsyncMock(return_value=([contested], _id_map(src.id, tgt.id)))
        review_llm = make_structured_llm(return_value=_tiebreaker([("not_a_real_type", "bad")]))

        agent = ReflectiveResolutionAgent(inner=inner, review_llm=review_llm)
        result_entries, _ = await agent._run_for_source(source=src, targets=[tgt])

        assert result_entries[0].rel_type == RelationshipType.SUPERSEDES

    async def test_count_mismatch_keeps_all_originals(self):
        src = make_node("src")
        tgt1 = make_node("tgt1")
        tgt2 = make_node("tgt2")
        inner = _make_inner()
        contested1 = _entry("0", "1", RelationshipType.SUPERSEDES, alt_rel_type=RelationshipType.AMENDS)
        contested2 = _entry("0", "2", RelationshipType.AMENDS, alt_rel_type=RelationshipType.CONFLICTS_WITH)
        inner._run_for_source = AsyncMock(
            return_value=([contested1, contested2], {"0": src.id, "1": tgt1.id, "2": tgt2.id})
        )
        review_llm = make_structured_llm(return_value=_tiebreaker([("amends", "only one")]))

        agent = ReflectiveResolutionAgent(inner=inner, review_llm=review_llm)
        result_entries, _ = await agent._run_for_source(source=src, targets=[tgt1, tgt2])

        assert result_entries[0].rel_type == RelationshipType.SUPERSEDES
        assert result_entries[1].rel_type == RelationshipType.AMENDS

    async def test_retry_exhaustion_keeps_all_originals(self):
        src = make_node("src")
        tgt = make_node("tgt")
        inner = _make_inner()
        contested = _entry(rel_type=RelationshipType.SUPERSEDES, alt_rel_type=RelationshipType.AMENDS)
        inner._run_for_source = AsyncMock(return_value=([contested], _id_map(src.id, tgt.id)))
        inner._retryable_structured_ainvoke = AsyncMock(side_effect=_SelfReviewRetryExhaustedError("exhausted"))

        agent = ReflectiveResolutionAgent(inner=inner, review_llm=MagicMock())
        result_entries, _ = await agent._run_for_source(source=src, targets=[tgt])

        assert result_entries[0].rel_type == RelationshipType.SUPERSEDES


class TestReflectiveResolutionAgentDelegation:
    def test_delegates_system_prompt_to_inner(self):
        inner = _make_inner()
        agent = ReflectiveResolutionAgent(inner=inner, review_llm=MagicMock())
        assert agent._system_prompt == inner._system_prompt

    def test_delegates_result_model_to_inner(self):
        inner = _make_inner()
        agent = ReflectiveResolutionAgent(inner=inner, review_llm=MagicMock())
        assert agent._result_model is inner._result_model

    def test_delegates_validate_relationships_to_inner(self):
        inner = _make_inner()
        agent = ReflectiveResolutionAgent(inner=inner, review_llm=MagicMock())
        src, tgt = uuid4(), uuid4()
        rels = [
            ResolvedRelationship(source_id=src, target_id=tgt, rel_type=RelationshipType.SUPERSEDES, rationale="test"),
            ResolvedRelationship(source_id=tgt, target_id=src, rel_type=RelationshipType.SUPERSEDES, rationale="test"),
        ]
        valid, dropped = agent._validate_relationships(rels)
        assert valid == []
        assert len(dropped) == 2

    def test_prompt_texts_includes_tiebreaker_key(self):
        inner = _make_inner()
        agent = ReflectiveResolutionAgent(inner=inner, review_llm=MagicMock())
        texts = agent.prompt_texts()
        assert "tiebreaker" in texts
        assert "system" in texts
