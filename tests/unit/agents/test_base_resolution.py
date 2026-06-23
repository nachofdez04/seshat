from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from seshat.agents.resolution.base import (
    ResolutionRetryExhaustedError,
    ResolvedRelationship,
    _EntryBase,
)
from seshat.agents.resolution.same_type.decision import DecisionResolutionAgent, _DecisionEntry, _DecisionResult
from seshat.config.settings import ResolutionLLMConfig
from seshat.models.enums import RelationshipType
from tests.helpers import make_node, make_structured_llm


def _make_agent() -> DecisionResolutionAgent:
    return DecisionResolutionAgent(llm=make_structured_llm(), config=ResolutionLLMConfig())


def _entry(source_id: str, target_id: str, rel_type: str | None) -> _EntryBase:
    return _EntryBase(source_id=source_id, target_id=target_id, rel_type=rel_type, rationale="test")


def _make_id_map(*uuids) -> dict[str, UUID]:
    return {str(i): uid for i, uid in enumerate(uuids)}


class TestToRelationships:
    def test_valid_entry_produces_resolved_relationship(self):
        agent = _make_agent()
        src, tgt = uuid4(), uuid4()
        entries = [_entry("0", "1", "supersedes")]

        result = agent._to_relationships(entries, _make_id_map(src, tgt))

        assert len(result) == 1
        assert result[0].source_id == src
        assert result[0].target_id == tgt
        assert result[0].rel_type == RelationshipType.SUPERSEDES

    @pytest.mark.parametrize("rel_type", [None, "not_a_real_type"])
    def test_invalid_rel_type_is_skipped(self, rel_type):
        agent = _make_agent()
        entries = [_entry("0", "1", rel_type)]

        result = agent._to_relationships(entries, _make_id_map(uuid4(), uuid4()))

        assert result == []

    def test_unknown_index_is_skipped(self):
        agent = _make_agent()
        entries = [_entry("0", "99", "supersedes")]

        result = agent._to_relationships(entries, _make_id_map(uuid4()))

        assert result == []

    def test_mixed_entries_only_valid_ones_returned(self):
        agent = _make_agent()
        src, tgt = uuid4(), uuid4()
        entries = [
            _entry("0", "1", "supersedes"),
            _entry("0", "1", None),
            _entry("0", "99", "amends"),
        ]

        result = agent._to_relationships(entries, _make_id_map(src, tgt))

        assert len(result) == 1
        assert result[0].rel_type == RelationshipType.SUPERSEDES


def _rel(source_id, target_id, rel_type: RelationshipType) -> ResolvedRelationship:
    return ResolvedRelationship(source_id=source_id, target_id=target_id, rel_type=rel_type, rationale="test")


class TestValidateRelationships:
    def test_empty_input_returns_empty(self):
        agent = _make_agent()
        valid, dropped = agent._validate_relationships([])
        assert valid == []
        assert dropped == []

    def test_valid_relationships_pass_through(self):
        agent = _make_agent()
        src, tgt = uuid4(), uuid4()
        rels = [_rel(src, tgt, RelationshipType.SUPERSEDES)]

        valid, dropped = agent._validate_relationships(rels)

        assert len(valid) == 1
        assert dropped == []

    def test_anti_symmetry_drops_both_directions(self):
        agent = _make_agent()
        a, b = uuid4(), uuid4()
        rels = [
            _rel(a, b, RelationshipType.SUPERSEDES),
            _rel(b, a, RelationshipType.SUPERSEDES),
        ]

        valid, dropped = agent._validate_relationships(rels)

        assert valid == []
        assert len(dropped) == 2

    def test_anti_symmetry_only_applies_per_rel_type(self):
        # A→B SUPERSEDES + B→A AMENDS: AMENDS is not anti-symmetric, so B→A is not a violation.
        agent = _make_agent()
        a, b = uuid4(), uuid4()
        rels = [
            _rel(a, b, RelationshipType.SUPERSEDES),
            _rel(b, a, RelationshipType.AMENDS),
        ]

        valid, dropped = agent._validate_relationships(rels)

        assert len(valid) == 2
        assert dropped == []

    def test_mutual_exclusion_keeps_amends_over_supersedes(self):
        agent = _make_agent()
        src, tgt = uuid4(), uuid4()
        rels = [
            _rel(src, tgt, RelationshipType.SUPERSEDES),
            _rel(src, tgt, RelationshipType.AMENDS),
        ]

        valid, dropped = agent._validate_relationships(rels)

        assert len(valid) == 1
        assert valid[0].rel_type == RelationshipType.AMENDS
        assert len(dropped) == 1
        assert dropped[0].rel_type == RelationshipType.SUPERSEDES

    def test_mutual_exclusion_keeps_amends_over_supersedes_and_conflicts_with(self):
        agent = _make_agent()
        src, tgt = uuid4(), uuid4()
        rels = [
            _rel(src, tgt, RelationshipType.SUPERSEDES),
            _rel(src, tgt, RelationshipType.CONFLICTS_WITH),
            _rel(src, tgt, RelationshipType.AMENDS),
        ]

        valid, dropped = agent._validate_relationships(rels)

        assert len(valid) == 1
        assert valid[0].rel_type == RelationshipType.AMENDS
        assert len(dropped) == 2

    def test_non_conflicting_different_pairs_both_kept(self):
        agent = _make_agent()
        a, b, c = uuid4(), uuid4(), uuid4()
        rels = [
            _rel(a, b, RelationshipType.SUPERSEDES),
            _rel(a, c, RelationshipType.AMENDS),
        ]

        valid, dropped = agent._validate_relationships(rels)

        assert len(valid) == 2
        assert dropped == []


def _make_agent_with_llm(return_value=None, side_effect=None, max_retries: int = 2) -> DecisionResolutionAgent:
    return DecisionResolutionAgent(
        llm=make_structured_llm(return_value=return_value, side_effect=side_effect),
        config=ResolutionLLMConfig(max_retries=max_retries),
    )


def _decision_result(source_id: str, target_id: str, rel_type: RelationshipType) -> _DecisionResult:
    from seshat.agents.resolution.same_type.decision import _DecisionEntry

    entry = _DecisionEntry(source_id=source_id, target_id=target_id, rel_type=rel_type, rationale="test")
    return _DecisionResult(entries=[entry])


class TestResolve:
    async def test_empty_source_nodes_returns_empty(self):
        agent = _make_agent_with_llm()
        node = make_node("n1")

        rels, failed = await agent.resolve(source_nodes=[], per_source_targets={node.id: [node]})

        assert rels == []
        assert failed == []
        assert agent._llm.with_structured_output.call_count == 0

    async def test_empty_per_source_targets_returns_empty(self):
        agent = _make_agent_with_llm()
        node = make_node("n1")

        rels, failed = await agent.resolve(source_nodes=[node], per_source_targets={})

        assert rels == []
        assert failed == []
        assert agent._llm.with_structured_output.call_count == 0

    async def test_successful_resolution_returns_relationships(self):
        src = make_node("src")
        tgt = make_node("tgt")
        result_schema = _decision_result("0", "1", RelationshipType.SUPERSEDES)
        agent = _make_agent_with_llm(return_value=result_schema)

        rels, failed = await agent.resolve(source_nodes=[src], per_source_targets={src.id: [tgt]})

        assert len(rels) == 1
        assert rels[0].rel_type == RelationshipType.SUPERSEDES
        assert rels[0].source_id == src.id
        assert rels[0].target_id == tgt.id
        assert failed == []

    async def test_source_excluded_from_its_own_targets(self):
        node = make_node("n1")
        agent = _make_agent_with_llm()

        rels, failed = await agent.resolve(source_nodes=[node], per_source_targets={node.id: [node]})

        assert rels == []
        assert failed == []
        assert agent._llm.with_structured_output.call_count == 0

    async def test_per_source_candidates_restricts_targets(self):
        src = make_node("src")
        included = make_node("included")
        excluded = make_node("excluded")
        result_schema = _decision_result("0", "1", RelationshipType.SUPERSEDES)
        agent = _make_agent_with_llm(return_value=result_schema)

        rels, _ = await agent.resolve(source_nodes=[src], per_source_targets={src.id: [included]})

        assert len(rels) == 1
        assert rels[0].target_id == included.id
        call_args = agent._llm.with_structured_output.return_value.ainvoke.call_args
        context_str = call_args[0][0][1].content
        assert excluded.id.hex not in context_str

    async def test_failed_task_is_skipped_other_results_kept(self):
        src1 = make_node("src1")
        src2 = make_node("src2")
        tgt = make_node("tgt")

        good_result = _decision_result("0", "1", RelationshipType.AMENDS)
        agent = _make_agent_with_llm(
            side_effect=[good_result, RuntimeError("LLM failure")],
            max_retries=1,
        )

        rels, failed = await agent.resolve(
            source_nodes=[src1, src2], per_source_targets={src1.id: [tgt], src2.id: [tgt]}
        )

        assert len(rels) == 1
        assert rels[0].rel_type == RelationshipType.AMENDS
        assert len(failed) == 1
        assert failed[0].node_id == src2.id


class TestIndexIdMapping:
    """LLM receives positional indices as IDs; _to_relationships restores full UUIDs via id_map."""

    async def test_context_uses_positional_indices_not_uuids(self):
        src = make_node("src")
        tgt = make_node("tgt")
        agent = _make_agent_with_llm(return_value=_decision_result("0", "1", RelationshipType.SUPERSEDES))

        await agent._run_for_source(source=src, targets=[tgt])

        call_args = agent._llm.with_structured_output.return_value.ainvoke.call_args
        context_str = call_args[0][0][1].content
        assert '"id": "0"' in context_str
        assert '"id": "1"' in context_str
        assert src.id.hex not in context_str
        assert tgt.id.hex not in context_str

    async def test_positional_indices_resolve_to_correct_uuids(self):
        src = make_node("src")
        tgt = make_node("tgt")
        agent = _make_agent_with_llm(return_value=_decision_result("0", "1", RelationshipType.SUPERSEDES))

        # resolve() applies _to_relationships — UUIDs are resolved in the final output
        rels, _ = await agent.resolve(source_nodes=[src], per_source_targets={src.id: [tgt]})

        assert len(rels) == 1
        assert rels[0].source_id == src.id
        assert rels[0].target_id == tgt.id


class TestSameTypeEntryAltRelType:
    def test_alt_rel_type_same_as_rel_type_rejected(self):
        with pytest.raises(ValidationError):
            _DecisionEntry(
                source_id="0",
                target_id="1",
                rel_type=RelationshipType.SUPERSEDES,
                alt_rel_type=RelationshipType.SUPERSEDES,
                rationale="test",
            )

    def test_alt_rel_type_invalid_value_rejected(self):
        with pytest.raises(ValidationError):
            _DecisionEntry(
                source_id="0",
                target_id="1",
                rel_type=RelationshipType.SUPERSEDES,
                alt_rel_type="mitigates",
                rationale="test",
            )

    def test_null_string_coerced_to_none(self):
        entry = _DecisionEntry(
            source_id="0",
            target_id="1",
            rel_type=RelationshipType.SUPERSEDES,
            alt_rel_type="null",
            rationale="test",
        )
        assert entry.alt_rel_type is None


class TestRunForSource:
    async def test_returns_empty_when_only_target_is_source_itself(self):
        node = make_node("n1")
        agent = _make_agent_with_llm()

        entries, _ = await agent._run_for_source(source=node, targets=[node])

        assert entries == []
        assert agent._llm.with_structured_output.call_count == 0

    async def test_successful_llm_call_returns_entries_and_id_map(self):
        src = make_node("src")
        tgt = make_node("tgt")
        result_schema = _decision_result("0", "1", RelationshipType.SUPERSEDES)
        agent = _make_agent_with_llm(return_value=result_schema)

        entries, id_map = await agent._run_for_source(source=src, targets=[tgt])

        assert len(entries) == 1
        assert entries[0].rel_type == RelationshipType.SUPERSEDES
        assert id_map["0"] == src.id
        assert id_map["1"] == tgt.id

    async def test_exhausted_retries_raises(self):
        src = make_node("src")
        tgt = make_node("tgt")
        agent = _make_agent_with_llm(side_effect=Exception("LLM error"), max_retries=3)

        with pytest.raises(ResolutionRetryExhaustedError):
            await agent._run_for_source(source=src, targets=[tgt])
