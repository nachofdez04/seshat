import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from seshat.agents.resolution.base import ResolvedRelationship
from seshat.agents.resolution.cross_type.registry import CrossTypeResolutionRegistry
from seshat.agents.resolution.registry import ResolutionRegistry
from seshat.agents.resolution.same_type.reflective import ReflectiveResolutionAgent
from seshat.agents.resolution.same_type.registry import SameTypeResolutionRegistry
from seshat.config.settings import ExtractionConfig
from seshat.models.enums import ConceptType, RelationshipType
from tests.helpers import make_node


def _make_same_type_registry() -> SameTypeResolutionRegistry:
    return SameTypeResolutionRegistry(llm=MagicMock(), config=ExtractionConfig())


def _make_cross_type_registry() -> CrossTypeResolutionRegistry:
    return CrossTypeResolutionRegistry(llm=MagicMock(), config=ExtractionConfig())


def _make_resolution_registry() -> ResolutionRegistry:
    return ResolutionRegistry(llm=MagicMock(), config=ExtractionConfig())


class TestSameTypeResolutionRegistry:
    def test_get_returns_agent_for_known_type(self):
        registry = _make_same_type_registry()
        for ct in (ConceptType.DECISION, ConceptType.RISK, ConceptType.ACTION_ITEM, ConceptType.OPEN_QUESTION):
            assert registry.get(ct) is not None

    def test_get_raises_for_unknown_type(self):
        registry = _make_same_type_registry()
        registry._agents.clear()
        with pytest.raises(KeyError):
            registry.get(ConceptType.DECISION)

    async def test_resolve_all_returns_empty_when_no_nodes(self):
        registry = _make_same_type_registry()
        rels, failed = await registry.resolve_all(source_nodes=[], per_source_targets={})
        assert rels == []
        assert failed == []

    async def test_resolve_all_partitions_by_type_and_collects_results(self):
        registry = _make_same_type_registry()

        rel = MagicMock(spec=ResolvedRelationship)
        rel.rel_type = RelationshipType.SUPERSEDES

        for agent in registry._agents.values():
            agent.resolve = AsyncMock(return_value=([rel], []))

        decision_src = make_node("src1", title="Use PostgreSQL")
        decision_tgt = make_node("tgt1", title="Use MySQL")

        rels, failed = await registry.resolve_all(
            source_nodes=[decision_src],
            per_source_targets={decision_src.id: [decision_tgt]},
        )

        assert len(rels) == 1
        assert rels[0].rel_type == RelationshipType.SUPERSEDES
        assert failed == []

    async def test_resolve_all_returns_empty_when_no_same_type_targets(self):
        registry = _make_same_type_registry()

        for agent in registry._agents.values():
            agent.resolve = AsyncMock(return_value=([], []))

        decision_node = make_node("n1")
        risk_node = make_node("n2")
        risk_node = risk_node.model_copy(update={"type": ConceptType.RISK})

        rels, failed = await registry.resolve_all(
            source_nodes=[decision_node],
            per_source_targets={decision_node.id: [risk_node]},
        )

        assert rels == []
        assert failed == []


class TestCrossTypeResolutionRegistry:
    def test_get_returns_agent_for_known_pair(self):
        registry = _make_cross_type_registry()
        agent = registry.get(ConceptType.DECISION, ConceptType.RISK)
        assert agent is not None

    def test_get_raises_for_unknown_pair(self):
        registry = _make_cross_type_registry()
        with pytest.raises(KeyError):
            registry.get(ConceptType.DECISION, ConceptType.DECISION)

    async def test_resolve_all_returns_empty_when_no_nodes(self):
        registry = _make_cross_type_registry()
        rels, failed = await registry.resolve_all(source_nodes=[], per_source_targets={})
        assert rels == []
        assert failed == []

    async def test_resolve_all_dispatches_to_matching_pair_agent(self):
        registry = _make_cross_type_registry()

        rel = MagicMock(spec=ResolvedRelationship)
        rel.rel_type = RelationshipType.MITIGATES

        for agent in registry._agents_mapping.values():
            agent.resolve = AsyncMock(return_value=([], []))

        decision_to_risk_agent = registry._agents_mapping[(ConceptType.DECISION, ConceptType.RISK)]
        decision_to_risk_agent.resolve = AsyncMock(return_value=([rel], []))

        decision_node = make_node("src1")
        risk_node = make_node("tgt1")
        risk_node = risk_node.model_copy(update={"type": ConceptType.RISK})

        rels, failed = await registry.resolve_all(
            source_nodes=[decision_node],
            per_source_targets={decision_node.id: [risk_node]},
        )

        assert len(rels) == 1
        assert rels[0].rel_type == RelationshipType.MITIGATES
        assert failed == []

    async def test_resolve_all_skips_pairs_with_no_matching_nodes(self):
        registry = _make_cross_type_registry()

        for agent in registry._agents_mapping.values():
            agent.resolve = AsyncMock(return_value=([], []))

        decision_node = make_node("n1")

        rels, failed = await registry.resolve_all(
            source_nodes=[decision_node],
            per_source_targets={decision_node.id: [decision_node]},
        )

        assert rels == []
        assert failed == []
        for agent in registry._agents_mapping.values():
            agent.resolve.assert_not_called()


class TestResolutionRegistry:
    @pytest.mark.parametrize("stripped_type", ["blocks"])
    async def test_invalid_rel_targeting_superseded_node_is_stripped(self, stripped_type):
        source = make_node("src")
        kb_target = make_node("kb1")

        def _make_rel(rel_type: RelationshipType):
            rel = MagicMock(spec=ResolvedRelationship)
            rel.source_id = source.id
            rel.target_id = kb_target.id
            rel.rel_type = rel_type
            return rel

        supersedes_rel = _make_rel(RelationshipType.SUPERSEDES)
        spurious_rel = _make_rel(RelationshipType(stripped_type))

        registry = _make_resolution_registry()
        registry._same_type.resolve_all = AsyncMock(return_value=([supersedes_rel, spurious_rel], []))
        registry._cross_type.resolve_all = AsyncMock(return_value=([], []))

        resolved, _ = await registry.resolve_all(source_nodes=[source], per_source_targets={source.id: [kb_target]})

        rel_types = {r.rel_type for r in resolved}
        assert RelationshipType(stripped_type) not in rel_types
        assert RelationshipType.SUPERSEDES in rel_types
        assert len(resolved) == 1


class TestGlobalSemaphoreForwarding:
    async def test_global_sem_forwarded_to_same_type_agents(self):
        registry = _make_same_type_registry()
        sem = asyncio.Semaphore(1)

        for agent in registry._agents.values():
            agent.resolve = AsyncMock(return_value=([], []))

        decision_src = make_node("src1")
        decision_tgt = make_node("tgt1")

        await registry.resolve_all(
            source_nodes=[decision_src],
            per_source_targets={decision_src.id: [decision_tgt]},
            global_sem=sem,
        )

        decision_agent = registry._agents[ConceptType.DECISION]
        call_kwargs = decision_agent.resolve.call_args
        assert call_kwargs.args[2] is sem or call_kwargs.kwargs.get("global_sem") is sem

    async def test_global_sem_forwarded_to_cross_type_agents(self):
        registry = _make_cross_type_registry()
        sem = asyncio.Semaphore(1)

        for agent in registry._agents_mapping.values():
            agent.resolve = AsyncMock(return_value=([], []))

        decision_node = make_node("src1")
        risk_node = make_node("tgt1")
        risk_node = risk_node.model_copy(update={"type": ConceptType.RISK})

        await registry.resolve_all(
            source_nodes=[decision_node],
            per_source_targets={decision_node.id: [risk_node]},
            global_sem=sem,
        )

        decision_to_risk = registry._agents_mapping[(ConceptType.DECISION, ConceptType.RISK)]
        call_kwargs = decision_to_risk.resolve.call_args
        assert call_kwargs.args[2] is sem or call_kwargs.kwargs.get("global_sem") is sem


class TestReflectiveWrappingInRegistry:
    def test_same_type_registry_wraps_agents_when_review_llm_provided(self):
        review_llm = MagicMock()
        config = ExtractionConfig(resolution_self_review={"enabled": True})
        registry = SameTypeResolutionRegistry(llm=MagicMock(), config=config, review_llm=review_llm)
        for agent in registry._agents.values():
            assert isinstance(agent, ReflectiveResolutionAgent)

    def test_same_type_registry_does_not_wrap_when_no_review_llm(self):
        registry = SameTypeResolutionRegistry(llm=MagicMock(), config=ExtractionConfig(), review_llm=None)
        for agent in registry._agents.values():
            assert not isinstance(agent, ReflectiveResolutionAgent)

    def test_cross_type_registry_never_wraps_agents(self):
        config = ExtractionConfig(resolution_self_review={"enabled": True})
        registry = CrossTypeResolutionRegistry(llm=MagicMock(), config=config)
        for agent in registry._agents_mapping.values():
            assert not isinstance(agent, ReflectiveResolutionAgent)
