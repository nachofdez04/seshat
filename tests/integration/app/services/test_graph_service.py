from __future__ import annotations

from datetime import UTC, date, datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest

from seshat.app.repositories.node_repository import NodeRepository
from seshat.app.services.graph import GraphService, NodeNotFoundError, NodePreconditionError
from seshat.core.config.settings import KBStoreConfig
from seshat.core.models.api_graph import (
    ManualNodeCreate,
    ManualNodeUpdate,
    NodeFilter,
    NodeOverride,
    RelationshipInput,
    SearchResult,
)
from seshat.core.models.enums import (
    ApprovalMethod,
    ConceptType,
    GraphDirection,
    IngestionSource,
    NodeState,
    NodeStatus,
    RelationshipType,
    SearchMode,
)
from seshat.core.models.nodes import ExtractionResult, KBNode, KBRelationship, NodeMetadata
from seshat.infra.knowledge_store.pg_store import PostgresKBStore
from tests.helpers import make_node
from tests.integration.conftest import SKIP_IF_NO_POSTGRES
from tests.integration.helpers import make_relationship

pytestmark = [pytest.mark.integration, SKIP_IF_NO_POSTGRES]


@pytest.fixture
async def kb_store(pg_test_url):
    config = KBStoreConfig()
    s = PostgresKBStore(config, pg_test_url)
    await s.connect()
    yield s
    await s.pool.execute(f"TRUNCATE {s._schema}.kb_relationships, {s._schema}.kb_nodes CASCADE")
    await s.close()


@pytest.fixture
def fake_vector_store():
    vs = MagicMock()
    vs.upsert = AsyncMock()
    vs.delete = AsyncMock()
    vs.search = AsyncMock(return_value=[])
    vs.update_metadata = AsyncMock()
    return vs


@pytest.fixture
def fake_extraction_orch():
    orch = MagicMock()
    orch.run_resolution = AsyncMock()
    return orch


@pytest.fixture
def svc(kb_store, fake_vector_store, fake_extraction_orch):
    node_repo = NodeRepository(kb_store, fake_vector_store)
    return GraphService(node_repo, fake_extraction_orch)


def _create_payload(
    title: str = "T",
    description: str = "D",
    meeting_date: date = date(2026, 1, 15),
    relationships=None,
) -> ManualNodeCreate:
    return ManualNodeCreate(
        type=ConceptType.DECISION,
        title=title,
        description=description,
        meeting_date=meeting_date,
        relationships=relationships,
    )


def _update_payload(
    title: str = "Updated",
    description: str = "Updated desc",
    reason: str | None = None,
    relationships=None,
) -> ManualNodeUpdate:
    return ManualNodeUpdate(title=title, description=description, reason=reason, relationships=relationships)


def _override_payload(
    title: str = "Override",
    description: str = "Override desc",
    reason: str = "Correction reason",
    relationships=None,
) -> NodeOverride:
    return NodeOverride(title=title, description=description, reason=reason, relationships=relationships)


class TestCreateIntegration:
    async def test_node_persisted_to_db(self, svc, kb_store):
        node = await svc.create(_create_payload(), user_id="alice")
        fetched = await kb_store.get_node(str(node.id))

        assert fetched is not None
        assert fetched.title == "T"
        assert fetched.metadata.ingestion_source == IngestionSource.MANUAL
        assert fetched.metadata.approval_method == ApprovalMethod.MANUAL
        assert fetched.metadata.job_id.startswith("manual_")

    async def test_relationships_persisted(self, svc, kb_store):
        target = make_node("tgt-create")
        await kb_store.write_node(target)

        rel = RelationshipInput(target_id=str(target.id), rel_type=RelationshipType.SUPERSEDES)
        node = await svc.create(_create_payload(relationships=[rel]), user_id="alice")

        neighbours = await kb_store.get_neighbours(str(node.id), direction=GraphDirection.OUTBOUND)
        assert any(n.id == target.id for n in neighbours)


class TestUpdateIntegration:
    async def test_updates_title_and_description(self, svc, kb_store):
        node = await svc.create(_create_payload(), user_id="alice")
        await svc.update(str(node.id), _update_payload(title="New Title"), user_id="bob")

        fetched = await kb_store.get_node(str(node.id))
        assert fetched is not None
        assert fetched.title == "New Title"
        assert fetched.metadata.corrected_by == "bob"

    async def test_replaces_relationships(self, svc, kb_store):
        tgt1 = make_node("tgt-upd-1")
        tgt2 = make_node("tgt-upd-2", "Second target")
        await kb_store.write_node(tgt1)
        await kb_store.write_node(tgt2)

        node = await svc.create(
            _create_payload(
                relationships=[RelationshipInput(target_id=str(tgt1.id), rel_type=RelationshipType.SUPERSEDES)]
            ),
            user_id="alice",
        )
        await svc.update(
            str(node.id),
            _update_payload(
                relationships=[RelationshipInput(target_id=str(tgt2.id), rel_type=RelationshipType.AMENDS)]
            ),
            user_id="alice",
        )

        neighbours = await kb_store.get_neighbours(str(node.id), direction=GraphDirection.OUTBOUND)
        neighbour_ids = {n.id for n in neighbours}
        assert tgt1.id not in neighbour_ids
        assert tgt2.id in neighbour_ids

    async def test_raises_not_found(self, svc):
        with pytest.raises(NodeNotFoundError):
            await svc.update("00000000-0000-0000-0000-000000000000", _update_payload(), user_id="alice")

    async def test_raises_precondition_for_pipeline_node(self, svc, kb_store):
        node = make_node("pipeline-node")
        await kb_store.write_node(node)
        with pytest.raises(NodePreconditionError):
            await svc.update(str(node.id), _update_payload(), user_id="alice")


class TestOverrideIntegration:
    async def test_stores_correction_reason(self, svc, kb_store):
        node = await svc.create(_create_payload(), user_id="alice")
        await svc.override(
            str(node.id), _override_payload(reason="Wrong decision"), user_id="admin", minimum_method=None
        )

        fetched = await kb_store.get_node(str(node.id))
        assert fetched is not None
        assert fetched.metadata.correction_reason == "Wrong decision"
        assert fetched.metadata.corrected_by == "admin"

    async def test_operator_can_override_auto_approved(self, svc, kb_store):
        node = make_node(
            "auto-node",
            metadata=NodeMetadata(
                job_id="job-1",
                ingestion_source=IngestionSource.PIPELINE,
                approval_method=ApprovalMethod.AUTO,
            ),
        )
        await kb_store.write_node(node)
        await svc.override(str(node.id), _override_payload(), user_id="operator", minimum_method=ApprovalMethod.AUTO)

        fetched = await kb_store.get_node(str(node.id))
        assert fetched is not None
        assert fetched.title == "Override"

    async def test_raises_precondition_when_method_mismatch(self, svc, kb_store):
        node = make_node("individual-node")
        await kb_store.write_node(node)
        with pytest.raises(NodePreconditionError):
            await svc.override(
                str(node.id), _override_payload(), user_id="operator", minimum_method=ApprovalMethod.AUTO
            )


class TestDeleteIntegration:
    async def test_cascade_removes_node_and_both_relationship_directions(self, svc, kb_store):
        node = await svc.create(_create_payload(), user_id="alice")
        other = make_node("delete-other")
        await kb_store.write_node(other)
        await kb_store.write_relationship(make_relationship(other, node))

        await svc.delete(str(node.id), cascade=True)

        assert await kb_store.get_node(str(node.id)) is None
        assert await kb_store.get_neighbours(str(other.id), direction=GraphDirection.OUTBOUND) == []

    async def test_safe_delete_succeeds_when_no_inbound(self, svc, kb_store):
        node = await svc.create(_create_payload(), user_id="alice")
        await svc.delete(str(node.id), cascade=False)
        assert await kb_store.get_node(str(node.id)) is None

    async def test_safe_delete_raises_when_inbound_exist(self, svc, kb_store):
        node = await svc.create(_create_payload(), user_id="alice")
        other = make_node("safe-delete-other")
        await kb_store.write_node(other)
        await kb_store.write_relationship(make_relationship(other, node))

        with pytest.raises(NodePreconditionError, match="referenced as a target"):
            await svc.delete(str(node.id), cascade=False)

        assert await kb_store.get_node(str(node.id)) is not None

    async def test_delete_superseding_node_reverts_target_to_current(self, svc, kb_store):
        """Deleting the only superseding node must revert the target back to CURRENT."""
        target = make_node("revert-target")
        await kb_store.write_node(target)

        superseder = await svc.create(
            _create_payload(
                relationships=[RelationshipInput(target_id=str(target.id), rel_type=RelationshipType.SUPERSEDES)]
            ),
            user_id="alice",
        )

        # Manually set target to SUPERSEDED (as the pipeline would)
        await kb_store.update_node_state(str(target.id), NodeState.SUPERSEDED)
        fetched_before = await kb_store.get_node(str(target.id))
        assert fetched_before is not None
        assert fetched_before.state == NodeState.SUPERSEDED

        await svc.delete(str(superseder.id), cascade=True)

        fetched_after = await kb_store.get_node(str(target.id))
        assert fetched_after is not None
        assert fetched_after.state == NodeState.CURRENT


class TestGraphServiceSearch:
    @pytest.fixture
    def fake_vector_store(self):
        store: dict[str, str] = {}
        vs = MagicMock()
        vs.upsert = AsyncMock(side_effect=lambda nid, text, _meta: store.update({nid: text}))
        vs.delete = AsyncMock()

        async def _search(query, **kwargs):
            q = query.lower()
            return [SearchResult(node_id=UUID(nid), score=1.0) for nid, text in store.items() if q in text.lower()]

        vs.search = _search
        vs.update_metadata = AsyncMock()
        return vs

    async def test_keyword_search_returns_matching_node(self, svc, kb_store):
        node_a = await svc.create(
            _create_payload(title="Zymurgy brewing process", description="We use zymurgy"),
            user_id="alice",
        )
        await svc.create(
            _create_payload(title="Unrelated caching decision", description="Use Redis"),
            user_id="alice",
        )

        results = await svc.search(
            query="zymurgy",
            limit=5,
            node_filter=NodeFilter(),
            mode=SearchMode.KEYWORD,
        )

        result_ids = [r.detail.node.id for r in results]
        assert node_a.id in result_ids
        assert all(r.score is not None for r in results)

    async def test_keyword_search_no_match_returns_empty(self, svc):
        await svc.create(
            _create_payload(title="PostgreSQL decision", description="Use PostgreSQL"),
            user_id="alice",
        )

        results = await svc.search(
            query="xylophone quasar",
            limit=5,
            node_filter=NodeFilter(),
            mode=SearchMode.KEYWORD,
        )

        assert results == []


class TestGraphServiceTraverseImpact:
    async def test_supersedes_chain_depth_two(self, svc, kb_store):
        """A→SUPERSEDES→B→SUPERSEDES→C; traverse_impact(A, depth=2) must return B and C."""
        node_a = await svc.create(_create_payload(title="A"), user_id="alice")
        node_b = await svc.create(
            _create_payload(
                title="B",
                relationships=[RelationshipInput(target_id=str(node_a.id), rel_type=RelationshipType.SUPERSEDES)],
            ),
            user_id="alice",
        )
        node_c = await svc.create(
            _create_payload(
                title="C",
                relationships=[RelationshipInput(target_id=str(node_b.id), rel_type=RelationshipType.SUPERSEDES)],
            ),
            user_id="alice",
        )

        await kb_store.update_node_state(str(node_a.id), NodeState.SUPERSEDED)
        await kb_store.update_node_state(str(node_b.id), NodeState.SUPERSEDED)

        impact = await svc.traverse_impact(
            node_id=node_c.id,
            depth=2,
            rel_types=None,
            min_confidence=0.0,
            direction=GraphDirection.OUTBOUND,
        )

        impact_node_ids = {n.node.id for n in impact.nodes}
        assert node_a.id in impact_node_ids
        assert node_b.id in impact_node_ids

        depth_by_id = {n.node.id: n.traversal_depth for n in impact.nodes}
        assert depth_by_id[node_b.id] == 1
        assert depth_by_id[node_a.id] == 2

    async def test_depth_one_does_not_include_two_hop_node(self, svc, kb_store):
        """traverse_impact with depth=1 must not include nodes two hops away."""
        node_a = await svc.create(_create_payload(title="A-d1"), user_id="alice")
        node_b = await svc.create(
            _create_payload(
                title="B-d1",
                relationships=[RelationshipInput(target_id=str(node_a.id), rel_type=RelationshipType.SUPERSEDES)],
            ),
            user_id="alice",
        )
        node_c = await svc.create(
            _create_payload(
                title="C-d1",
                relationships=[RelationshipInput(target_id=str(node_b.id), rel_type=RelationshipType.SUPERSEDES)],
            ),
            user_id="alice",
        )

        await kb_store.update_node_state(str(node_a.id), NodeState.SUPERSEDED)
        await kb_store.update_node_state(str(node_b.id), NodeState.SUPERSEDED)

        impact = await svc.traverse_impact(
            node_id=node_c.id,
            depth=1,
            rel_types=None,
            min_confidence=0.0,
            direction=GraphDirection.OUTBOUND,
        )

        impact_node_ids = {n.node.id for n in impact.nodes}
        assert node_b.id in impact_node_ids
        assert node_a.id not in impact_node_ids


class TestWriteBatchStateTransitions:
    async def test_supersedes_relationship_sets_target_state(self, kb_store, fake_vector_store):
        node_repo = NodeRepository(kb_store, fake_vector_store)

        existing = KBNode(
            id=uuid4(),
            type=ConceptType.DECISION,
            title="Use MySQL for the database",
            description="Prior decision to use MySQL.",
            confidence=0.9,
            status=NodeStatus.APPROVED,
            metadata=NodeMetadata(
                job_id="old-job",
                meeting_date=date(2026, 1, 1),
                ingestion_source=IngestionSource.PIPELINE,
            ),
        )
        await node_repo.write_node(existing)

        new_node = KBNode(
            id=uuid4(),
            type=ConceptType.DECISION,
            title="Use PostgreSQL for the database",
            description="New decision superseding the old one.",
            confidence=0.9,
            status=NodeStatus.APPROVED,
            metadata=NodeMetadata(
                job_id="new-job",
                meeting_date=date(2026, 1, 15),
                ingestion_source=IngestionSource.PIPELINE,
            ),
        )
        rel = KBRelationship(
            source_id=new_node.id,
            target_id=existing.id,
            rel_type=RelationshipType.SUPERSEDES,
            job_id="new-job",
            created_at=datetime.now(UTC),
        )
        result = ExtractionResult(job_id="new-job", nodes=[new_node], relationships=[rel])
        await node_repo.write_batch(result)

        updated = await kb_store.get_node(str(existing.id))
        assert updated is not None
        assert updated.state == NodeState.SUPERSEDED
