from __future__ import annotations

from datetime import date
from unittest.mock import AsyncMock, MagicMock

import pytest

from seshat.config.settings import KBStoreConfig
from seshat.knowledge_store.pg_store import PostgresKBStore
from seshat.models.api_graph import ManualNodeCreate, ManualNodeUpdate, NodeOverride, RelationshipInput
from seshat.models.enums import (
    ApprovalMethod,
    ConceptType,
    GraphDirection,
    IngestionSource,
    RelationshipType,
)
from seshat.models.nodes import NodeMetadata
from seshat.worker.manual_ingestion import ManualIngestionService, NodeNotFoundError, NodePreconditionError
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
    return vs


@pytest.fixture
def fake_extraction_orch():
    orch = MagicMock()
    orch.run_resolution = AsyncMock()
    return orch


@pytest.fixture
def svc(kb_store, fake_vector_store, fake_extraction_orch):
    return ManualIngestionService(kb_store, fake_vector_store, fake_extraction_orch)


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

    async def test_vector_store_upserted(self, svc, fake_vector_store):
        node = await svc.create(_create_payload(), user_id="alice")
        fake_vector_store.upsert.assert_called_once_with(
            str(node.id), f"{node.title} {node.description}", node.metadata.model_dump(mode="json")
        )


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
                ingestion_source=IngestionSource.JOB,
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
        # verify the inbound relationship row (other → node) was also removed
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

    async def test_delete_calls_vector_store(self, svc, fake_vector_store):
        node = await svc.create(_create_payload(), user_id="alice")
        await svc.delete(str(node.id), cascade=True)
        fake_vector_store.delete.assert_called_once_with(str(node.id))
