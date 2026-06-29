from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

import pytest

from seshat.models.api_graph import (
    BulkNodeCreate,
    BulkNodeDelete,
    ManualNodeCreate,
    ManualNodeUpdate,
    NodeOverride,
    RelationshipInput,
)
from seshat.models.enums import ApprovalMethod, ConceptType, IngestionSource, NodeState, NodeStatus, RelationshipType
from seshat.models.nodes import KBNode, NodeMetadata
from seshat.worker.manual_ingestion import ManualIngestionService, NodeNotFoundError, NodePreconditionError
from tests.helpers import make_node


def _make_service(*, node: KBNode | None = None, inbound_count: int = 0):
    @asynccontextmanager
    async def _tx():
        conn = AsyncMock()
        yield conn

    kb = MagicMock()
    kb.transaction = _tx
    kb.get_node = AsyncMock(return_value=node)
    kb.write_node = AsyncMock()
    kb.write_relationship = AsyncMock()
    kb.update_node = AsyncMock()
    kb.delete_node = AsyncMock()
    kb.delete_relationships_for_node = AsyncMock()
    kb.count_inbound_relationships = AsyncMock(return_value=inbound_count)

    vs = MagicMock()
    vs.upsert = AsyncMock()
    vs.delete = AsyncMock()

    extraction_orch = MagicMock()

    return ManualIngestionService(kb, vs, extraction_orch), kb, vs


def _manual_metadata() -> NodeMetadata:
    return NodeMetadata(
        job_id="manual_abc",
        ingestion_source=IngestionSource.MANUAL,
        approval_method=ApprovalMethod.MANUAL,
    )


def _auto_metadata() -> NodeMetadata:
    return NodeMetadata(job_id="job-1", ingestion_source=IngestionSource.JOB, approval_method=ApprovalMethod.AUTO)


def _create_payload(
    title: str = "T",
    description: str = "D",
    relationships=None,
    source_quote: str | None = None,
    blob_key: str | None = None,
) -> ManualNodeCreate:
    return ManualNodeCreate(
        type=ConceptType.DECISION,
        title=title,
        description=description,
        relationships=relationships,
        source_quote=source_quote,
        blob_key=blob_key,
    )


def _update_payload(
    title: str = "T2",
    description: str = "D2",
    reason: str | None = None,
    relationships=None,
) -> ManualNodeUpdate:
    return ManualNodeUpdate(title=title, description=description, reason=reason, relationships=relationships)


def _override_payload(
    title: str = "T3",
    description: str = "D3",
    reason: str = "Correction",
    relationships=None,
) -> NodeOverride:
    return NodeOverride(title=title, description=description, reason=reason, relationships=relationships)


class TestCreate:
    async def test_returns_kb_node(self):
        svc, _, _ = _make_service()
        node = await svc.create(_create_payload(), user_id="alice")

        assert isinstance(node, KBNode)
        assert node.status == NodeStatus.APPROVED
        assert node.state == NodeState.CURRENT
        assert node.metadata.ingestion_source == IngestionSource.MANUAL
        assert node.metadata.approval_method == ApprovalMethod.MANUAL
        assert node.metadata.approved_by == "alice"

    async def test_job_id_has_manual_prefix(self):
        svc, _, _ = _make_service()
        node = await svc.create(_create_payload(), user_id="alice")
        assert node.metadata.job_id.startswith("manual_")

    async def test_job_id_is_unique_across_creates(self):
        svc, _, _ = _make_service()
        n1 = await svc.create(_create_payload(), user_id="alice")
        n2 = await svc.create(_create_payload(), user_id="alice")
        assert n1.metadata.job_id != n2.metadata.job_id

    async def test_writes_to_kb_and_vector_store(self):
        svc, kb, vs = _make_service()
        node = await svc.create(_create_payload(), user_id="alice")

        kb.write_node.assert_called_once()
        vs.upsert.assert_called_once_with(
            str(node.id), f"{node.title} {node.description}", node.metadata.model_dump(mode="json")
        )

    async def test_writes_relationships_when_provided(self):
        target = make_node("tgt")
        svc, kb, _ = _make_service()
        rel = RelationshipInput(target_id=str(target.id), rel_type=RelationshipType.SUPERSEDES)
        await svc.create(_create_payload(relationships=[rel]), user_id="alice")

        kb.write_relationship.assert_called_once()
        written = kb.write_relationship.call_args.args[0]
        assert written.rel_type == RelationshipType.SUPERSEDES
        assert written.target_id == target.id

    async def test_skips_relationships_when_none(self):
        svc, kb, _ = _make_service()
        await svc.create(_create_payload(relationships=None), user_id="alice")
        kb.write_relationship.assert_not_called()

    async def test_logs_warning_for_source_quote(self, caplog):
        svc, _, _ = _make_service()
        payload = _create_payload(source_quote="some quote", blob_key="blobs/key")
        with caplog.at_level(logging.WARNING, logger="seshat.worker.manual_ingestion"):
            await svc.create(payload, user_id="alice")
        assert "not yet implemented" in caplog.text


class TestUpdate:
    async def test_raises_not_found_when_node_missing(self):
        svc, _, _ = _make_service(node=None)
        with pytest.raises(NodeNotFoundError):
            await svc.update("missing-id", _update_payload(), user_id="alice")

    async def test_raises_precondition_for_non_manual_node(self):
        node = make_node()  # ingestion_source=JOB by default
        svc, _, _ = _make_service(node=node)
        with pytest.raises(NodePreconditionError, match="manually-created"):
            await svc.update(str(node.id), _update_payload(), user_id="alice")

    async def test_updates_manual_node(self):
        node = make_node(metadata=_manual_metadata())
        svc, kb, vs = _make_service(node=node)
        result = await svc.update(str(node.id), _update_payload(title="New Title"), user_id="alice")

        assert result.title == "New Title"
        assert result.metadata.corrected_by == "alice"
        kb.update_node.assert_called_once()
        vs.upsert.assert_called_once()

    async def test_reason_is_none_for_plain_update(self):
        node = make_node(metadata=_manual_metadata())
        svc, _, _ = _make_service(node=node)
        result = await svc.update(str(node.id), _update_payload(), user_id="alice")
        assert result.metadata.correction_reason is None

    async def test_replaces_relationships_when_provided(self):
        target = make_node("tgt")
        node = make_node(metadata=_manual_metadata())
        svc, kb, _ = _make_service(node=node)
        rel = RelationshipInput(target_id=str(target.id), rel_type=RelationshipType.SUPERSEDES)
        await svc.update(str(node.id), _update_payload(relationships=[rel]), user_id="alice")

        kb.delete_relationships_for_node.assert_called_once()
        delete_call = kb.delete_relationships_for_node.call_args
        assert delete_call.args[0] == str(node.id)
        assert delete_call.kwargs.get("cascade") is False

        kb.write_relationship.assert_called_once()
        written = kb.write_relationship.call_args.args[0]
        assert written.rel_type == RelationshipType.SUPERSEDES
        assert written.target_id == target.id

    async def test_preserves_relationships_when_none(self):
        node = make_node(metadata=_manual_metadata())
        svc, kb, _ = _make_service(node=node)
        await svc.update(str(node.id), _update_payload(relationships=None), user_id="alice")

        kb.delete_relationships_for_node.assert_not_called()
        kb.write_relationship.assert_not_called()


class TestOverride:
    async def test_raises_not_found_when_node_missing(self):
        svc, _, _ = _make_service(node=None)
        with pytest.raises(NodeNotFoundError):
            await svc.override("missing-id", _override_payload(), user_id="alice", minimum_method=None)

    async def test_raises_precondition_when_method_does_not_match(self):
        node = make_node()  # approval_method=INDIVIDUAL by default
        svc, _, _ = _make_service(node=node)
        with pytest.raises(NodePreconditionError, match="Insufficient role"):
            await svc.override(str(node.id), _override_payload(), user_id="alice", minimum_method=ApprovalMethod.AUTO)

    async def test_allows_when_method_matches_exactly(self):
        node = make_node(metadata=_manual_metadata())  # approval_method=MANUAL
        svc, kb, _ = _make_service(node=node)
        await svc.override(str(node.id), _override_payload(), user_id="alice", minimum_method=ApprovalMethod.MANUAL)
        kb.update_node.assert_called_once()

    async def test_allows_any_node_when_minimum_method_is_none(self):
        node = make_node()  # INDIVIDUAL approval — would be blocked by a method check
        svc, kb, _ = _make_service(node=node)
        result = await svc.override(
            str(node.id),
            _override_payload(reason="Admin fix"),
            user_id="admin",
            minimum_method=None,
        )

        assert result.metadata.correction_reason == "Admin fix"
        kb.update_node.assert_called_once()

    async def test_stores_reason_and_corrected_by_in_metadata(self):
        node = make_node(metadata=_auto_metadata())
        svc, _, _ = _make_service(node=node)
        result = await svc.override(
            str(node.id),
            _override_payload(reason="Wrong decision"),
            user_id="alice",
            minimum_method=ApprovalMethod.AUTO,
        )

        assert result.metadata.correction_reason == "Wrong decision"
        assert result.metadata.corrected_by == "alice"


class TestDelete:
    async def test_cascade_deletes_all_relationships_and_node(self):
        svc, kb, vs = _make_service()
        await svc.delete("node-id", cascade=True)

        kb.delete_relationships_for_node.assert_called_once()
        delete_call = kb.delete_relationships_for_node.call_args
        assert delete_call.kwargs.get("cascade") is True
        kb.delete_node.assert_called_once()
        vs.delete.assert_called_once_with("node-id")

    async def test_safe_delete_passes_when_no_inbound(self):
        svc, kb, vs = _make_service(inbound_count=0)
        await svc.delete("node-id", cascade=False)

        kb.delete_node.assert_called_once()
        vs.delete.assert_called_once_with("node-id")

    async def test_safe_delete_raises_when_inbound_relationships_exist(self):
        svc, kb, _ = _make_service(inbound_count=3)
        with pytest.raises(NodePreconditionError, match="referenced as a target by 3"):
            await svc.delete("node-id", cascade=False)

        kb.delete_node.assert_not_called()


class TestBulkCreate:
    async def test_returns_succeeded_ids(self):
        svc, _, _ = _make_service()
        payload = BulkNodeCreate(nodes=[_create_payload(), _create_payload(title="T2")])
        result = await svc.bulk_create(payload, user_id="alice")

        assert len(result.succeeded) == 2
        assert result.failed == []

    async def test_stop_on_error_propagates_exception(self):
        svc, kb, _ = _make_service()
        kb.write_node = AsyncMock(side_effect=RuntimeError("db down"))
        payload = BulkNodeCreate(nodes=[_create_payload()], on_error="stop")

        with pytest.raises(RuntimeError, match="db down"):
            await svc.bulk_create(payload, user_id="alice")

    async def test_continue_on_error_collects_failures(self):
        svc, kb, _ = _make_service()
        kb.write_node = AsyncMock(side_effect=RuntimeError("db down"))
        payload = BulkNodeCreate(nodes=[_create_payload(), _create_payload(title="T2")], on_error="continue")
        result = await svc.bulk_create(payload, user_id="alice")

        assert result.succeeded == []
        assert len(result.failed) == 2
        assert "db down" in result.failed[0].error


class TestBulkDelete:
    async def test_returns_succeeded_ids(self):
        svc, _, _ = _make_service()
        payload = BulkNodeDelete(node_ids=["id-1", "id-2"])
        result = await svc.bulk_delete(payload)

        assert result.succeeded == ["id-1", "id-2"]
        assert result.failed == []

    async def test_stop_on_error_propagates_exception(self):
        svc, _, _ = _make_service(inbound_count=1)
        payload = BulkNodeDelete(node_ids=["id-1"], on_error="stop")

        with pytest.raises(NodePreconditionError):
            await svc.bulk_delete(payload, cascade=False)

    async def test_continue_on_error_collects_failures(self):
        svc, _, _ = _make_service(inbound_count=1)
        payload = BulkNodeDelete(node_ids=["id-1", "id-2"], on_error="continue")
        result = await svc.bulk_delete(payload, cascade=False)

        assert result.succeeded == []
        assert len(result.failed) == 2
        assert result.failed[0].node_id == "id-1"
