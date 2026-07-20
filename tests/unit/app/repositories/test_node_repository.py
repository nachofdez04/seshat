from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

from seshat.app.repositories.node_repository import NodeRepository
from seshat.core.models.enums import NodeState, NodeStatus, RelationshipType
from seshat.core.models.nodes import ExtractionResult
from tests.helpers import make_node
from tests.integration.helpers import make_relationship


def _make_repo() -> tuple[NodeRepository, MagicMock, MagicMock]:
    kb_store = MagicMock()
    kb_store.write_node = AsyncMock()
    kb_store.write_relationship = AsyncMock()
    kb_store.update_node = AsyncMock()
    kb_store.update_node_state = AsyncMock()
    kb_store.delete_node = AsyncMock()
    kb_store.delete_relationships_for_node = AsyncMock()
    kb_store.get_outbound_state_transition_targets = AsyncMock(return_value=[])
    kb_store.count_remaining_state_transition_sources = AsyncMock(return_value=0)

    @asynccontextmanager
    async def _fake_transaction():
        yield MagicMock()

    kb_store.transaction = _fake_transaction

    vs = MagicMock()
    vs.upsert = AsyncMock()
    vs.delete = AsyncMock()

    return NodeRepository(kb_store, vs), kb_store, vs


class TestWriteBatch:
    async def test_approved_node_written(self):
        repo, kb, vs = _make_repo()
        node = make_node()
        result = ExtractionResult(job_id="job-1", nodes=[node], relationships=[])
        await repo.write_batch(result)

        kb.write_node.assert_called_once()
        vs.upsert.assert_called_once()

    async def test_rejected_node_not_written(self):
        repo, kb, vs = _make_repo()
        node = make_node(status=NodeStatus.REJECTED)
        result = ExtractionResult(job_id="job-1", nodes=[node], relationships=[])
        await repo.write_batch(result)

        kb.write_node.assert_not_called()
        vs.upsert.assert_not_called()

    async def test_supersedes_triggers_state_transition(self):
        repo, kb, _vs = _make_repo()
        existing_node = make_node("existing")
        new_node = make_node("new")
        rel = make_relationship(new_node, existing_node, rel_type=RelationshipType.SUPERSEDES)

        result = ExtractionResult(job_id="job-1", nodes=[new_node], relationships=[rel])
        await repo.write_batch(result)

        args, kwargs = kb.update_node_state.call_args
        assert args == (str(existing_node.id), NodeState.SUPERSEDED)
        assert "conn" in kwargs

    async def test_relationship_not_written_if_source_rejected(self):
        repo, kb, _vs = _make_repo()
        source = make_node("source", status=NodeStatus.REJECTED)
        target = make_node("target")
        rel = make_relationship(source, target)

        result = ExtractionResult(job_id="job-1", nodes=[source, target], relationships=[rel])
        await repo.write_batch(result)

        kb.write_relationship.assert_not_called()

    async def test_supersedes_missing_target_logs_warning(self):
        repo, kb, _vs = _make_repo()
        new_node = make_node("new")
        existing_node = make_node("existing")
        rel = make_relationship(new_node, existing_node, rel_type=RelationshipType.SUPERSEDES)

        kb.update_node_state = AsyncMock(side_effect=KeyError("not found"))

        result = ExtractionResult(job_id="job-1", nodes=[new_node], relationships=[rel])
        written_nodes, _written_rels = await repo.write_batch(result)
        assert written_nodes == 1


class TestWriteNode:
    async def test_writes_kb_and_vs(self):
        repo, kb, vs = _make_repo()
        node = make_node()
        await repo.write_node(node)

        kb.write_node.assert_called_once()
        vs.upsert.assert_called_once()

    async def test_writes_relationships_in_transaction(self):
        repo, kb, _vs = _make_repo()
        node = make_node()
        other = make_node("other")
        rel = make_relationship(node, other)
        await repo.write_node(node, relationships=[rel])

        kb.write_relationship.assert_called_once()


_NODE_UUID = UUID("00000000-0000-0000-0000-000000000001")
_SOURCE_UUID = UUID("00000000-0000-0000-0000-000000000002")


class TestDeleteNode:
    async def test_deletes_from_kb_and_vs(self):
        repo, kb, vs = _make_repo()
        await repo.delete_node(_NODE_UUID)

        kb.delete_relationships_for_node.assert_called_once()
        kb.delete_node.assert_called_once()
        vs.delete.assert_called_once_with(str(_NODE_UUID))

    async def test_reverts_superseded_target_to_current(self):
        repo, kb, _vs = _make_repo()
        kb.get_outbound_state_transition_targets = AsyncMock(return_value=["target-1"])
        kb.count_remaining_state_transition_sources = AsyncMock(return_value=0)

        await repo.delete_node(_SOURCE_UUID)

        kb.update_node_state.assert_called_once()
        args, _kwargs = kb.update_node_state.call_args
        assert args == ("target-1", NodeState.CURRENT)

    async def test_does_not_revert_if_another_source_remains(self):
        repo, kb, _vs = _make_repo()
        kb.get_outbound_state_transition_targets = AsyncMock(return_value=["target-1"])
        kb.count_remaining_state_transition_sources = AsyncMock(return_value=1)

        await repo.delete_node(_SOURCE_UUID)

        kb.update_node_state.assert_not_called()

    async def test_reverts_only_targets_with_no_remaining_sources(self):
        repo, kb, _vs = _make_repo()
        kb.get_outbound_state_transition_targets = AsyncMock(return_value=["target-1", "target-2"])
        kb.count_remaining_state_transition_sources = AsyncMock(side_effect=[0, 1])

        await repo.delete_node(_SOURCE_UUID)

        assert kb.update_node_state.call_count == 1
        args, _ = kb.update_node_state.call_args
        assert args[0] == "target-1"


class TestUpdateNode:
    async def test_updates_kb_and_vs(self):
        repo, kb, vs = _make_repo()
        node = make_node()
        kb.update_node = AsyncMock()
        await repo.update_node(node)

        kb.update_node.assert_called_once()
        vs.upsert.assert_called_once()

    async def test_replaces_outbound_rels_when_flag_set(self):
        repo, kb, _vs = _make_repo()
        node = make_node()
        other = make_node("other")
        rel = make_relationship(node, other)
        kb.update_node = AsyncMock()

        await repo.update_node(node, relationships=[rel], replace_outbound_rels=True)

        kb.delete_relationships_for_node.assert_called_once()
        kb.write_relationship.assert_called_once()

    async def test_skips_delete_when_flag_not_set(self):
        repo, kb, _vs = _make_repo()
        node = make_node()
        kb.update_node = AsyncMock()

        await repo.update_node(node, replace_outbound_rels=False)

        kb.delete_relationships_for_node.assert_not_called()


class TestUpdateNodeState:
    async def test_delegates_to_kb(self):
        repo, kb, _vs = _make_repo()
        await repo.update_node_state(_NODE_UUID, NodeState.SUPERSEDED)
        kb.update_node_state.assert_called_once_with(str(_NODE_UUID), NodeState.SUPERSEDED)


class TestWriteBatchAmends:
    async def test_amends_triggers_amended_state(self):
        repo, kb, _vs = _make_repo()
        existing_node = make_node("existing")
        new_node = make_node("new")
        rel = make_relationship(new_node, existing_node, rel_type=RelationshipType.AMENDS)

        result = ExtractionResult(job_id="job-1", nodes=[new_node], relationships=[rel])
        await repo.write_batch(result)

        args, _kwargs = kb.update_node_state.call_args
        assert args == (str(existing_node.id), NodeState.AMENDED)

    async def test_relationship_written_when_target_already_in_kb(self):
        """A relationship whose target is NOT in the batch (i.e., already in KB) should be written."""
        repo, kb, _vs = _make_repo()
        new_node = make_node("new")
        existing_kb_node = make_node("existing")
        # existing_kb_node is NOT included in the batch's nodes list — it's already in KB
        rel = make_relationship(new_node, existing_kb_node)

        result = ExtractionResult(job_id="job-1", nodes=[new_node], relationships=[rel])
        await repo.write_batch(result)

        kb.write_relationship.assert_called_once()


class TestDeleteNodeCascade:
    async def test_non_cascade_passes_flag_to_kb(self):
        repo, kb, _vs = _make_repo()
        await repo.delete_node(_NODE_UUID, cascade=False)

        args, kwargs = kb.delete_relationships_for_node.call_args
        assert kwargs.get("cascade") is False or (len(args) > 1 and args[1] is False)

    async def test_cascade_default_is_true(self):
        repo, kb, _vs = _make_repo()
        await repo.delete_node(_NODE_UUID)

        args, kwargs = kb.delete_relationships_for_node.call_args
        passed_cascade = kwargs.get("cascade", args[1] if len(args) > 1 else None)
        assert passed_cascade is True


class TestWriteNodeRelationshipsTransaction:
    async def test_both_source_and_target_approved_writes_relationship(self):
        repo, kb, _vs = _make_repo()
        source = make_node("src")
        target = make_node("tgt")
        rel = make_relationship(source, target)

        result = ExtractionResult(job_id="job-1", nodes=[source, target], relationships=[rel])
        await repo.write_batch(result)

        kb.write_relationship.assert_called_once()

    async def test_relationship_not_written_if_target_rejected(self):
        repo, kb, _vs = _make_repo()
        source = make_node("src")
        target = make_node("tgt", status=NodeStatus.REJECTED)
        rel = make_relationship(source, target)

        result = ExtractionResult(job_id="job-1", nodes=[source, target], relationships=[rel])
        await repo.write_batch(result)

        kb.write_relationship.assert_not_called()
