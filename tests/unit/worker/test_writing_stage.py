from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

from seshat.models.enums import NodeState, NodeStatus, RelationshipType
from seshat.models.nodes import ExtractionResult
from seshat.worker.writing_stage import WritingStage
from tests.helpers import make_node
from tests.integration.helpers import make_relationship


def _make_kb_store() -> MagicMock:
    kb_store = MagicMock()
    kb_store.write_node = AsyncMock()
    kb_store.write_relationship = AsyncMock()
    kb_store.update_node_state = AsyncMock()

    @asynccontextmanager
    async def _fake_transaction():
        yield MagicMock()

    kb_store.transaction = _fake_transaction
    return kb_store


class TestWritingStage:
    async def test_approved_node_written(self):
        kb_store = _make_kb_store()
        vs = MagicMock()
        vs.upsert = AsyncMock()

        node = make_node()
        result = ExtractionResult(job_id="job-1", nodes=[node], relationships=[])
        await WritingStage(kb_store, vs).write(result)

        kb_store.write_node.assert_called_once()
        vs.upsert.assert_called_once()

    async def test_rejected_node_not_written(self):
        kb_store = _make_kb_store()
        vs = MagicMock()
        vs.upsert = AsyncMock()

        node = make_node(status=NodeStatus.REJECTED)
        result = ExtractionResult(job_id="job-1", nodes=[node], relationships=[])
        await WritingStage(kb_store, vs).write(result)

        kb_store.write_node.assert_not_called()
        vs.upsert.assert_not_called()

    async def test_supersedes_triggers_state_transition(self):
        existing_node = make_node("existing")
        new_node = make_node("new")
        rel = make_relationship(new_node, existing_node, rel_type=RelationshipType.SUPERSEDES)

        kb_store = _make_kb_store()
        vs = MagicMock()
        vs.upsert = AsyncMock()

        result = ExtractionResult(job_id="job-1", nodes=[new_node], relationships=[rel])
        await WritingStage(kb_store, vs).write(result)

        args, kwargs = kb_store.update_node_state.call_args
        assert args == (str(existing_node.id), NodeState.SUPERSEDED)
        assert "conn" in kwargs

    async def test_relationship_not_written_if_source_rejected(self):
        source = make_node("source", status=NodeStatus.REJECTED)
        target = make_node("target")
        rel = make_relationship(source, target)

        kb_store = _make_kb_store()
        vs = MagicMock()
        vs.upsert = AsyncMock()

        result = ExtractionResult(job_id="job-1", nodes=[source, target], relationships=[rel])
        await WritingStage(kb_store, vs).write(result)

        kb_store.write_relationship.assert_not_called()

    async def test_supersedes_missing_target_logs_warning(self):
        new_node = make_node("new")
        existing_node = make_node("existing")
        rel = make_relationship(new_node, existing_node, rel_type=RelationshipType.SUPERSEDES)

        kb_store = _make_kb_store()
        kb_store.update_node_state = AsyncMock(side_effect=KeyError("not found"))
        vs = MagicMock()
        vs.upsert = AsyncMock()

        result = ExtractionResult(job_id="job-1", nodes=[new_node], relationships=[rel])
        # should not raise
        written = await WritingStage(kb_store, vs).write(result)
        assert written == 1
