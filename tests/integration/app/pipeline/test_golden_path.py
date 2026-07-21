from __future__ import annotations

from datetime import UTC, date, datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
import yaml

from seshat.app.agents.identification.registry import IdentificationRegistry
from seshat.app.agents.resolution.registry import ResolutionRegistry
from seshat.app.pipeline.extraction.node_retriever import NodeRetriever
from seshat.app.pipeline.extraction.orchestrator import ExtractionOrchestrator
from seshat.app.pipeline.extraction.search_engine import SearchEngine
from seshat.app.pipeline.ingestion.orchestrator import IngestionOrchestrator
from seshat.app.repositories.node_repository import NodeRepository
from seshat.core.config.settings import ExtractionConfig, RAGConfig, TranscriptionConfig
from seshat.core.models.api_graph import NodeFilter
from seshat.core.models.enums import ConceptType, IngestionSource, NodeState, NodeStatus, RelationshipType
from seshat.core.models.nodes import ExtractionResult, KBNode, KBRelationship, NodeMetadata
from tests.integration.conftest import (
    SKIP_IF_NO_LLM_API,
    SKIP_IF_NO_LOCALSTACK,
    SKIP_IF_NO_POSTGRES,
)
from tests.integration.helpers import (
    cheap_identification_config,
    cheap_resolution_config,
    make_cheap_llm,
)

pytestmark = [
    # module loop required: kb_store fixture is module-scoped and asyncpg pools are loop-bound
    pytest.mark.asyncio(loop_scope="module"),
    pytest.mark.usefixtures("_truncate_kb_tables"),
    pytest.mark.integration,
    pytest.mark.llm,
    pytest.mark.agents,
    SKIP_IF_NO_POSTGRES,
    SKIP_IF_NO_LOCALSTACK,
    SKIP_IF_NO_LLM_API,
]

_MEETING_DATE = date(2026, 1, 15)
_TRANSCRIPT_YAML = yaml.dump(
    {
        "date": _MEETING_DATE.isoformat(),
        "content": (
            "We reviewed the infrastructure options. "
            "We decided to use PostgreSQL for the primary datastore — it has the best support for our workload. "
            "There is a risk that the migration could cause downtime if the schema changes are not backward-compatible."
        ),
    }
).encode()


@pytest.fixture
def fake_vector_store():
    vs = MagicMock()
    vs.upsert = AsyncMock()
    vs.delete = AsyncMock()
    vs.update_metadata = AsyncMock()
    vs.search_dense = AsyncMock(return_value=[])
    vs.search_sparse = AsyncMock(return_value=[])
    return vs


@pytest.fixture
def extraction_orch(kb_store, fake_vector_store, blob_store):
    llm = make_cheap_llm()
    id_config = cheap_identification_config()
    res_config = cheap_resolution_config()
    extraction_config = ExtractionConfig(identification=id_config, resolution=res_config)
    node_repo = NodeRepository(kb_store, fake_vector_store)
    rag_config = RAGConfig()
    search_engine = SearchEngine(
        rag_config=rag_config, vector_store=fake_vector_store, keyword_llm=None, multi_query_llm=None
    )
    retriever = NodeRetriever(rag_config=rag_config, node_repo=node_repo, search_engine=search_engine)
    return ExtractionOrchestrator(
        config=extraction_config,
        identification_registry=IdentificationRegistry(llm, extraction_config),
        resolution_registry=ResolutionRegistry(llm, extraction_config),
        node_retriever=retriever,
        node_repo=node_repo,
        blob_repo=blob_store,
    )


@pytest.fixture
def ingestion_orch(blob_store):
    transcriber = MagicMock()
    return IngestionOrchestrator(transcriber, blob_store, TranscriptionConfig())


class TestGoldenPath:
    async def test_text_ingest_and_extract_produces_current_nodes(
        self, ingestion_orch, extraction_orch, kb_store, fake_vector_store
    ):
        job_id = "golden-path-job-1"

        doc = await ingestion_orch.ingest_text(_TRANSCRIPT_YAML, _MEETING_DATE, job_id, "meeting.yaml")

        ident_result = await extraction_orch.run_identification(doc, job_id)

        approved = [n._with(status=NodeStatus.APPROVED) for n in ident_result.nodes]
        res_result = await extraction_orch.run_resolution(job_id, approved=approved)

        node_repo = NodeRepository(kb_store, fake_vector_store)
        extraction_result = ExtractionResult(
            job_id=job_id,
            nodes=approved,
            relationships=res_result.relationships,
            confidence_breakdowns={str(k): v for k, v in ident_result.confidence_breakdowns.items()},
        )
        _nodes_written, _rels_written = await node_repo.write_batch(extraction_result)

        current_nodes = await kb_store.query(NodeFilter(state=NodeState.CURRENT))
        assert len(current_nodes) >= 1
        assert all(n.status == NodeStatus.APPROVED for n in current_nodes)

        assert fake_vector_store.upsert.call_count >= 1

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
                meeting_date=_MEETING_DATE,
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
        extraction_result = ExtractionResult(
            job_id="new-job",
            nodes=[new_node],
            relationships=[rel],
        )
        await node_repo.write_batch(extraction_result)

        updated = await kb_store.get_node(str(existing.id))
        assert updated is not None
        assert updated.state == NodeState.SUPERSEDED
