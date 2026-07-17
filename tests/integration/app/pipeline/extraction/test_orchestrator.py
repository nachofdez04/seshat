from datetime import date
from uuid import uuid4

import pytest

from seshat.app.agents.grounding import GroundingAgent
from seshat.app.agents.identification.registry import IdentificationAgentRegistry
from seshat.app.agents.resolution.registry import ResolutionRegistry
from seshat.app.pipeline.extraction.node_retriever import NodeRetriever
from seshat.app.pipeline.extraction.orchestrator import ExtractionOrchestrator
from seshat.app.pipeline.extraction.search_engine import SearchEngine
from seshat.app.repositories.blob_repository import BlobRepository
from seshat.app.repositories.node_repository import NodeRepository
from seshat.core.config.settings import ExtractionConfig, RAGConfig
from seshat.core.models.enums import ConceptType, NodeStatus
from seshat.core.models.transcript import TranscriptDocument, TranscriptMetadata
from tests.helpers import make_node
from tests.integration.conftest import (
    SKIP_IF_NO_EMBEDDINGS_API,
    SKIP_IF_NO_LOCALSTACK,
    SKIP_IF_NO_POSTGRES,
)
from tests.integration.helpers import (
    cheap_grounding_config,
    cheap_identification_config,
    make_cheap_llm,
    seed_node,
)

pytestmark = [
    # module loop required: kb_store fixture is module-scoped and asyncpg pools are loop-bound
    pytest.mark.asyncio(loop_scope="module"),
    pytest.mark.integration,
    pytest.mark.llm,
    pytest.mark.agents,
    pytest.mark.embedding,
    SKIP_IF_NO_POSTGRES,
    SKIP_IF_NO_EMBEDDINGS_API,
    SKIP_IF_NO_LOCALSTACK,
]

_TRANSCRIPT = """
We've reviewed the options. We're going to use PostgreSQL for the user database — it has the best JSON support and the
team is familiar with it.

Agreed. One risk though: we haven't done a migration rehearsal yet, so there's a chance of data corruption if the
script fails.

Fair point. Let's make sure Sergio writes the migration script by Friday.

What cloud provider are we actually deploying to? That's still not decided.
"""

_MEETING_DATE = date(2026, 1, 15)


async def _upload_transcript(blob_repo: BlobRepository, job_id: str, content: str) -> TranscriptDocument:
    """Write transcript to the key the orchestrator expects, return the matching TranscriptDocument."""
    await blob_repo.put_raw_transcript(_MEETING_DATE, job_id, content.encode())
    blob_key = BlobRepository.raw_transcript_key(_MEETING_DATE, job_id)
    return TranscriptDocument(
        source_type="text",
        blob_key=blob_key,
        metadata=TranscriptMetadata(meeting_date=_MEETING_DATE),
    )


@pytest.fixture
def extraction_config():
    llm_cfg = cheap_identification_config()
    return ExtractionConfig(identification=llm_cfg)


def _build_orchestrator(kb_store, vector_store, blob_store, extraction_config, *, grounding_agent=None):
    llm = make_cheap_llm()
    node_repo = NodeRepository(kb_store, vector_store)
    rag_config = RAGConfig()
    search_engine = SearchEngine(
        rag_config=rag_config, vector_store=vector_store, keyword_llm=None, multi_query_llm=None
    )
    rag = NodeRetriever(rag_config=rag_config, node_repo=node_repo, search_engine=search_engine)
    return ExtractionOrchestrator(
        config=extraction_config,
        identification_registry=IdentificationAgentRegistry(llm, extraction_config),
        resolution_registry=ResolutionRegistry(llm, extraction_config),
        node_retriever=rag,
        node_repo=node_repo,
        blob_repo=blob_store,
        grounding_agent=grounding_agent,
    )


@pytest.fixture
def orchestrator(kb_store, vector_store, blob_store, extraction_config):
    return _build_orchestrator(kb_store, vector_store, blob_store, extraction_config)


@pytest.fixture
def orchestrator_with_grounding(kb_store, vector_store, blob_store):
    llm_cfg = cheap_identification_config()
    grounding_llm_cfg = cheap_grounding_config()
    _extraction_config = ExtractionConfig(identification=llm_cfg, grounding=grounding_llm_cfg)
    grounder = GroundingAgent(llm=make_cheap_llm(), config=grounding_llm_cfg)
    return _build_orchestrator(kb_store, vector_store, blob_store, _extraction_config, grounding_agent=grounder)


class TestExtractionOrchestrator:
    async def test_run_identification_returns_nodes_with_confidence(self, orchestrator, blob_store):
        job_id = str(uuid4())
        doc = await _upload_transcript(blob_store, job_id, _TRANSCRIPT)

        result = await orchestrator.run_identification(doc, job_id)

        assert result.job_id == job_id
        assert len(result.nodes) >= 1
        for node in result.nodes:
            assert node.metadata.job_id == job_id
            assert node.metadata.confidence_breakdown is not None
            assert 0.0 <= node.confidence <= 1.0

    async def test_empty_transcript_returns_no_nodes(self, orchestrator, blob_store):
        job_id = str(uuid4())
        doc = await _upload_transcript(blob_store, job_id, "The weather today is nice and sunny.")

        result = await orchestrator.run_identification(doc, job_id)

        assert result.job_id == job_id
        assert result.nodes == []

    async def test_grounding_enabled_populates_grounding_score(self, orchestrator_with_grounding, blob_store):
        job_id = str(uuid4())
        doc = await _upload_transcript(blob_store, job_id, _TRANSCRIPT)

        result = await orchestrator_with_grounding.run_identification(doc, job_id)

        assert len(result.nodes) >= 1
        for node in result.nodes:
            breakdown = node.metadata.confidence_breakdown
            assert breakdown is not None
            assert breakdown.grounding_enabled is True
            assert breakdown.grounding_passed is not None

    async def test_run_resolution_returns_relationships(self, orchestrator, node_repo, blob_store):
        job_id = str(uuid4())
        seed_job_id = str(uuid4())
        await _upload_transcript(blob_store, job_id, _TRANSCRIPT)

        old_decision = make_node(
            node_id="orch-old-decision",
            title="Use MySQL for the user database",
            description="The team previously decided to use MySQL for the user database.",
            type=ConceptType.DECISION,
            status=NodeStatus.APPROVED,
        )
        new_decision = make_node(
            node_id="orch-new-decision",
            title="Use PostgreSQL for the user database",
            description="The team decided to switch to PostgreSQL for better JSON support.",
            type=ConceptType.DECISION,
            status=NodeStatus.APPROVED,
        )

        # old_decision is from a prior job — a candidate target, not a source for this run
        await seed_node(old_decision, node_repo, job_id=seed_job_id)

        # new_decision is from the current job — the source node for resolution
        await seed_node(new_decision, node_repo, job_id=job_id)

        result = await orchestrator.run_resolution(job_id)

        assert result.job_id == job_id
        assert len(result.relationships) >= 1

    async def test_identification_then_resolution_nodes_appear_in_relationships(
        self, orchestrator, node_repo, blob_store
    ):
        """Nodes produced by run_identification are used as sources in run_resolution for the same job_id."""
        job_id = str(uuid4())
        prior_job_id = str(uuid4())
        doc = await _upload_transcript(blob_store, job_id, _TRANSCRIPT)

        # Seed a prior-job node so resolution has a candidate target to resolve against.
        prior_node = make_node(
            node_id="orch-prior-for-resolution",
            title="Use MySQL for the user database",
            description="The team previously decided to use MySQL for the user database.",
            type=ConceptType.DECISION,
            status=NodeStatus.APPROVED,
        )
        await seed_node(prior_node, node_repo, job_id=prior_job_id)

        identification_result = await orchestrator.run_identification(doc, job_id)
        assert len(identification_result.nodes) >= 1

        identified_ids = {node.id for node in identification_result.nodes}
        for node in identification_result.nodes:
            await seed_node(node, node_repo, job_id=job_id)

        resolution_result = await orchestrator.run_resolution(job_id)

        assert resolution_result.job_id == job_id
        source_ids = {rel.source_id for rel in resolution_result.relationships}
        assert identified_ids & source_ids, "no identified node appears as a source in any resolved relationship"
