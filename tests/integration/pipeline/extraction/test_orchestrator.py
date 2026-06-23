from uuid import uuid4

import pytest

from seshat.agents.identification.registry import IdentificationAgentRegistry
from seshat.agents.resolution.registry import ResolutionRegistry
from seshat.agents.verification import VerificationAgent
from seshat.blob_store.s3_store import S3BlobStore
from seshat.config.settings import BlobStoreConfig, ExtractionConfig, RAGConfig
from seshat.models.enums import ConceptType, NodeStatus
from seshat.pipeline.extraction.node_retriever import NodeRetriever
from seshat.pipeline.extraction.orchestrator import ExtractionOrchestrator
from tests.helpers import make_doc, make_node
from tests.integration.conftest import (
    LOCALSTACK_REGION,
    LOCALSTACK_TEST_BUCKET,
    SKIP_IF_NO_EMBEDDINGS_API,
    SKIP_IF_NO_LOCALSTACK,
    SKIP_IF_NO_POSTGRES,
)
from tests.integration.helpers import (
    cheap_identification_config,
    cheap_verification_config,
    make_cheap_llm,
    seed_node,
    upload_transcript,
)

pytestmark = [
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


@pytest.fixture
async def blob_store(localstack_s3_url):
    config = BlobStoreConfig(
        bucket=LOCALSTACK_TEST_BUCKET,
        region=LOCALSTACK_REGION,
        endpoint_url=localstack_s3_url,
    )
    store = S3BlobStore(config)
    await store.connect()
    yield store
    await store.close()


@pytest.fixture
def extraction_config():
    llm_cfg = cheap_identification_config()
    return ExtractionConfig(identification=llm_cfg)


def _build_orchestrator(kb_store, vector_store, blob_store, extraction_config, *, verification_agent=None):
    llm = make_cheap_llm()
    rag = NodeRetriever(RAGConfig(), kb_store, vector_store)
    return ExtractionOrchestrator(
        config=extraction_config,
        identification_registry=IdentificationAgentRegistry(llm, extraction_config),
        resolution_registry=ResolutionRegistry(llm, extraction_config.resolution),
        node_retriever=rag,
        kb_store=kb_store,
        blob_store=blob_store,
        verification_agent=verification_agent,
    )


@pytest.fixture
def orchestrator(kb_store, vector_store, blob_store, extraction_config):
    return _build_orchestrator(kb_store, vector_store, blob_store, extraction_config)


@pytest.fixture
def orchestrator_with_verification(kb_store, vector_store, blob_store):
    llm_cfg = cheap_identification_config()
    verification_llm_cfg = cheap_verification_config()
    _extraction_config = ExtractionConfig(identification=llm_cfg, verification=verification_llm_cfg)
    verifier = VerificationAgent(llm=make_cheap_llm(), config=verification_llm_cfg)
    return _build_orchestrator(kb_store, vector_store, blob_store, _extraction_config, verification_agent=verifier)


class TestExtractionOrchestrator:
    async def test_run_identification_returns_nodes_with_confidence(self, orchestrator, blob_store):
        blob_key = await upload_transcript(blob_store, _TRANSCRIPT)
        job_id = str(uuid4())

        result = await orchestrator.run_identification(make_doc(blob_key), job_id)

        assert result.job_id == job_id
        assert len(result.nodes) >= 1
        for node in result.nodes:
            assert node.metadata.job_id == job_id
            assert node.metadata.confidence_breakdown is not None
            assert 0.0 <= node.confidence <= 1.0

    async def test_empty_transcript_returns_no_nodes(self, orchestrator, blob_store):
        blob_key = await upload_transcript(blob_store, "The weather today is nice and sunny.")
        job_id = str(uuid4())

        result = await orchestrator.run_identification(make_doc(blob_key), job_id)

        assert result.job_id == job_id
        assert result.nodes == []

    async def test_verification_enabled_populates_verification_score(self, orchestrator_with_verification, blob_store):
        blob_key = await upload_transcript(blob_store, _TRANSCRIPT)
        job_id = str(uuid4())

        result = await orchestrator_with_verification.run_identification(make_doc(blob_key), job_id)

        assert len(result.nodes) >= 1
        for node in result.nodes:
            breakdown = node.metadata.confidence_breakdown
            assert breakdown is not None
            assert breakdown.verification_enabled is True
            assert breakdown.verification is not None

    async def test_run_resolution_returns_relationships(self, orchestrator, kb_store, vector_store, blob_store):
        blob_key = await upload_transcript(blob_store, _TRANSCRIPT)
        job_id = str(uuid4())
        seed_job_id = str(uuid4())

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
        await seed_node(old_decision, kb_store, vector_store, job_id=seed_job_id)

        # new_decision is from the current job — the source node for resolution
        await seed_node(new_decision, kb_store, vector_store, job_id=job_id)

        result = await orchestrator.run_resolution(make_doc(blob_key), job_id)

        assert result.job_id == job_id
        assert len(result.relationships) >= 1

    async def test_identification_then_resolution_nodes_appear_in_relationships(
        self, orchestrator, kb_store, vector_store, blob_store
    ):
        """Nodes produced by run_identification are used as sources in run_resolution for the same job_id."""
        blob_key = await upload_transcript(blob_store, _TRANSCRIPT)
        job_id = str(uuid4())
        prior_job_id = str(uuid4())

        # Seed a prior-job node so resolution has a candidate target to resolve against.
        prior_node = make_node(
            node_id="orch-prior-for-resolution",
            title="Use MySQL for the user database",
            description="The team previously decided to use MySQL for the user database.",
            type=ConceptType.DECISION,
            status=NodeStatus.APPROVED,
        )
        await seed_node(prior_node, kb_store, vector_store, job_id=prior_job_id)

        identification_result = await orchestrator.run_identification(make_doc(blob_key), job_id)
        assert len(identification_result.nodes) >= 1

        identified_ids = {node.id for node in identification_result.nodes}
        for node in identification_result.nodes:
            await seed_node(node, kb_store, vector_store, job_id=job_id)

        resolution_result = await orchestrator.run_resolution(make_doc(blob_key), job_id)

        assert resolution_result.job_id == job_id
        source_ids = {rel.source_id for rel in resolution_result.relationships}
        assert identified_ids & source_ids, "no identified node appears as a source in any resolved relationship"
