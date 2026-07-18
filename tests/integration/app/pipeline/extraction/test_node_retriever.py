import pytest

from seshat.app.pipeline.extraction.node_retriever import NodeRetriever
from seshat.app.pipeline.extraction.search_engine import SearchEngine
from seshat.core.config.settings import RAGConfig
from seshat.core.models.enums import ConceptType, NodeStatus, RelationshipType
from tests.helpers import make_node
from tests.integration.conftest import SKIP_IF_NO_EMBEDDINGS_API, SKIP_IF_NO_POSTGRES
from tests.integration.helpers import make_relationship, seed_node

pytestmark = [
    # module loop required: kb_store fixture is module-scoped and asyncpg pools are loop-bound
    pytest.mark.asyncio(loop_scope="module"),
    pytest.mark.usefixtures("_truncate_kb_tables", "_reset_vector_store"),
    pytest.mark.integration,
    pytest.mark.llm,
    pytest.mark.embedding,
    SKIP_IF_NO_POSTGRES,
    SKIP_IF_NO_EMBEDDINGS_API,
]


def _make_retriever(node_repo, rag_config: RAGConfig | None = None) -> NodeRetriever:
    cfg = rag_config or RAGConfig()
    search_engine = SearchEngine(
        rag_config=cfg,
        vector_store=node_repo._vs,
        keyword_llm=None,
        multi_query_llm=None,
    )
    return NodeRetriever(rag_config=cfg, node_repo=node_repo, search_engine=search_engine)


@pytest.fixture
def node_retriever(node_repo) -> NodeRetriever:
    return _make_retriever(node_repo)


class TestNodeRetrieverRetrieveCandidates:
    async def test_returns_seeded_node_for_similar_query(self, node_retriever, node_repo):
        seeded = make_node(
            node_id="rag-seed",
            title="Use PostgreSQL for the user database",
            description="The team agreed to use PostgreSQL v15 due to its JSON support and performance.",
            status=NodeStatus.APPROVED,
        )
        await seed_node(seeded, node_repo)

        query_node = make_node(
            node_id="rag-query",
            title="Switch to PostgreSQL",
            description="We should adopt PostgreSQL as our primary database.",
            type=ConceptType.DECISION,
            status=NodeStatus.APPROVED,
        )

        results = await node_retriever.retrieve(query_node)

        assert any(r.id == seeded.id for r in results)

    async def test_orphan_vector_hit_is_silently_skipped(self, node_retriever, vector_store):
        orphan = make_node(
            node_id="rag-orphan",
            title="Use Kafka for the event bus",
            description="The team decided to use Kafka as the event streaming backbone.",
            status=NodeStatus.APPROVED,
        )
        # Write to VS only — intentionally no KB entry to simulate a stale/dangling vector
        await vector_store.upsert(
            str(orphan.id),
            orphan.vector_store_text,
            {"node_type": orphan.type.value, "confidence": orphan.confidence},
        )

        query_node = make_node(
            node_id="rag-query-orphan",
            title="Event streaming with Kafka",
            description="We should adopt Kafka for event streaming.",
            type=ConceptType.DECISION,
            status=NodeStatus.APPROVED,
        )

        results = await node_retriever.retrieve(query_node)

        assert all(r.id != orphan.id for r in results)

    async def test_exclude_job_id_filters_nodes_from_same_job(self, node_retriever, node_repo):
        current_job = "job-current"
        prior_job = "job-prior"

        current_node = make_node(
            node_id="rag-current",
            title="Use PostgreSQL for the user database",
            description="The team agreed to use PostgreSQL v15 due to its JSON support and performance.",
            status=NodeStatus.APPROVED,
        )
        prior_node = make_node(
            node_id="rag-prior",
            title="Use PostgreSQL as the primary database",
            description="Earlier decision to adopt PostgreSQL for the project.",
            status=NodeStatus.APPROVED,
        )
        await seed_node(current_node, node_repo, job_id=current_job)
        await seed_node(prior_node, node_repo, job_id=prior_job)

        query_node = make_node(
            node_id="rag-query-jobfilter",
            title="Switch to PostgreSQL",
            description="We should adopt PostgreSQL as our primary database.",
            type=ConceptType.DECISION,
            status=NodeStatus.APPROVED,
        )

        results = await node_retriever.retrieve(query_node, exclude_job_id=current_job)

        assert all(r.id != current_node.id for r in results)

    async def test_neighbour_expansion_includes_graph_hop(self, node_repo, kb_store):
        # direct_hit is in both vector store and KB — retrieved by vector search
        direct_hit = make_node(
            node_id="rag-neighbour-direct",
            title="Use PostgreSQL for the user database",
            description="The team agreed to use PostgreSQL v15 due to its JSON support.",
            status=NodeStatus.APPROVED,
        )
        await seed_node(direct_hit, node_repo)
        # neighbour is KB-only — reachable only via graph traversal, not vector search. the asymmetry is intentional
        # to verify that neighbours are included even if they wouldn't be retrieved by vector search alone
        neighbour = make_node(
            node_id="rag-neighbour-hop",
            title="Use PostgreSQL v12 for the user database",
            description="Earlier decision to use PostgreSQL v12, superseded by v15.",
            status=NodeStatus.APPROVED,
        )
        await kb_store.write_node(neighbour)
        await node_repo.write_relationship(
            make_relationship(direct_hit, neighbour, rel_type=RelationshipType.SUPERSEDES, job_id="job-neighbour")
        )

        # top_k=1 → cap=2; direct_hit fills seen (len=1 < cap=2) so neighbour expansion runs
        retriever = _make_retriever(node_repo, RAGConfig(top_k=1))
        query_node = make_node(
            node_id="rag-query-neighbour",
            title="Switch to PostgreSQL",
            description="We should adopt PostgreSQL as our primary database.",
            status=NodeStatus.APPROVED,
        )

        results = await retriever.retrieve(query_node)

        assert any(r.id == neighbour.id for r in results)

    async def test_token_budget_truncation_limits_results(self, node_repo):
        nodes = [
            make_node(
                node_id=f"rag-budget-{i}",
                title=f"Use PostgreSQL variant {i}",
                description=f"Decision {i}: team adopted PostgreSQL variant {i} for the user database.",
                status=NodeStatus.APPROVED,
            )
            for i in range(5)
        ]
        for node in nodes:
            await seed_node(node, node_repo)

        # max_context_tokens=1 forces exit after the first node's token cost is counted
        retriever = _make_retriever(node_repo, RAGConfig(max_context_tokens=1))
        query_node = make_node(
            node_id="rag-query-budget",
            title="PostgreSQL database decision",
            description="We need to decide on the PostgreSQL version.",
            status=NodeStatus.APPROVED,
        )

        results = await retriever.retrieve(query_node)

        assert len(results) < len(nodes)
