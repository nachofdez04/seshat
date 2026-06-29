import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from seshat.agents.grounding import GroundingRetryExhaustedError
from seshat.agents.identification.base import IdentificationRetryExhaustedError
from seshat.config.settings import ExtractionConfig, GroundingLLMConfig
from seshat.models.enums import ApprovalMethod, ConceptType, NodeStatus
from seshat.pipeline.extraction.orchestrator import ExtractionOrchestrator, _assemble_kb_hint
from tests.helpers import make_anchored_concept, make_doc, make_node

TRANSCRIPT = "we will use PostgreSQL for the main database"


def _make_concept(title: str, description: str = "A decision.", quote: str | None = None):
    return make_anchored_concept(title, description=description, quote=quote, transcript=TRANSCRIPT)


def _make_orchestrator(
    extraction_results: list | None = None,
    all_rels: list | None = None,
    targets: list | None = None,
    grounder=None,
    auto_mode: bool = False,
    confidence_threshold: float = 0.5,
    extraction_registry=None,
    concept_types: list[ConceptType] | None = None,
    kb_approved_nodes: list | None = None,
    identification_timeout_seconds: float | None = None,
    resolution_timeout_seconds: float | None = None,
):
    config = ExtractionConfig(
        concept_types=concept_types or [ConceptType.DECISION],
        auto_mode=auto_mode,
        confidence_threshold=confidence_threshold,
        grounding=GroundingLLMConfig() if grounder is not None else None,
        identification_timeout_seconds=identification_timeout_seconds,
        resolution_timeout_seconds=resolution_timeout_seconds,
    )

    if extraction_registry is None:
        agent = MagicMock()
        agent.identify = AsyncMock(return_value=extraction_results or [])
        extraction_registry = MagicMock()
        extraction_registry.get = MagicMock(return_value=agent)

    resolution_registry = MagicMock()
    resolution_registry.resolve_all = AsyncMock(return_value=(all_rels or [], []))

    node_retriever = MagicMock()
    node_retriever.retrieve = AsyncMock(return_value=targets or [])
    node_retriever.max_concurrent_retrievals = 10

    kb_store = MagicMock()
    kb_store.paginated_query = AsyncMock(return_value=kb_approved_nodes or [])

    blob_store = MagicMock()
    blob_store.get = AsyncMock(return_value=TRANSCRIPT.encode())

    return ExtractionOrchestrator(
        config=config,
        identification_registry=extraction_registry,
        resolution_registry=resolution_registry,
        node_retriever=node_retriever,
        kb_store=kb_store,
        blob_store=blob_store,
        grounding_agent=grounder,
    )


class TestExtractionOrchestrator:
    async def test_returns_nodes_from_extraction(self):
        concept = _make_concept("Use PostgreSQL", quote="use PostgreSQL")
        orchestrator = _make_orchestrator(extraction_results=[concept])

        result = await orchestrator.run_identification(make_doc(), job_id="job-1")

        assert len(result.nodes) == 1
        assert result.nodes[0].title == "Use PostgreSQL"
        assert result.nodes[0].metadata.job_id == "job-1"
        assert result.nodes[0].metadata.confidence_breakdown.grounding_enabled is False

    async def test_empty_extraction_returns_empty_result(self):
        orchestrator = _make_orchestrator(extraction_results=[])

        result = await orchestrator.run_identification(make_doc(), job_id="job-1")

        assert result.nodes == []

    async def test_failed_extraction_tasks_land_in_failed_concept_types(self):
        """Both generic exceptions and IdentificationRetryExhaustedError populate failed_concept_types."""
        good_agent = MagicMock()
        good_agent.identify = AsyncMock(return_value=[_make_concept("Use PostgreSQL")])
        runtime_error_agent = MagicMock()
        runtime_error_agent.identify = AsyncMock(side_effect=RuntimeError("LLM timeout"))
        exhausted_agent = MagicMock()
        exhausted_agent.identify = AsyncMock(
            side_effect=IdentificationRetryExhaustedError("agent RISK exhausted 3 retries")
        )

        def _get_agent(ct):
            if ct == ConceptType.DECISION:
                return good_agent
            if ct == ConceptType.ACTION_ITEM:
                return runtime_error_agent
            return exhausted_agent

        extraction_registry = MagicMock()
        extraction_registry.get = MagicMock(side_effect=_get_agent)

        orchestrator = _make_orchestrator(
            extraction_registry=extraction_registry,
            concept_types=[ConceptType.DECISION, ConceptType.ACTION_ITEM, ConceptType.RISK],
        )

        result = await orchestrator.run_identification(make_doc(), job_id="job-1")

        assert len(result.nodes) == 1
        assert ConceptType.ACTION_ITEM in result.failed_concept_types
        assert ConceptType.RISK in result.failed_concept_types

    async def test_empty_extraction_result_not_counted_as_failure(self):
        agent = MagicMock()
        agent.identify = AsyncMock(return_value=[])

        extraction_registry = MagicMock()
        extraction_registry.get = MagicMock(return_value=agent)

        orchestrator = _make_orchestrator(
            extraction_registry=extraction_registry,
            concept_types=[ConceptType.DECISION],
        )

        result = await orchestrator.run_identification(make_doc(), job_id="job-1")

        assert result.nodes == []
        assert result.failed_concept_types == []

    async def test_auto_mode_assigns_auto_approved_status(self):
        concept = _make_concept("Use PostgreSQL")
        orchestrator = _make_orchestrator(extraction_results=[concept], auto_mode=True, confidence_threshold=0.0)

        result = await orchestrator.run_identification(make_doc(), job_id="job-1")

        node = result.nodes[0]
        assert node.status == NodeStatus.APPROVED
        assert node.metadata.approval_method == ApprovalMethod.AUTO

    async def test_high_confidence_assigns_threshold_approval(self):
        concept = _make_concept("Use PostgreSQL", quote="use PostgreSQL")
        orchestrator = _make_orchestrator(
            extraction_results=[concept],
            confidence_threshold=0.01,
        )

        result = await orchestrator.run_identification(make_doc(), job_id="job-1")

        node = result.nodes[0]
        assert node.status == NodeStatus.APPROVED
        assert node.metadata.approval_method == ApprovalMethod.THRESHOLD

    async def test_low_confidence_assigns_pending_review(self):
        concept = _make_concept("maybe something", description="unclear")
        orchestrator = _make_orchestrator(
            extraction_results=[concept],
            confidence_threshold=0.99,
        )

        result = await orchestrator.run_identification(make_doc(), job_id="job-1")

        assert result.nodes[0].status == NodeStatus.PENDING_REVIEW

    async def test_relationships_built_from_resolution(self):
        from seshat.agents.resolution.base import ResolvedRelationship
        from seshat.models.enums import RelationshipType

        candidate = make_node("n2", title="Use MySQL")
        approved_node = make_node("n1", title="Use PostgreSQL", status=NodeStatus.APPROVED)

        def make_rel():
            rel = MagicMock(spec=ResolvedRelationship)
            rel.source_id = approved_node.id
            rel.target_id = candidate.id
            rel.rel_type = RelationshipType.SUPERSEDES
            return rel

        orchestrator = _make_orchestrator(
            targets=[candidate],
            all_rels=[make_rel()],
            kb_approved_nodes=[approved_node],
        )

        result = await orchestrator.run_resolution(job_id="job-1")

        assert len(result.relationships) == 1
        assert result.relationships[0].rel_type == RelationshipType.SUPERSEDES

    async def test_grounding_called_when_grounder_present(self):
        from seshat.agents.grounding import GroundingResult

        concept = _make_concept("Use PostgreSQL")

        grounder = MagicMock()
        grounder.verify = AsyncMock(return_value=GroundingResult(supported=True))

        orchestrator = _make_orchestrator(extraction_results=[concept], grounder=grounder)

        result = await orchestrator.run_identification(make_doc(), job_id="job-1")

        grounder.verify.assert_called_once()
        assert len(result.nodes) == 1
        assert result.nodes[0].metadata.confidence_breakdown.grounding_enabled is True

    async def test_grounding_exhausted_retries_leaves_node_with_heuristics_only(self):
        concept = _make_concept("Use PostgreSQL", quote="use PostgreSQL")

        grounder = MagicMock()
        grounder.verify = AsyncMock(side_effect=GroundingRetryExhaustedError("exhausted"))

        orchestrator = _make_orchestrator(
            extraction_results=[concept],
            grounder=grounder,
        )

        result = await orchestrator.run_identification(make_doc(), job_id="job-1")

        assert len(result.nodes) == 1
        assert result.nodes[0].metadata.confidence_breakdown.grounding_passed is None
        assert result.nodes[0].metadata.confidence_breakdown.grounding_enabled is True

    async def test_grounding_supported_node_approved_unsupported_rejected_in_auto_mode(self):
        from seshat.agents.grounding import GroundingResult

        concept = _make_concept("Use PostgreSQL", quote="use PostgreSQL")

        supported_grounder = MagicMock()
        supported_grounder.verify = AsyncMock(return_value=GroundingResult(supported=True))

        unsupported_grounder = MagicMock()
        unsupported_grounder.verify = AsyncMock(return_value=GroundingResult(supported=False))

        result_supported = await _make_orchestrator(
            extraction_results=[concept],
            grounder=supported_grounder,
            auto_mode=True,
            confidence_threshold=0.0,
        ).run_identification(make_doc(), job_id="job-1")

        result_unsupported = await _make_orchestrator(
            extraction_results=[concept],
            grounder=unsupported_grounder,
            auto_mode=True,
            confidence_threshold=0.0,
        ).run_identification(make_doc(), job_id="job-1")

        assert result_supported.nodes[0].status == NodeStatus.APPROVED
        assert result_unsupported.nodes[0].status == NodeStatus.REJECTED

    async def test_nodes_by_type_counts_per_type(self):
        concepts = [
            _make_concept("Use PostgreSQL"),
            _make_concept("Use Redis"),
        ]
        orchestrator = _make_orchestrator(extraction_results=concepts)

        result = await orchestrator.run_identification(make_doc(), job_id="job-1")

        assert result.nodes_by_type[ConceptType.DECISION] == 2

    async def test_nodes_by_type_zero_for_configured_type_with_no_results(self):
        orchestrator = _make_orchestrator(
            extraction_results=[],
            concept_types=[ConceptType.DECISION, ConceptType.RISK],
        )

        result = await orchestrator.run_identification(make_doc(), job_id="job-1")

        assert result.nodes_by_type[ConceptType.DECISION] == 0
        assert result.nodes_by_type[ConceptType.RISK] == 0

    async def test_resolution_retrieves_without_type_filter(self):
        approved_node = make_node("n1", status=NodeStatus.APPROVED)
        orchestrator = _make_orchestrator(kb_approved_nodes=[approved_node])

        await orchestrator.run_resolution(job_id="job-1")

        call_kwargs = orchestrator._retriever.retrieve.call_args.kwargs
        node_filter = call_kwargs["node_filter"]
        assert node_filter.node_type is None

    async def test_resolution_failed_sources_propagated_to_result(self):
        from seshat.models.nodes import FailedResolutionSource

        approved_node = make_node("n1", status=NodeStatus.APPROVED)
        failed_source = FailedResolutionSource(node_id=approved_node.id, concept_type=ConceptType.DECISION)

        orchestrator = _make_orchestrator(kb_approved_nodes=[approved_node])
        orchestrator._resolution_registry.resolve_all = AsyncMock(return_value=([], [failed_source]))

        result = await orchestrator.run_resolution(job_id="job-1")

        assert result.relationships == []
        assert len(result.failed_sources) == 1
        assert result.failed_sources[0].node_id == approved_node.id

    async def test_run_resolution_with_no_approved_nodes_returns_empty(self):
        orchestrator = _make_orchestrator()
        orchestrator._kb.paginated_query = AsyncMock(return_value=[])

        result = await orchestrator.run_resolution(job_id="job-1")

        assert result.relationships == []


class TestJobTimeout:
    async def test_extraction_raises_timeout_when_exceeded(self):
        async def _slow(*_args, **_kwargs):
            await asyncio.sleep(10)
            return []

        agent = MagicMock()
        agent.identify = _slow
        registry = MagicMock()
        registry.get = MagicMock(return_value=agent)

        orchestrator = _make_orchestrator(
            extraction_registry=registry,
            concept_types=[ConceptType.DECISION],
            identification_timeout_seconds=0.01,
        )

        with pytest.raises(asyncio.TimeoutError):
            await orchestrator.run_identification(make_doc(), job_id="job-1")

    async def test_resolution_raises_timeout_when_exceeded(self):
        async def _slow(*_args, **_kwargs):
            await asyncio.sleep(10)
            return [], []

        orchestrator = _make_orchestrator(resolution_timeout_seconds=0.01)
        orchestrator._resolution_registry.resolve_all = _slow

        with pytest.raises(asyncio.TimeoutError):
            await orchestrator.run_resolution(job_id="job-1")


class TestKbHintIsolation:
    """KB hint fetching happens in _run_identification, not inside _identify_concept_type."""

    async def test_identify_concept_type_accepts_kb_hint_and_does_not_query_kb(self):
        """_identify_concept_type takes kb_hint as a parameter and never calls the KB itself."""
        concept = _make_concept("Use PostgreSQL", quote="use PostgreSQL")
        agent = MagicMock()
        agent.identify = AsyncMock(return_value=[concept])
        registry = MagicMock()
        registry.get = MagicMock(return_value=agent)

        orchestrator = _make_orchestrator(
            extraction_registry=registry,
            concept_types=[ConceptType.DECISION],
        )

        await orchestrator._identify_concept_type(
            TRANSCRIPT, "blob-key", ConceptType.DECISION, "job-1", kb_hint="prebuilt hint"
        )

        orchestrator._kb.paginated_query.assert_not_called()
        args, _ = agent.identify.call_args
        assert args[1] == "prebuilt hint"

    async def test_run_identification_gathers_kb_hints_before_agent_calls(self):
        """KB queries for all concept types are issued before any agent.identify call."""
        call_log: list[str] = []

        async def tracking_query(_node_filter):
            call_log.append("kb_query")
            return []

        async def tracking_identify(*args, **kwargs):
            call_log.append("identify")
            return []

        agent = MagicMock()
        agent.identify = tracking_identify
        registry = MagicMock()
        registry.get = MagicMock(return_value=agent)

        orchestrator = _make_orchestrator(
            extraction_registry=registry,
            concept_types=[ConceptType.DECISION, ConceptType.RISK],
        )
        orchestrator._kb.paginated_query = tracking_query

        await orchestrator.run_identification(make_doc(), job_id="job-1")

        # All KB queries must precede any identify call
        last_kb = max(i for i, e in enumerate(call_log) if e == "kb_query")
        first_identify = min(i for i, e in enumerate(call_log) if e == "identify")
        assert last_kb < first_identify


class TestAssembleKbHint:
    def test_empty_nodes_returns_empty_string(self):
        assert _assemble_kb_hint([], max_hint_tokens=1000) == ""

    def test_single_node_appears_in_output(self):
        node = make_node("n1")
        result = _assemble_kb_hint([node], max_hint_tokens=1000)
        assert "Use PostgreSQL" in result
        assert "2026-04-21" in result

    def test_token_cap_truncates_nodes(self):
        nodes = [make_node(f"n{i}", title=f"Decision {'x' * 50} {i}") for i in range(20)]
        result = _assemble_kb_hint(nodes, max_hint_tokens=20)
        lines = [line for line in result.splitlines() if line.strip()]
        assert len(lines) < 20

    def test_unknown_date_shown_when_missing(self):
        node = make_node("n1")
        node = node.model_copy(update={"metadata": node.metadata.model_copy(update={"meeting_date": None})})
        result = _assemble_kb_hint([node], max_hint_tokens=1000)
        assert "unknown" in result
