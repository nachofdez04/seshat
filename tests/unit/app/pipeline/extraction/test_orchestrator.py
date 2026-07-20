import asyncio
import logging
from datetime import date
from unittest.mock import AsyncMock, MagicMock

import pytest

from seshat.app.agents.grounding import GroundingRetryExhaustedError
from seshat.app.pipeline.extraction.orchestrator import ExtractionOrchestrator, _assemble_kb_hint, _deduplicate
from seshat.app.pipeline.extraction.pending_node import _PendingNode
from seshat.core.config.settings import ExtractionConfig, GroundingLLMConfig
from seshat.core.models.enums import ApprovalMethod, ConceptType, NodeStatus
from seshat.core.utils.tokens import count_tokens
from seshat.infra.blob_store.s3_store import BlobNotFoundError
from tests.helpers import make_anchored_concept, make_doc, make_node

TRANSCRIPT = "we will use PostgreSQL for the main database"


def _make_pending_node(title: str, heuristics: float, description: str = "desc") -> _PendingNode:
    return _PendingNode(
        concept_type=ConceptType.DECISION,
        title=title,
        description=description,
        quote_anchors=[],
        concept_fields={},
        job_id="job-1",
        heuristics=heuristics,
    )


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
        types = concept_types or [ConceptType.DECISION]
        extraction_registry = MagicMock()
        extraction_registry.run_all = AsyncMock(return_value=({ct: extraction_results or [] for ct in types}, []))

    resolution_registry = MagicMock()
    resolution_registry.resolve_all = AsyncMock(return_value=(all_rels or [], []))

    node_retriever = MagicMock()
    node_retriever.retrieve = AsyncMock(return_value=targets or [])
    node_retriever.max_concurrent_retrievals = 10

    node_repo = MagicMock()
    node_repo.paginated_query = AsyncMock(return_value=kb_approved_nodes or [])

    blob_repo = MagicMock()
    blob_repo.get_raw_transcript = AsyncMock(return_value=TRANSCRIPT.encode())

    return ExtractionOrchestrator(
        config=config,
        identification_registry=extraction_registry,
        resolution_registry=resolution_registry,
        node_retriever=node_retriever,
        node_repo=node_repo,
        blob_repo=blob_repo,
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

    async def test_failed_types_from_registry_thread_through_to_result(self):
        """The registry's failed-type list is surfaced on IdentificationResult.failed_concept_types."""
        extraction_registry = MagicMock()
        extraction_registry.run_all = AsyncMock(
            return_value=(
                {ConceptType.DECISION: [_make_concept("Use PostgreSQL")]},
                [ConceptType.ACTION_ITEM, ConceptType.RISK],
            )
        )

        orchestrator = _make_orchestrator(
            extraction_registry=extraction_registry,
            concept_types=[ConceptType.DECISION, ConceptType.ACTION_ITEM, ConceptType.RISK],
        )

        result = await orchestrator.run_identification(make_doc(), job_id="job-1")

        assert len(result.nodes) == 1
        assert ConceptType.ACTION_ITEM in result.failed_concept_types
        assert ConceptType.RISK in result.failed_concept_types

    async def test_empty_extraction_result_not_counted_as_failure(self):
        orchestrator = _make_orchestrator(
            extraction_results=[],
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
        from seshat.app.agents.resolution.base import ResolvedRelationship
        from seshat.core.models.enums import RelationshipType

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
        from seshat.app.agents.grounding import GroundingResult

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
        from seshat.app.agents.grounding import GroundingResult

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
        from seshat.core.models.nodes import FailedResolutionSource

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
        orchestrator._repo.paginated_query = AsyncMock(return_value=[])

        result = await orchestrator.run_resolution(job_id="job-1")

        assert result.relationships == []


class TestJobTimeout:
    async def test_extraction_raises_timeout_when_exceeded(self):
        async def _slow(*_args, **_kwargs):
            await asyncio.sleep(10)
            return {}, []

        registry = MagicMock()
        registry.run_all = _slow

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


class TestConfigOverride:
    async def test_config_override_changes_threshold(self):
        """A config_override with threshold=0.99 should leave a low-confidence node as PENDING_REVIEW
        even when the base config would approve it (threshold=0.0)."""
        concept = _make_concept("Use PostgreSQL", quote="use PostgreSQL")
        orchestrator = _make_orchestrator(
            extraction_results=[concept],
            auto_mode=False,
            confidence_threshold=0.0,
        )

        override = ExtractionConfig(
            concept_types=[ConceptType.DECISION],
            auto_mode=False,
            confidence_threshold=0.99,
        )

        result = await orchestrator.run_identification(make_doc(), job_id="job-1", config_override=override)

        assert result.nodes[0].status == NodeStatus.PENDING_REVIEW

    async def test_all_agents_failed_raises_runtime_error(self):
        extraction_registry = MagicMock()
        extraction_registry.run_all = AsyncMock(return_value=({}, [ConceptType.DECISION]))

        orchestrator = _make_orchestrator(
            extraction_registry=extraction_registry,
            concept_types=[ConceptType.DECISION],
        )

        with pytest.raises(RuntimeError, match="All identification agents failed"):
            await orchestrator.run_identification(make_doc(), job_id="job-1")


class TestKbHintIsolation:
    """KB hint fetching happens in _run_identification, before identification fans out."""

    async def test_prebuilt_hints_are_forwarded_to_run_all_without_querying_kb(self):
        """When hints are supplied to _run_identification, they reach run_all and the KB is not queried."""
        registry = MagicMock()
        registry.run_all = AsyncMock(return_value=({ConceptType.DECISION: []}, []))

        orchestrator = _make_orchestrator(
            extraction_registry=registry,
            concept_types=[ConceptType.DECISION],
        )

        await orchestrator._run_identification(
            TRANSCRIPT,
            "blob-key",
            "job-1",
            hints={ConceptType.DECISION: "prebuilt hint"},
            meeting_date=date(2026, 1, 1),
        )

        orchestrator._repo.paginated_query.assert_not_called()
        assert registry.run_all.call_args.args[2] == {ConceptType.DECISION: "prebuilt hint"}

    async def test_run_identification_gathers_kb_hints_before_agent_calls(self):
        """KB queries for all concept types are issued before any identification call."""
        call_log: list[str] = []

        async def tracking_query(_node_filter):
            call_log.append("kb_query")
            return []

        async def tracking_run_all(*args, **kwargs):
            call_log.append("identify")
            return {}, []

        registry = MagicMock()
        registry.run_all = tracking_run_all

        orchestrator = _make_orchestrator(
            extraction_registry=registry,
            concept_types=[ConceptType.DECISION, ConceptType.RISK],
        )
        orchestrator._repo.paginated_query = tracking_query

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

    def test_first_node_exactly_at_token_cap_is_included_second_breaks(self):
        node1 = make_node("n1", title="A")
        snippet1 = f"{node1.title} (date {node1.metadata.meeting_date.isoformat()}): {node1.description[:80]}"
        cost1 = count_tokens(snippet1)

        node2 = make_node("n2", title="B")

        result = _assemble_kb_hint([node1, node2], max_hint_tokens=cost1)

        lines = [line for line in result.splitlines() if line.strip()]
        assert len(lines) == 1
        assert node1.title in result
        assert node2.title not in result


class TestDeduplication:
    def test_higher_score_incoming_replaces_existing(self, caplog):
        existing = _make_pending_node("Use PostgreSQL", heuristics=0.5, description="short")
        incoming = _make_pending_node("Use PostgreSQL", heuristics=0.9, description="short")

        with caplog.at_level(logging.WARNING, logger="seshat.app.pipeline.extraction.orchestrator"):
            result = _deduplicate([existing, incoming])

        assert len(result) == 1
        assert result[0].heuristics == 0.9
        assert any("replacing" in r.message.lower() for r in caplog.records)

    def test_lower_score_incoming_is_discarded(self, caplog):
        existing = _make_pending_node("Use PostgreSQL", heuristics=0.9)
        incoming = _make_pending_node("Use PostgreSQL", heuristics=0.5)

        with caplog.at_level(logging.WARNING, logger="seshat.app.pipeline.extraction.orchestrator"):
            result = _deduplicate([existing, incoming])

        assert len(result) == 1
        assert result[0].heuristics == 0.9

    def test_equal_score_longer_description_wins(self):
        short = _make_pending_node("Use PostgreSQL", heuristics=0.8, description="short")
        long = _make_pending_node("Use PostgreSQL", heuristics=0.8, description="much longer description here")

        result = _deduplicate([short, long])

        assert len(result) == 1
        assert result[0].description == "much longer description here"


class TestRunIdentificationBlobNotFound:
    async def test_blob_returns_none_raises_blob_not_found_error(self):
        orchestrator = _make_orchestrator()
        orchestrator._blob.get_raw_transcript = AsyncMock(return_value=None)

        with pytest.raises(BlobNotFoundError):
            await orchestrator.run_identification(make_doc(), job_id="job-1")


class TestRunResolutionNoLLMCallWhenNoTargets:
    async def test_approved_nodes_but_retriever_returns_empty_skips_resolve_all(self):
        approved = make_node("n1", status=NodeStatus.APPROVED)
        orchestrator = _make_orchestrator(kb_approved_nodes=[approved])
        orchestrator._retriever.retrieve = AsyncMock(return_value=[])

        result = await orchestrator.run_resolution(job_id="job-1")

        orchestrator._resolution_registry.resolve_all.assert_called_once()
        call_args = orchestrator._resolution_registry.resolve_all.call_args
        per_source_targets = call_args.args[1]
        assert all(len(targets) == 0 for targets in per_source_targets.values())
        assert result.relationships == []
