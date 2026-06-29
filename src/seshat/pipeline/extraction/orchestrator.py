from __future__ import annotations

import asyncio
import time
from datetime import UTC, date, datetime
from typing import TYPE_CHECKING

from seshat.agents.grounding import GroundingRetryExhaustedError
from seshat.models.api_graph import NodeFilter
from seshat.models.enums import ConceptType, NodeStatus
from seshat.models.nodes import (
    ConfidenceBreakdown,
    IdentificationResult,
    KBNode,
    KBRelationship,
    ResolutionResult,
)
from seshat.observability.latency_tracker import track_latency_profile
from seshat.observability.usage_tracker import UsageTracker, track_token_budget
from seshat.pipeline.extraction.heuristics_scorer import HeuristicsScorer
from seshat.pipeline.extraction.pending_node import PendingNodeBuilder, _deduplicate, _PendingNode, _quote_text
from seshat.utils.log import get_logger
from seshat.utils.tokens import count_tokens

if TYPE_CHECKING:
    from uuid import UUID

    from seshat.agents.grounding import GroundingAgent
    from seshat.agents.identification.registry import IdentificationAgentRegistry
    from seshat.agents.resolution.base import ResolvedRelationship
    from seshat.agents.resolution.registry import ResolutionRegistry
    from seshat.blob_store.s3_store import S3BlobStore
    from seshat.config.settings import ExtractionConfig
    from seshat.knowledge_store.pg_store import PostgresKBStore
    from seshat.models.transcript import TranscriptDocument
    from seshat.pipeline.extraction.node_retriever import NodeRetriever

logger = get_logger(__name__)


class ExtractionOrchestrator:
    def __init__(
        self,
        config: ExtractionConfig,
        identification_registry: IdentificationAgentRegistry,
        resolution_registry: ResolutionRegistry,
        node_retriever: NodeRetriever,
        kb_store: PostgresKBStore,
        blob_store: S3BlobStore,
        grounding_agent: GroundingAgent | None = None,
    ) -> None:
        self._config = config
        self._identification_registry = identification_registry
        self._resolution_registry = resolution_registry
        self._retriever = node_retriever
        self._kb = kb_store
        self._blob = blob_store
        self._grounder = grounding_agent
        self._heuristics_scorer = HeuristicsScorer()
        self._job_tracker = UsageTracker.uncapped()

    @property
    def usage(self) -> UsageTracker:
        return self._job_tracker

    @track_token_budget(
        max_input_fn=lambda self: self._config.max_total_input_tokens,
        max_output_fn=lambda self: self._config.max_total_output_tokens,
        label="identification",
        accumulate_to_fn=lambda self: self._job_tracker,
    )
    @track_latency_profile("identification")
    async def run_identification(self, doc: TranscriptDocument, job_id: str) -> IdentificationResult:
        transcript = (await self._blob.get(doc.blob_key)).decode()
        coro = self._run_identification(transcript, doc.blob_key, job_id)
        if self._config.identification_timeout_seconds is not None:
            return await asyncio.wait_for(coro, self._config.identification_timeout_seconds)
        return await coro

    async def _run_identification(
        self,
        transcript: str,
        blob_key: str,
        job_id: str,
        hints: dict[ConceptType, str] | None = None,
    ) -> IdentificationResult:
        t0 = time.perf_counter()
        logger.info("Starting identification run for blob_key=%s", blob_key)

        if hints is None:
            hints = await self._fetch_kb_hints()
        pending, failed_concept_types = await self._identification_pass(transcript, blob_key, job_id, hints)

        nodes = await self._score_and_finalize(pending, transcript)

        elapsed_ms = round((time.perf_counter() - t0) * 1000)
        logger.info(
            "Identification run complete: %d nodes, %d failed types (elapsed: %dms)",
            len(nodes),
            len(failed_concept_types),
            elapsed_ms,
            extra={"elapsed_ms": elapsed_ms},
        )
        nodes_by_type: dict[ConceptType, int] = dict.fromkeys(self.concept_types, 0)
        for node in nodes:
            nodes_by_type[node.type] += 1

        return IdentificationResult(
            job_id=job_id,
            nodes=nodes,
            confidence_breakdowns={node.id: node.metadata.confidence_breakdown for node in nodes},  # type: ignore[misc]
            failed_concept_types=failed_concept_types,
            nodes_by_type=nodes_by_type,
        )

    @track_token_budget(
        max_input_fn=lambda self: self._config.max_total_input_tokens,
        max_output_fn=lambda self: self._config.max_total_output_tokens,
        max_embedding_fn=lambda self: self._config.max_total_embedding_tokens,
        label="resolution",
        accumulate_to_fn=lambda self: self._job_tracker,
    )
    @track_latency_profile("resolution")
    async def run_resolution(self, job_id: str, *, approved: list[KBNode] | None = None) -> ResolutionResult:
        if approved is None:
            approved = await self._kb.paginated_query(NodeFilter(job_id=job_id, status=NodeStatus.APPROVED))

        logger.info("Resolution run: %d approved nodes", len(approved))

        rag_sem = asyncio.Semaphore(self._retriever.max_concurrent_retrievals)

        async def _retrieve(node: KBNode) -> list[KBNode]:
            async with rag_sem:
                return await self._retriever.retrieve(
                    # node_type=None: resolution needs candidates of all types so cross-type
                    # agents (e.g. DECISION→RISK MITIGATES) can find their targets.
                    node,
                    node_filter=NodeFilter(node_type=None),
                    exclude_job_id=job_id,
                )

        retrieval_results = await asyncio.gather(*[_retrieve(node) for node in approved])
        per_source_targets: dict[UUID, list[KBNode]] = {
            node.id: targets for node, targets in zip(approved, retrieval_results, strict=True)
        }

        coro = self._run_resolution(approved, per_source_targets, job_id)
        if self._config.resolution_timeout_seconds is not None:
            return await asyncio.wait_for(coro, self._config.resolution_timeout_seconds)
        return await coro

    async def _run_resolution(
        self,
        source_nodes: list[KBNode],
        per_source_targets: dict[UUID, list[KBNode]],
        job_id: str,
    ) -> ResolutionResult:
        t0 = time.perf_counter()
        logger.info("Starting resolution run for job_id=%s", job_id)

        resolution_sem = asyncio.Semaphore(self._config.resolution.max_global_calls)
        all_rels, failed_sources = await self._resolution_registry.resolve_all(
            source_nodes, per_source_targets, resolution_sem
        )

        relationships = [_build_relationship(rel, job_id) for rel in all_rels]
        elapsed_ms = round((time.perf_counter() - t0) * 1000)
        logger.info(
            "Resolution run complete: %d relationships, %d failed sources (elapsed: %dms)",
            len(relationships),
            len(failed_sources),
            elapsed_ms,
            extra={"elapsed_ms": elapsed_ms, "failed_count": len(failed_sources)},
        )
        return ResolutionResult(
            job_id=job_id,
            relationships=relationships,
            failed_sources=failed_sources,
        )

    @property
    def concept_types(self) -> list[ConceptType]:
        return self._config.concept_types

    async def _fetch_kb_hints(self) -> dict[ConceptType, str]:
        async def _hint_for(concept_type: ConceptType) -> tuple[ConceptType, str]:
            recent = await self._kb.paginated_query(
                NodeFilter(node_type=concept_type, limit=self._config.max_hint_nodes)
            )
            recent.sort(key=lambda n: n.metadata.meeting_date or date.min, reverse=True)
            return concept_type, _assemble_kb_hint(recent, self._config.max_hint_tokens)

        pairs = await asyncio.gather(*[_hint_for(ct) for ct in self.concept_types])
        return dict(pairs)

    async def _identification_pass(
        self, transcript: str, blob_key: str, job_id: str, hints: dict[ConceptType, str]
    ) -> tuple[list[_PendingNode], list[ConceptType]]:
        """Fan-out identification across all concept types, then deduplicate within the meeting."""
        t0 = time.perf_counter()
        logger.info("Identifying %d concept types concurrently", len(self.concept_types))
        tasks = [
            self._identify_concept_type(transcript, blob_key, concept_type, job_id, hints.get(concept_type, ""))
            for concept_type in self.concept_types
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        pending: list[_PendingNode] = []
        failed: list[ConceptType] = []
        for concept_type, result in zip(self.concept_types, results, strict=True):
            if isinstance(result, Exception):
                logger.error("Identification task failed for %s: %s", concept_type, result)
                failed.append(concept_type)
                continue

            assert isinstance(result, list)
            pending.extend(result)

        deduped = _deduplicate(pending)
        elapsed_ms = round((time.perf_counter() - t0) * 1000)
        logger.info(
            "Identification pass done: %d pending nodes after dedup, %d failed types (elapsed: %dms)",
            len(deduped),
            len(failed),
            elapsed_ms,
            extra={"elapsed_ms": elapsed_ms},
        )
        return deduped, failed

    async def _identify_concept_type(
        self,
        transcript: str,
        transcript_file: str,
        concept_type: ConceptType,
        job_id: str,
        kb_hint: str = "",
    ) -> list[_PendingNode]:
        t0 = time.perf_counter()
        agent = self._identification_registry.get(concept_type)
        raw = await agent.identify(transcript, kb_hint, transcript_file)

        builder = PendingNodeBuilder(concept_type, job_id, transcript, self._heuristics_scorer)
        pending = builder.build_all(raw)
        elapsed_ms = round((time.perf_counter() - t0) * 1000)
        logger.info(
            "Identified %d %r node(s) (elapsed: %dms)",
            len(pending),
            concept_type.value,
            elapsed_ms,
            extra={"elapsed_ms": elapsed_ms, "concept_type": concept_type.value},
        )
        return pending

    async def _score_and_finalize(self, pending: list[_PendingNode], transcript: str) -> list[KBNode]:
        """Score pending nodes (heuristics + optional grounding gate), assign status, build KBNodes."""
        grounding_enabled = self._grounder is not None

        if self._config.grounding is not None and not grounding_enabled:
            logger.warning("grounding is configured but no grounding_agent was provided — running heuristics-only")

        t0 = time.perf_counter()
        logger.debug("Scoring %d nodes (grounder %s)", len(pending), "enabled" if grounding_enabled else "disabled")
        if grounding_enabled:
            await self._run_grounding(pending, transcript)

        for pnode in pending:
            pnode.breakdown = ConfidenceBreakdown(
                heuristics=pnode.heuristics,
                grounding_passed=pnode.grounding,
                grounding_enabled=grounding_enabled,
            )
            pnode.assign_status(self._config)

        nodes = [p.build() for p in pending]
        elapsed_ms = round((time.perf_counter() - t0) * 1000)
        logger.info(
            "Scoring done: %d nodes finalized (elapsed: %dms)",
            len(nodes),
            elapsed_ms,
            extra={"elapsed_ms": elapsed_ms},
        )
        return nodes

    async def _run_grounding(self, pending: list[_PendingNode], transcript: str) -> None:
        assert self._config.grounding is not None
        sem = asyncio.Semaphore(self._config.grounding.max_concurrent_calls)

        async def _ground(pnode: _PendingNode) -> None:
            assert self._grounder is not None
            async with sem:
                quote_text = _quote_text(pnode.quote_anchors, transcript)
                try:
                    grounding_result = await self._grounder.verify(
                        pnode.title, pnode.description, quote_text, transcript=transcript
                    )
                except GroundingRetryExhaustedError:
                    logger.warning("Grounding exhausted retries for %r — skipping (grounding_passed=None)", pnode.title)
                else:
                    pnode.grounding = grounding_result.supported

        await asyncio.gather(*[_ground(pnode) for pnode in pending])


def _build_relationship(rel: ResolvedRelationship, job_id: str) -> KBRelationship:
    return KBRelationship(
        source_id=rel.source_id,
        target_id=rel.target_id,
        rel_type=rel.rel_type,
        job_id=job_id,
        created_at=datetime.now(UTC),
    )


def _assemble_kb_hint(nodes: list[KBNode], max_hint_tokens: int) -> str:
    lines: list[str] = []
    used = 0
    for node in nodes:
        date_tag = node.metadata.meeting_date.isoformat() if node.metadata.meeting_date else "unknown"
        snippet = f"{node.title} (date {date_tag}): {node.description[:80]}"
        cost = count_tokens(snippet)
        if used + cost > max_hint_tokens:
            break
        lines.append(snippet)
        used += cost
    return "\n".join(lines)
