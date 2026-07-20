from __future__ import annotations

import asyncio
import time
from datetime import UTC, date, datetime
from typing import TYPE_CHECKING

from seshat.app.agents.grounding import GroundingRetryExhaustedError
from seshat.app.pipeline.extraction.heuristics_scorer import HeuristicsScorer
from seshat.app.pipeline.extraction.pending_node import PendingNodeBuilder, _PendingNode, _quote_text
from seshat.app.platform.observability.latency_tracker import track_latency_profile
from seshat.app.platform.observability.usage_tracker import UsageTracker, track_token_budget
from seshat.app.repositories.blob_repository import BlobRepository
from seshat.core.models.api_graph import NodeFilter
from seshat.core.models.enums import ConceptType, NodeStatus
from seshat.core.models.nodes import (
    ConfidenceBreakdown,
    IdentificationResult,
    KBNode,
    KBRelationship,
    ResolutionResult,
)
from seshat.core.utils.log import get_logger
from seshat.core.utils.tokens import count_tokens
from seshat.infra.blob_store.s3_store import BlobNotFoundError

if TYPE_CHECKING:
    from uuid import UUID

    from seshat.app.agents.grounding import GroundingAgent
    from seshat.app.agents.identification.registry import IdentificationRegistry
    from seshat.app.agents.resolution.base import ResolvedRelationship
    from seshat.app.agents.resolution.registry import ResolutionRegistry
    from seshat.app.pipeline.extraction.node_retriever import NodeRetriever
    from seshat.app.repositories.node_repository import NodeRepository
    from seshat.core.config.settings import ExtractionConfig
    from seshat.core.models.transcript import TranscriptDocument

logger = get_logger(__name__)


class ExtractionOrchestrator:
    def __init__(
        self,
        config: ExtractionConfig,
        identification_registry: IdentificationRegistry,
        resolution_registry: ResolutionRegistry,
        node_retriever: NodeRetriever,
        node_repo: NodeRepository,
        blob_repo: BlobRepository,
        grounding_agent: GroundingAgent | None = None,
    ) -> None:
        self._config = config
        self._identification_registry = identification_registry
        self._resolution_registry = resolution_registry
        self._retriever = node_retriever
        self._repo = node_repo
        self._blob = blob_repo
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
    async def run_identification(
        self,
        doc: TranscriptDocument,
        job_id: str,
        user_id: str | None = None,
        config_override: ExtractionConfig | None = None,
    ) -> IdentificationResult:
        raw = await self._blob.get_raw_transcript(doc.metadata.meeting_date, job_id)
        if raw is None:
            raise BlobNotFoundError("", BlobRepository.raw_transcript_key(doc.metadata.meeting_date, job_id))

        transcript = raw.decode()
        coro = self._run_identification(
            transcript,
            doc.blob_key,
            job_id,
            meeting_date=doc.metadata.meeting_date,
            user_id=user_id,
            config_override=config_override,
        )
        if self._config.identification_timeout_seconds is not None:
            return await asyncio.wait_for(coro, self._config.identification_timeout_seconds)
        return await coro

    async def _run_identification(
        self,
        transcript: str,
        blob_key: str,
        job_id: str,
        hints: dict[ConceptType, str] | None = None,
        meeting_date: date | None = None,
        user_id: str | None = None,
        config_override: ExtractionConfig | None = None,
    ) -> IdentificationResult:
        t0 = time.perf_counter()
        logger.info("Starting identification run for blob_key=%s", blob_key)

        if hints is None:
            hints = await self._fetch_kb_hints()
        pending, failed_concept_types = await self._identification_pass(
            transcript, blob_key, job_id, hints, meeting_date=meeting_date
        )

        if len(failed_concept_types) == len(self.concept_types):
            raise RuntimeError(f"All identification agents failed: {[ct.value for ct in failed_concept_types]}")

        nodes = await self._score_and_finalize(pending, transcript, user_id=user_id, config_override=config_override)

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
            approved = await self._repo.paginated_query(NodeFilter(job_id=job_id, status=NodeStatus.APPROVED))

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
            recent = await self._repo.paginated_query(
                NodeFilter(node_type=concept_type, limit=self._config.max_hint_nodes)
            )
            recent.sort(key=lambda n: n.metadata.meeting_date or date.min, reverse=True)
            return concept_type, _assemble_kb_hint(recent, self._config.max_hint_tokens)

        pairs = await asyncio.gather(*[_hint_for(ct) for ct in self.concept_types])
        return dict(pairs)

    async def _identification_pass(
        self,
        transcript: str,
        blob_key: str,
        job_id: str,
        hints: dict[ConceptType, str],
        meeting_date: date | None = None,
    ) -> tuple[list[_PendingNode], list[ConceptType]]:
        """Identify all concept types, build pending nodes, then deduplicate within the meeting."""
        t0 = time.perf_counter()
        logger.info("Identifying %d concept types concurrently", len(self.concept_types))
        raw_by_type, failed = await self._identification_registry.run_all(
            transcript, blob_key, hints, concept_types=self.concept_types
        )

        pending: list[_PendingNode] = []
        for concept_type, raw in raw_by_type.items():
            builder = PendingNodeBuilder(
                concept_type, job_id, transcript, self._heuristics_scorer, meeting_date=meeting_date
            )
            pending.extend(builder.build_all(raw))

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

    async def _score_and_finalize(
        self,
        pending: list[_PendingNode],
        transcript: str,
        user_id: str | None = None,
        config_override: ExtractionConfig | None = None,
    ) -> list[KBNode]:
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
            pnode.assign_status(config_override or self._config, user_id=user_id)

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


def _deduplicate(pending: list[_PendingNode]) -> list[_PendingNode]:
    # TOCONSIDER: use a more sophisticated deduplication strategy (e.g. fuzzy matching, cosine similarity)
    by_type: dict[ConceptType, dict[str, _PendingNode]] = {}
    for pnode in pending:
        normalised = " ".join(pnode.title.lower().split())
        bucket = by_type.setdefault(pnode.concept_type, {})
        if normalised in bucket:
            existing = bucket[normalised]
            if (pnode.heuristics, len(pnode.description)) <= (existing.heuristics, len(existing.description)):
                logger.warning(
                    "Dedup collision for %s/%r — keeping existing (score %.2f >= incoming %.2f)",
                    pnode.concept_type.value,
                    pnode.title,
                    existing.heuristics,
                    pnode.heuristics,
                    extra={"concept_type": pnode.concept_type.value, "title": pnode.title},
                )
                continue
            logger.warning(
                "Dedup collision for %s/%r — replacing existing (score %.2f) with higher-scoring incoming (%.2f)",
                pnode.concept_type.value,
                pnode.title,
                existing.heuristics,
                pnode.heuristics,
                extra={"concept_type": pnode.concept_type.value, "title": pnode.title},
            )

        bucket[normalised] = pnode

    return [pnode for group in by_type.values() for pnode in group.values()]


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
