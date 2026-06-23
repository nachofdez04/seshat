from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from datetime import UTC, date, datetime
from typing import TYPE_CHECKING

from seshat.agents.verification import VerificationRetryExhaustedError
from seshat.models.api import NodeFilter
from seshat.models.enums import ConceptType, NodeStatus
from seshat.models.nodes import (
    IdentificationResult,
    KBNode,
    KBRelationship,
    ResolutionResult,
)
from seshat.pipeline.extraction.heuristics_scorer import HeuristicsScorer
from seshat.pipeline.extraction.pending_node import PendingNodeBuilder, _deduplicate, _PendingNode, _quote_text
from seshat.pipeline.extraction.weighted_scorer import compute_confidence
from seshat.utils.log import get_logger
from seshat.utils.retry import async_retry

if TYPE_CHECKING:
    from uuid import UUID

    from seshat.agents.identification.registry import IdentificationAgentRegistry
    from seshat.agents.resolution.base import ResolvedRelationship
    from seshat.agents.resolution.registry import ResolutionRegistry
    from seshat.agents.verification import VerificationAgent
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
        verification_agent: VerificationAgent | None = None,
    ) -> None:
        self._config = config
        self._identification_registry = identification_registry
        self._resolution_registry = resolution_registry
        self._retriever = node_retriever
        self._kb = kb_store
        self._blob = blob_store
        self._verifier = verification_agent
        self._heuristics_scorer = HeuristicsScorer()

    async def run_identification(self, doc: TranscriptDocument, job_id: str) -> IdentificationResult:
        transcript = await self._fetch_transcript(doc.blob_key)
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
        # TODO: implement token budget enforcement (max_total_input_tokens / max_total_output_tokens).
        # Approach: LangChain callback handler or explicit tracker injection — warn at cap, abort at
        # n*cap. Start with LLM calls only; embeddings and transcription can be added later
        # (RAG context is already bounded by max_context_tokens).
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
        nodes_by_type: dict[ConceptType, int] = defaultdict(int, dict.fromkeys(self.concept_types, 0))
        for node in nodes:
            nodes_by_type[node.type] += 1

        return IdentificationResult(
            job_id=job_id,
            nodes=nodes,
            confidence_breakdowns={node.id: node.metadata.confidence_breakdown for node in nodes},  # type: ignore[misc]
            failed_concept_types=failed_concept_types,
            nodes_by_type=dict(nodes_by_type),
        )

    async def run_resolution(self, doc: TranscriptDocument, job_id: str) -> ResolutionResult:
        transcript = await self._fetch_transcript(doc.blob_key)
        approved = await self._query(NodeFilter(job_id=job_id, status=NodeStatus.APPROVED))
        logger.info("Resolution run: %d approved nodes retrieved", len(approved))
        # TODO: resolution should run exactly once, after the job is fully settled:
        # - auto_mode / all-above-threshold jobs: trigger immediately after identification (current path, correct).
        # - jobs with PENDING_REVIEW nodes: defer until the reviewer has acted on every pending node
        #   (job transitions AWAITING_REVIEW → REVIEWED), then run resolution once against all approved nodes.
        # Running resolution on partial approval and re-running on each subsequent reviewer action causes
        # the full O(4N) LLM fan-out to repeat for already-resolved nodes on every approval event.

        rag_sem = asyncio.Semaphore(self._retriever.max_concurrent_retrievals)

        async def _retrieve(node: KBNode) -> list[KBNode]:
            async with rag_sem:
                return await self._retriever.retrieve(
                    node, transcript, node_filter=NodeFilter(node_type=None), exclude_job_id=job_id
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

    @async_retry()
    async def _fetch_transcript(self, blob_key: str) -> str:
        return (await self._blob.get(blob_key)).decode()

    async def _query(self, node_filter: NodeFilter) -> list[KBNode]:
        """Paginate through all matching nodes, respecting node_filter.limit as the page size."""

        @async_retry()
        async def _paginated_query(offset: int) -> list[KBNode]:
            return await self._kb.query(node_filter.model_copy(update={"offset": offset}))

        results: list[KBNode] = []
        page_size = node_filter.limit
        offset = node_filter.offset
        while True:
            page = await _paginated_query(offset)
            results.extend(page)
            if len(page) < page_size:
                break
            offset += page_size
        return results

    async def _fetch_kb_hints(self) -> dict[ConceptType, str]:
        async def _hint_for(concept_type: ConceptType) -> tuple[ConceptType, str]:
            recent = await self._query(NodeFilter(node_type=concept_type, limit=self._config.max_hint_nodes))
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
            "Identified %d concepts for %s (elapsed: %dms)",
            len(pending),
            concept_type.value,
            elapsed_ms,
            extra={"elapsed_ms": elapsed_ms, "concept_type": concept_type.value},
        )
        return pending

    async def _score_and_finalize(self, pending: list[_PendingNode], transcript: str) -> list[KBNode]:
        """Score pending nodes (heuristics + optional verification), assign confidence and status, build KBNodes."""
        verification_enabled = self._verifier is not None

        t0 = time.perf_counter()
        logger.debug("Scoring %d nodes (verifier=%s)", len(pending), "enabled" if verification_enabled else "disabled")
        if verification_enabled:
            assert self._config.verification is not None
            sem = asyncio.Semaphore(self._config.verification.max_concurrent_calls)

            async def _verify(pnode: _PendingNode) -> None:
                assert self._verifier is not None
                async with sem:
                    quote_text = _quote_text(pnode.quote_anchors, transcript)
                    try:
                        verification_result = await self._verifier.verify(
                            pnode.title, pnode.description, quote_text, transcript=transcript
                        )
                    except VerificationRetryExhaustedError:
                        logger.warning(
                            "Verification exhausted retries for %r — skipping (verification=None)", pnode.title
                        )
                    else:
                        pnode.verification = 1.0 if verification_result.supported else 0.0

            await asyncio.gather(*[_verify(pnode) for pnode in pending])

        disabled = set() if verification_enabled else {"verification"}
        active_weights = self._config.confidence_weights.redistribute(disabled)

        for pnode in pending:
            pnode.breakdown = compute_confidence(
                verification=pnode.verification,
                heuristics=pnode.heuristics,
                weights=active_weights,
                verification_enabled=verification_enabled,
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
        cost = len(snippet) // 4
        if used + cost > max_hint_tokens:
            break
        lines.append(snippet)
        used += cost
    return "\n".join(lines)
