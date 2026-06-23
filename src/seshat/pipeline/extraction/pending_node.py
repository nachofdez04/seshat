from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from seshat.agents.identification.grouping import ConceptGroup
from seshat.models.enums import ApprovalMethod, ConceptType, IngestionSource, NodeStatus
from seshat.models.nodes import ConfidenceBreakdown, KBNode, NodeMetadata
from seshat.utils.log import get_logger

logger = get_logger(__name__)

if TYPE_CHECKING:
    from seshat.agents.identification.base import AnchoredConcept
    from seshat.config.settings import ExtractionConfig
    from seshat.models.quote_anchor import QuoteAnchor
    from seshat.pipeline.extraction.heuristics_scorer import HeuristicsScorer


_BASE_CONCEPT_FIELDS: frozenset[str] = frozenset({"quote", "title", "description"})


def _quote_text(anchors: list[QuoteAnchor], transcript: str) -> str:
    if not anchors:
        return ""

    return "\n".join(transcript[anchor.char_start : anchor.char_end] for anchor in anchors)


@dataclass
class _PendingNode:
    """Accumulates all signals for one identified concept before KBNode is constructed."""

    concept_type: ConceptType
    title: str
    description: str
    quote_anchors: list[QuoteAnchor]
    concept_fields: dict[str, Any]
    job_id: str
    heuristics: float
    grounding: bool | None = None
    breakdown: ConfidenceBreakdown | None = None
    status: NodeStatus = NodeStatus.PENDING_REVIEW
    approval_method: ApprovalMethod | None = None
    approved_at: datetime | None = None
    pending_reason: str | None = None

    def build(self) -> KBNode:
        assert self.breakdown is not None, "breakdown must be set before building KBNode"
        return KBNode(
            type=self.concept_type,
            title=self.title,
            description=self.description,
            confidence=self.breakdown.heuristics,
            quote_anchors=self.quote_anchors,
            status=self.status,
            metadata=NodeMetadata(
                job_id=self.job_id,
                ingestion_source=IngestionSource.JOB,
                confidence_breakdown=self.breakdown,
                approval_method=self.approval_method,
                approved_at=self.approved_at,
                pending_reason=self.pending_reason,
                concept_fields=self.concept_fields or None,
            ),
        )

    def assign_status(self, config: ExtractionConfig) -> None:
        assert self.breakdown is not None

        threshold = config.confidence_threshold
        if config.per_type_thresholds and self.concept_type in config.per_type_thresholds:
            threshold = config.per_type_thresholds[self.concept_type]

        # TODO: set approved_by to job submitter once user_id is threaded through ExtractionConfig
        passes_grounding = self.breakdown.grounding_passed is None or self.breakdown.grounding_passed
        passes_threshold = self.breakdown.heuristics >= threshold

        if config.auto_mode:
            # No human in the loop — nodes either pass the gate or are rejected entirely.
            if passes_grounding and passes_threshold:
                self.status = NodeStatus.APPROVED
                self.approval_method = ApprovalMethod.AUTO
                self.approved_at = datetime.now(UTC)
            else:
                self.status = NodeStatus.REJECTED
        else:
            # Human in the loop — nodes that pass both gates are auto-approved; anything else goes to PENDING_REVIEW.
            # Grounding failures must not be silently auto-approved even in manual mode.
            if passes_grounding and passes_threshold:
                self.status = NodeStatus.APPROVED
                self.approval_method = ApprovalMethod.THRESHOLD
                self.approved_at = datetime.now(UTC)
            elif not passes_threshold:
                self.status = NodeStatus.PENDING_REVIEW
                self.pending_reason = f"heuristics {self.breakdown.heuristics:.2f} < threshold {threshold:.2f}"
            else:
                self.status = NodeStatus.PENDING_REVIEW
                self.pending_reason = "grounding failed"


class PendingNodeBuilder:
    """Constructs _PendingNode instances from raw agent output for a fixed (concept_type, job_id, transcript)."""

    def __init__(
        self,
        concept_type: ConceptType,
        job_id: str,
        transcript: str,
        scorer: HeuristicsScorer,
    ) -> None:
        self._concept_type = concept_type
        self._job_id = job_id
        self._transcript = transcript
        self._scorer = scorer

    def from_anchored(self, anchored_concept: AnchoredConcept) -> _PendingNode:
        item = anchored_concept.item
        anchors = [anchored_concept.quote_anchor] if anchored_concept.quote_anchor is not None else []
        concept_fields = {k: v for k, v in item.model_dump().items() if k not in _BASE_CONCEPT_FIELDS}
        return _PendingNode(
            concept_type=self._concept_type,
            title=item.title,
            description=item.description,
            quote_anchors=anchors,
            concept_fields=concept_fields,
            job_id=self._job_id,
            heuristics=self._scorer.score(_quote_text(anchors, self._transcript), item.title, item.description),
        )

    def from_group(self, group: ConceptGroup) -> _PendingNode:
        anchors = [m.quote_anchor for m in group.members if m.quote_anchor is not None]
        members_fields = [
            {k: v for k, v in m.item.model_dump().items() if k not in _BASE_CONCEPT_FIELDS} for m in group.members
        ]
        return _PendingNode(
            concept_type=self._concept_type,
            title=group.group_title,
            description=group.group_description,
            quote_anchors=anchors,
            concept_fields={"members": members_fields},
            job_id=self._job_id,
            heuristics=self._scorer.score(
                _quote_text(anchors, self._transcript),
                group.group_title,
                group.group_description,
            ),
        )

    def build_all(self, raw: list[AnchoredConcept] | list[ConceptGroup]) -> list[_PendingNode]:
        result = []
        for item in raw:
            if isinstance(item, ConceptGroup):
                result.append(self.from_group(item))
            else:
                result.append(self.from_anchored(item))
        return result


def _deduplicate(pending: list[_PendingNode]) -> list[_PendingNode]:
    by_type: dict[ConceptType, dict[str, _PendingNode]] = {}
    for pnode in pending:
        normalised = " ".join(pnode.title.lower().split())
        bucket = by_type.setdefault(pnode.concept_type, {})
        if normalised in bucket:
            existing = bucket[normalised]

            # Keep higher heuristics score; break ties by longer description.
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
