from datetime import date
from unittest.mock import AsyncMock, MagicMock
from uuid import NAMESPACE_DNS, uuid5

from seshat.agents.identification.base import AnchoredConcept
from seshat.agents.identification.decision import Decision
from seshat.models.enums import ConceptType, IngestionSource, NodeStatus
from seshat.models.nodes import KBNode, NodeMetadata
from seshat.models.quote_anchor import QuoteAnchor
from seshat.models.transcript import TranscriptDocument, TranscriptMetadata


def make_structured_llm(return_value=None, side_effect=None) -> MagicMock:
    """Return a mock LLM whose with_structured_output() returns a mock that wraps ainvoke."""
    structured_llm = MagicMock()
    if side_effect is not None:
        structured_llm.ainvoke = AsyncMock(side_effect=side_effect)
    else:
        structured_llm.ainvoke = AsyncMock(return_value=return_value)

    llm = MagicMock()
    llm.with_structured_output = MagicMock(return_value=structured_llm)
    return llm


def make_anchored_concept(
    title: str,
    description: str = "A decision.",
    quote: str | None = None,
    transcript: str = "",
) -> AnchoredConcept:
    """Build an AnchoredConcept<Decision> with an optional quote anchor."""
    item = Decision(
        title=title,
        description=description,
        quote=quote or title,
        decision=title,
        rationale="Not stated.",
    )
    anchor = None
    if quote is not None and transcript and quote in transcript:
        start = transcript.find(quote)
        anchor = QuoteAnchor(transcript_file="test.txt", char_start=start, char_end=start + len(quote))
    return AnchoredConcept(item=item, quote_anchor=anchor)


def make_doc(
    blob_key: str = "transcripts/meeting.txt",
    meeting_date: date | None = None,
) -> TranscriptDocument:
    return TranscriptDocument(
        source_type="text",
        blob_key=blob_key,
        metadata=TranscriptMetadata(meeting_date=meeting_date or date(2026, 4, 21)),
    )


def make_node(
    node_id: str = "n1",
    title: str = "Use PostgreSQL",
    confidence: float = 0.9,
    team: str | None = None,
    type: ConceptType = ConceptType.DECISION,
    description: str = "Team decided to use PostgreSQL.",
    status: NodeStatus = NodeStatus.APPROVED,
    quote: str | None = None,
    transcript: str | None = None,
    quote_anchors: list[QuoteAnchor] | None = None,
    metadata: NodeMetadata | None = None,
) -> KBNode:
    if quote_anchors is not None:
        anchors = quote_anchors
    elif quote is not None and transcript is not None:
        start = transcript.index(quote)
        anchors = [QuoteAnchor(transcript_file="test.txt", char_start=start, char_end=start + len(quote))]
    else:
        anchors = [QuoteAnchor(transcript_file="test.txt", char_start=0, char_end=22)]

    if metadata is None:
        metadata = NodeMetadata(
            job_id="job-1",
            meeting_date=date(2026, 4, 21),
            ingestion_source=IngestionSource.JOB,
            team=team,
        )

    return KBNode(
        id=uuid5(NAMESPACE_DNS, node_id),
        type=type,
        title=title,
        description=description,
        confidence=confidence,
        quote_anchors=anchors,
        status=status,
        metadata=metadata,
    )
