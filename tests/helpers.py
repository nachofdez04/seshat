from datetime import date
from uuid import NAMESPACE_DNS, uuid5

from seshat.models.enums import ConceptType, IngestionSource, NodeStatus
from seshat.models.nodes import KBNode, NodeMetadata


def make_node(
    node_id: str = "n1",
    title: str = "Use PostgreSQL",
    confidence: float = 0.9,
    team: str | None = None,
) -> KBNode:
    return KBNode(
        id=uuid5(NAMESPACE_DNS, node_id),
        type=ConceptType.ADR,
        title=title,
        description="Team decided to use PostgreSQL.",
        confidence=confidence,
        source_quote="we will use PostgreSQL",
        status=NodeStatus.AUTO_APPROVED,
        metadata=NodeMetadata(
            job_id="job-1",
            meeting_date=date(2026, 4, 21),
            ingestion_source=IngestionSource.JOB,
            team=team,
        ),
    )
