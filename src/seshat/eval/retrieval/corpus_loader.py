from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

import yaml

from seshat.eval.models import RetrievalCorpusExample, RetrievalCorpusNode
from seshat.models.enums import NodeState, NodeStatus
from seshat.models.nodes import KBNode, NodeMetadata

if TYPE_CHECKING:
    from pathlib import Path


def load_corpus(corpus_dir: Path) -> list[RetrievalCorpusExample]:
    examples = []
    for path in sorted(corpus_dir.glob("*.yaml")):
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        ex = RetrievalCorpusExample(**data)
        _validate_example(ex)
        examples.append(ex)
    return examples


def _validate_example(ex: RetrievalCorpusExample) -> None:
    known = {n.id for n in ex.candidate_nodes}
    missing = [s for s in ex.expected_relevant_ids if s not in known]
    if missing:
        raise ValueError(f"corpus_id {ex.corpus_id!r}: expected_relevant_ids {missing} not found in candidate_nodes")


def build_kb_nodes(
    example: RetrievalCorpusExample,
) -> tuple[KBNode, list[KBNode], dict[str, UUID]]:
    """Return (query_kb_node, candidate_kb_nodes, slug→UUID map)."""
    slug_map: dict[str, UUID] = {}

    query_id = uuid4()
    slug_map[example.query_node.id] = query_id
    query_kb_node = _to_kb_node(example.query_node, query_id)

    candidate_kb_nodes = []
    for cn in example.candidate_nodes:
        node_id = uuid4()
        slug_map[cn.id] = node_id
        candidate_kb_nodes.append(_to_kb_node(cn, node_id))

    return query_kb_node, candidate_kb_nodes, slug_map


def _to_kb_node(node: RetrievalCorpusNode, node_id: UUID) -> KBNode:
    return KBNode(
        id=node_id,
        type=node.type,
        title=node.title,
        description=node.description,
        confidence=1.0,
        quote_anchors=[],
        status=NodeStatus.APPROVED,
        state=NodeState.CURRENT,
        metadata=NodeMetadata(
            job_id="eval",
            approved_at=datetime.now(UTC),
        ),
    )
