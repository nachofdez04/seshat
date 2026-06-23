from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

import yaml

from seshat.eval.models import ResolutionCorpusExample, ResolutionCorpusNode
from seshat.models.enums import NodeState, NodeStatus
from seshat.models.nodes import KBNode, NodeMetadata

if TYPE_CHECKING:
    from pathlib import Path


def load_corpus(corpus_dir: Path) -> list[ResolutionCorpusExample]:
    examples = []
    for path in sorted(corpus_dir.glob("*.yaml")):
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        ex = ResolutionCorpusExample(**data)
        _validate_example(ex)
        examples.append(ex)
    return examples


def _validate_example(ex: ResolutionCorpusExample) -> None:
    known = {n.id for n in ex.source_nodes + ex.kb_nodes}
    missing = [(r.source, r.target) for r in ex.expected_relations if r.source not in known or r.target not in known]
    if missing:
        raise ValueError(f"corpus_id {ex.corpus_id!r}: expected_relations reference unknown slugs: {missing}")


def build_kb_nodes(
    example: ResolutionCorpusExample,
) -> tuple[dict[str, KBNode], dict[str, UUID]]:
    """Build KBNode objects for all nodes in the example; return nodes dict and slug→UUID map."""
    slug_map: dict[str, UUID] = {}
    kb_nodes: dict[str, KBNode] = {}

    for corpus_node in example.source_nodes + example.kb_nodes:
        node_id = uuid4()
        slug_map[corpus_node.id] = node_id
        kb_nodes[corpus_node.id] = _corpus_node_to_kb_node(corpus_node, node_id)

    return kb_nodes, slug_map


def _corpus_node_to_kb_node(node: ResolutionCorpusNode, node_id: UUID) -> KBNode:
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
