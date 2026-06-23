from __future__ import annotations

from typing import TYPE_CHECKING, Any

import yaml

from seshat.eval.models import IdentificationCorpusExample, IdentificationCorpusNode
from seshat.models.enums import ConceptType

if TYPE_CHECKING:
    from pathlib import Path

_BASE_CORPUS_FIELDS: frozenset[str] = frozenset({"quote", "title", "description"})


def load_corpus(corpus_dir: Path) -> list[IdentificationCorpusExample]:
    examples = []
    for path in sorted(corpus_dir.glob("*.yaml")):
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        examples.append(_parse_example(path.stem, data))
    return examples


def _parse_example(corpus_id: str, data: dict[str, Any]) -> IdentificationCorpusExample:
    nodes: list[IdentificationCorpusNode] = []
    for concept_type in ConceptType:
        for entry in data.get("expected", {}).get(concept_type, []):
            extra = {k: v for k, v in entry.items() if k not in _BASE_CORPUS_FIELDS}
            nodes.append(
                IdentificationCorpusNode(
                    quote=entry["quote"],
                    type=concept_type,
                    title=entry["title"],
                    description=entry["description"],
                    extra_fields=extra,
                )
            )
    return IdentificationCorpusExample(
        corpus_id=corpus_id,
        transcript=data["transcript"],
        expected_nodes=nodes,
    )
