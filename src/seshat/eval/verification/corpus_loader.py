from __future__ import annotations

from typing import TYPE_CHECKING, Any

import yaml
from pydantic import BaseModel, Field

from seshat.eval.common import matches_tags

if TYPE_CHECKING:
    from pathlib import Path


class VerificationCorpusNode(BaseModel):
    title: str
    description: str
    quote: str
    expected_supported: bool


class VerificationCorpusExample(BaseModel):
    corpus_id: str
    description: str
    transcript: str | None
    nodes: list[VerificationCorpusNode]
    tags: dict[str, Any] = Field(default_factory=dict)


def load_corpus(
    corpus_dir: Path,
    tag_filter: dict[str, str | list[str]] | None = None,
) -> list[VerificationCorpusExample]:
    examples = []
    for path in sorted(corpus_dir.glob("*.yaml")):
        with open(path, encoding="utf-8") as f:
            data: dict[str, Any] = yaml.safe_load(f)
        examples.append(_parse_example(path.stem, data))

    if tag_filter:
        examples = [ex for ex in examples if matches_tags(ex.tags, tag_filter)]

    return examples


def _parse_example(corpus_id: str, data: dict[str, Any]) -> VerificationCorpusExample:
    return VerificationCorpusExample(
        corpus_id=corpus_id,
        description=data["description"],
        transcript=data.get("transcript"),
        nodes=[VerificationCorpusNode(**n) for n in data["nodes"]],
        tags=data.get("tags") or {},
    )
