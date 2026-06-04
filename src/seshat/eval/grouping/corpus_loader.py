from __future__ import annotations

from typing import TYPE_CHECKING, Any

import yaml
from pydantic import BaseModel, Field

from seshat.eval.common import matches_tags

if TYPE_CHECKING:
    from pathlib import Path


class GroupingCorpusItem(BaseModel):
    id: str
    title: str
    description: str
    quote: str = ""  # not used by the grouping agent; omit in corpus files


class GroupingCorpusExample(BaseModel):
    corpus_id: str
    description: str
    items: list[GroupingCorpusItem]
    expected_groups: list[list[str]]  # each inner list is a set of item IDs
    tags: dict[str, Any] = Field(default_factory=dict)


def load_corpus(
    corpus_dir: Path,
    tag_filter: dict[str, str | list[str]] | None = None,
) -> list[GroupingCorpusExample]:
    examples = []
    for path in sorted(corpus_dir.glob("*.yaml")):
        with open(path, encoding="utf-8") as f:
            data: dict[str, Any] = yaml.safe_load(f)
        examples.append(_parse_example(path.stem, data))

    if tag_filter:
        examples = [ex for ex in examples if matches_tags(ex.tags, tag_filter)]

    return examples


def _parse_example(corpus_id: str, data: dict[str, Any]) -> GroupingCorpusExample:
    return GroupingCorpusExample(
        corpus_id=corpus_id,
        description=data["description"],
        items=[GroupingCorpusItem(**item) for item in data["items"]],
        expected_groups=data["expected_groups"],
        tags=data.get("tags") or {},
    )
