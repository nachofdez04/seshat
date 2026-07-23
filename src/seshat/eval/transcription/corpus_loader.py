from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from seshat.core.config.settings import PROJECT_ROOT
from seshat.eval.corpus_tags import matches_tags
from seshat.eval.models import TranscriptionCorpusExample

if TYPE_CHECKING:
    from seshat.eval.corpus_tags import CorpusTagFilter


def load_corpus(
    corpus_dir: Path,
    tag_filter: CorpusTagFilter | None = None,
) -> list[TranscriptionCorpusExample]:
    examples = []
    for path in sorted(corpus_dir.glob("*.yaml")):
        with open(path, encoding="utf-8") as f:
            data: dict[str, Any] = yaml.safe_load(f)
        examples.append(_parse_example(path.stem, data))

    if tag_filter:
        examples = [ex for ex in examples if matches_tags(ex.tags, tag_filter)]

    return examples


def _parse_example(corpus_id: str, data: dict[str, Any]) -> TranscriptionCorpusExample:
    # The audio is read here, before any paid API call, so a missing or renamed fixture
    # fails the whole load rather than half a run.
    audio_file = Path(data["audio_file"])
    audio_path = PROJECT_ROOT / audio_file
    if not audio_path.is_file():
        raise FileNotFoundError(f"corpus example {corpus_id!r} references a missing audio file: {audio_path}")

    return TranscriptionCorpusExample(
        corpus_id=corpus_id,
        audio_file=audio_file,
        reference=data["reference"],
        audio_sha256=hashlib.sha256(audio_path.read_bytes()).hexdigest(),
        tags=data.get("tags") or {},
    )
