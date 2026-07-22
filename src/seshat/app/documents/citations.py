from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from seshat.core.models.quote_anchor import QuoteAnchor

_EXCERPT_MAX_CHARS = 200


class CitationRegistry:
    """Assigns sequential 1-indexed numbers to QuoteAnchors, deduplicating by (transcript_file, char_start, char_end).

    Registration is idempotent: registering the same span twice returns the same number.
    """

    def __init__(self) -> None:
        self._anchors: list[QuoteAnchor] = []
        self._index_by_span: dict[tuple[str, int, int], int] = {}

    def register(self, anchor: QuoteAnchor) -> int:
        key = (anchor.transcript_file, anchor.char_start, anchor.char_end)
        existing = self._index_by_span.get(key)
        if existing is not None:
            return existing

        self._anchors.append(anchor)
        index = len(self._anchors)
        self._index_by_span[key] = index
        return index

    def register_all(self, anchors: list[QuoteAnchor]) -> list[int]:
        return [self.register(anchor) for anchor in anchors]

    def get(self, index: int) -> QuoteAnchor | None:
        if 1 <= index <= len(self._anchors):
            return self._anchors[index - 1]

        return None

    @property
    def ordered_anchors(self) -> list[QuoteAnchor]:
        return list(self._anchors)


def escape_footnote_collisions(text: str) -> str:
    """Escape `[^` sequences in LLM-authored text so they cannot collide with footnote syntax."""
    return text.replace("[^", "\\[^")


def render_footnote_block(
    registry: CitationRegistry,
    used: set[int],
    excerpt_for: Callable[[QuoteAnchor], str],
) -> str:
    """Render `[^N]: "<excerpt…>" — transcript chars S-E` lines for used indices only."""
    lines = []
    for index in sorted(used):
        anchor = registry.get(index)
        if anchor is None:
            continue

        excerpt = _sanitize_excerpt(excerpt_for(anchor))
        lines.append(f'[^{index}]: "{excerpt}" — transcript chars {anchor.char_start}-{anchor.char_end}')

    return "\n".join(lines)


def _sanitize_excerpt(text: str) -> str:
    # Footnote definitions must stay on one line, so collapse all whitespace runs first.
    excerpt = " ".join(text.split())
    if len(excerpt) > _EXCERPT_MAX_CHARS:
        excerpt = excerpt[:_EXCERPT_MAX_CHARS] + "…"

    return escape_footnote_collisions(excerpt).replace("`", "\\`")
