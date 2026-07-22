from __future__ import annotations

import unicodedata
from typing import TYPE_CHECKING

from seshat.app.documents.citations import CitationRegistry, escape_footnote_collisions, render_footnote_block
from seshat.core.models.enums import ConceptType

if TYPE_CHECKING:
    from datetime import date

    from seshat.core.models.nodes import KBNode

_SECTIONS: tuple[tuple[ConceptType, str], ...] = (
    (ConceptType.DECISION, "Decisions"),
    (ConceptType.ACTION_ITEM, "Action Items"),
    (ConceptType.RISK, "Risks"),
    (ConceptType.OPEN_QUESTION, "Open Questions"),
)


def render_meeting_summary(
    job_id: str,
    meeting_date: date | None,
    nodes: list[KBNode],
    transcript: str,
) -> str:
    """Render a deterministic Markdown meeting summary with footnote citations. No I/O, no LLM."""
    # Anchors were computed against the NFKC-normalized transcript (see
    # QuoteAnchor.from_transcript_quote), so excerpts must slice the same normalization.
    norm_transcript = unicodedata.normalize("NFKC", transcript)
    registry = CitationRegistry()
    used: set[int] = set()

    title = "# Meeting Summary"
    if meeting_date is not None:
        title += f" — {meeting_date.isoformat()}"

    lines = [title, "", f"Job: `{job_id}`"]

    for concept_type, heading in _SECTIONS:
        section_nodes = sorted((n for n in nodes if n.type is concept_type), key=lambda n: str(n.id))
        if not section_nodes:
            continue

        lines.extend(["", f"## {heading}", ""])
        for node in section_nodes:
            indices = registry.register_all(node.quote_anchors)
            used.update(indices)
            markers = "".join(f"[^{i}]" for i in indices)
            text = f"**{escape_footnote_collisions(node.title)}** — {escape_footnote_collisions(node.description)}"
            lines.append(f"- {text} {markers}" if markers else f"- {text}")

    footnotes = render_footnote_block(registry, used, lambda a: norm_transcript[a.char_start : a.char_end])
    if footnotes:
        lines.extend(["", "---", "", footnotes])

    return "\n".join(lines) + "\n"
