from __future__ import annotations

from seshat.app.documents.citations import (
    CitationRegistry,
    escape_footnote_collisions,
    render_footnote_block,
)
from seshat.core.models.quote_anchor import QuoteAnchor


def _anchor(start: int = 0, end: int = 10, transcript_file: str = "transcript.txt") -> QuoteAnchor:
    return QuoteAnchor(transcript_file=transcript_file, char_start=start, char_end=end)


class TestCitationRegistry:
    def test_register_assigns_sequential_one_indexed_numbers(self):
        registry = CitationRegistry()
        assert registry.register(_anchor(0, 10)) == 1
        assert registry.register(_anchor(20, 30)) == 2

    def test_register_is_idempotent_and_dedups_by_span(self):
        registry = CitationRegistry()
        first = registry.register(_anchor(0, 10))

        assert registry.register(_anchor(0, 10)) == first
        # Same offsets in a different file are a distinct source.
        assert registry.register(_anchor(0, 10, transcript_file="other.txt")) == 2
        assert len(registry.ordered_anchors) == 2

    def test_register_all_preserves_order(self):
        registry = CitationRegistry()
        anchors = [_anchor(0, 10), _anchor(20, 30), _anchor(0, 10)]
        assert registry.register_all(anchors) == [1, 2, 1]

    def test_get_returns_anchor_or_none_when_out_of_range(self):
        registry = CitationRegistry()
        anchor = _anchor(0, 10)
        registry.register(anchor)

        assert registry.get(1) == anchor
        assert registry.get(0) is None
        assert registry.get(2) is None

    def test_ordered_anchors_returns_registration_order(self):
        registry = CitationRegistry()
        first, second = _anchor(20, 30), _anchor(0, 10)
        registry.register(first)
        registry.register(second)
        assert registry.ordered_anchors == [first, second]


def test_escape_footnote_collisions_escapes_marker_prefix():
    assert escape_footnote_collisions("see [^1] here") == "see \\[^1] here"
    assert escape_footnote_collisions("no markers") == "no markers"


def test_render_footnote_block_lists_used_indices_only():
    registry = CitationRegistry()
    registry.register(_anchor(0, 5))
    registry.register(_anchor(10, 15))

    block = render_footnote_block(registry, {2}, lambda a: "excerpt")

    assert block == '[^2]: "excerpt" — transcript chars 10-15'


def test_render_footnote_block_sorts_indices():
    registry = CitationRegistry()
    registry.register(_anchor(0, 5))
    registry.register(_anchor(10, 15))

    block = render_footnote_block(registry, {2, 1}, lambda a: f"chars {a.char_start}")

    assert block.splitlines() == [
        '[^1]: "chars 0" — transcript chars 0-5',
        '[^2]: "chars 10" — transcript chars 10-15',
    ]


def test_render_footnote_block_skips_unknown_indices():
    registry = CitationRegistry()
    registry.register(_anchor(0, 5))

    block = render_footnote_block(registry, {1, 99}, lambda a: "excerpt")

    assert "[^99]" not in block
    assert "[^1]" in block


def test_render_footnote_block_sanitizes_excerpts():
    registry = CitationRegistry()
    registry.register(_anchor(0, 5))

    block = render_footnote_block(registry, {1}, lambda a: "multi\nline `code` and [^1] marker")

    assert "\\`code\\`" in block
    assert "\\[^1] marker" in block
    assert "multi line" in block


def test_render_footnote_block_truncates_long_excerpts():
    registry = CitationRegistry()
    registry.register(_anchor(0, 5))

    block = render_footnote_block(registry, {1}, lambda a: "x" * 500)

    assert "x" * 200 + "…" in block
    assert "x" * 201 not in block
