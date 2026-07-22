from __future__ import annotations

from datetime import date

from seshat.app.documents.renderer import render_meeting_summary
from seshat.core.models.enums import ConceptType
from seshat.core.models.quote_anchor import QuoteAnchor
from tests.helpers import make_node

_TRANSCRIPT = "We agreed to ship v2 in March. Alice will own the rollout plan. Latency remains a concern."

_GOLDEN = """\
# Meeting Summary — 2026-04-21

Job: `job-1`

## Decisions

- **Ship v2 in March** — Team agreed to ship v2 in March. [^1]

## Action Items

- **Create rollout plan** — Alice owns the rollout plan. [^2]

## Risks

- **March deadline may slip** — Deadline risk. [^1]

---

[^1]: "We agreed to ship v2 in March." — transcript chars 0-30
[^2]: "Alice will own the rollout plan." — transcript chars 31-63
"""


def _golden_nodes() -> list:
    decision = make_node(
        "n-decision",
        title="Ship v2 in March",
        description="Team agreed to ship v2 in March.",
        quote="We agreed to ship v2 in March.",
        transcript=_TRANSCRIPT,
    )
    action = make_node(
        "n-action",
        title="Create rollout plan",
        description="Alice owns the rollout plan.",
        type=ConceptType.ACTION_ITEM,
        quote="Alice will own the rollout plan.",
        transcript=_TRANSCRIPT,
    )
    # Shares the decision's anchor span — must dedup to the same footnote number.
    risk = make_node(
        "n-risk",
        title="March deadline may slip",
        description="Deadline risk.",
        type=ConceptType.RISK,
        quote_anchors=list(decision.quote_anchors),
    )
    return [decision, action, risk]


def test_golden_meeting_summary():
    result = render_meeting_summary("job-1", date(2026, 4, 21), _golden_nodes(), _TRANSCRIPT)
    assert result == _GOLDEN


def test_deterministic_output():
    first = render_meeting_summary("job-1", date(2026, 4, 21), _golden_nodes(), _TRANSCRIPT)
    second = render_meeting_summary("job-1", date(2026, 4, 21), _golden_nodes(), _TRANSCRIPT)
    assert first == second


def test_deterministic_output_when_nodes_arrive_in_different_order():
    first_node = make_node(
        "n-first",
        title="First decision",
        quote="We agreed to ship v2 in March.",
        transcript=_TRANSCRIPT,
    )
    second_node = make_node(
        "n-second",
        title="Second decision",
        quote="Alice will own the rollout plan.",
        transcript=_TRANSCRIPT,
    )

    forward = render_meeting_summary("job-1", date(2026, 4, 21), [first_node, second_node], _TRANSCRIPT)
    reversed_order = render_meeting_summary(
        "job-1", date(2026, 4, 21), [second_node, first_node], _TRANSCRIPT
    )

    assert forward == reversed_order


def test_shared_anchor_dedups_to_single_footnote():
    result = render_meeting_summary("job-1", date(2026, 4, 21), _golden_nodes(), _TRANSCRIPT)
    footnote_lines = [line for line in result.splitlines() if line.startswith("[^")]
    assert len(footnote_lines) == 2
    assert result.count(" [^1]") == 2  # decision and risk bullets share the number


def test_empty_sections_omitted():
    nodes = [make_node("n1", quote="We agreed to ship v2 in March.", transcript=_TRANSCRIPT)]
    result = render_meeting_summary("job-1", date(2026, 4, 21), nodes, _TRANSCRIPT)

    assert "## Decisions" in result
    assert "## Action Items" not in result
    assert "## Risks" not in result
    assert "## Open Questions" not in result


def test_no_nodes_renders_header_without_footnote_block():
    result = render_meeting_summary("job-1", None, [], _TRANSCRIPT)

    assert result.startswith("# Meeting Summary\n")
    assert "---" not in result
    assert "[^" not in result


def test_node_without_anchors_renders_without_markers():
    nodes = [make_node("n1", quote_anchors=[])]
    result = render_meeting_summary("job-1", date(2026, 4, 21), nodes, _TRANSCRIPT)

    assert "- **Use PostgreSQL** — Team decided to use PostgreSQL." in result
    assert "[^" not in result


def test_title_with_footnote_syntax_is_escaped():
    nodes = [
        make_node(
            "n1",
            title="Weird [^1] title",
            description="Also [^2] in description.",
            quote="We agreed to ship v2 in March.",
            transcript=_TRANSCRIPT,
        )
    ]
    result = render_meeting_summary("job-1", date(2026, 4, 21), nodes, _TRANSCRIPT)

    assert "**Weird \\[^1] title**" in result
    assert "Also \\[^2] in description." in result
    # The real footnote for the anchor is unaffected by the escaping.
    assert '[^1]: "We agreed to ship v2 in March."' in result


def test_transcript_is_nfkc_normalized_before_slicing():
    raw_transcript = "The ﬁnal budget is approved."  # ﬁ ligature widens to "fi" under NFKC
    anchor = QuoteAnchor.from_transcript_quote("final budget is approved", raw_transcript, "test.txt")
    assert anchor is not None
    nodes = [make_node("n1", title="Approve budget", description="Budget approved.", quote_anchors=[anchor])]

    result = render_meeting_summary("job-1", date(2026, 4, 21), nodes, raw_transcript)

    assert '[^1]: "final budget is approved"' in result
