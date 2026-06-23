from unittest.mock import MagicMock, patch

import pytest

from seshat.agents.identification.base import AnchoredConcept
from seshat.agents.identification.decision import Decision
from seshat.agents.identification.grouping import ConceptGroup
from seshat.config.settings import ExtractionConfig
from seshat.models.enums import ApprovalMethod, ConceptType, IngestionSource, NodeStatus
from seshat.models.nodes import ConfidenceBreakdown
from seshat.models.quote_anchor import QuoteAnchor
from seshat.pipeline.extraction.pending_node import PendingNodeBuilder, _deduplicate, _PendingNode

TRANSCRIPT = "we will use PostgreSQL for the main database"


def _make_decision(
    title: str,
    description: str = "A decision.",
    quote: str = "we will use PostgreSQL",
    decision: str = "Use PostgreSQL",
) -> Decision:
    return Decision(
        title=title,
        description=description,
        quote=quote,
        decision=decision,
        rationale="Not stated",
    )


def _make_anchor(quote: str, transcript: str = TRANSCRIPT) -> QuoteAnchor:
    start = transcript.find(quote)
    return QuoteAnchor(transcript_file="test.txt", char_start=start, char_end=start + len(quote))


def _make_anchored(title: str, quote: str | None = None) -> AnchoredConcept:
    item = _make_decision(title, quote=quote or title)
    anchor = _make_anchor(quote, TRANSCRIPT) if quote and quote in TRANSCRIPT else None
    return AnchoredConcept(item=item, quote_anchor=anchor)


def _make_pending(
    title: str,
    concept_type: ConceptType = ConceptType.DECISION,
    heuristics: float = 0.5,
    description: str = "desc",
) -> _PendingNode:
    return _PendingNode(
        concept_type=concept_type,
        title=title,
        description=description,
        quote_anchors=[],
        concept_fields={},
        job_id="job-1",
        heuristics=heuristics,
    )


def _make_config(**kwargs) -> ExtractionConfig:
    return ExtractionConfig(**kwargs)


def _make_builder(scorer=None) -> PendingNodeBuilder:
    if scorer is None:
        scorer = MagicMock()
        scorer.score = MagicMock(return_value=0.7)
    return PendingNodeBuilder(
        concept_type=ConceptType.DECISION,
        job_id="job-1",
        transcript=TRANSCRIPT,
        scorer=scorer,
    )


class TestDeduplicate:
    @pytest.mark.parametrize(
        ("title_a", "title_b"),
        [
            ("Use PostgreSQL", "Use PostgreSQL"),  # exact match
            ("Use PostgreSQL", "use postgresql"),  # case-insensitive
            ("Use  PostgreSQL", "Use PostgreSQL"),  # whitespace-normalised
        ],
    )
    def test_duplicate_titles_deduplicated_to_one(self, title_a, title_b):
        nodes = [_make_pending(title_a), _make_pending(title_b)]
        assert len(_deduplicate(nodes)) == 1

    def test_different_titles_both_kept(self):
        nodes = [_make_pending("Use PostgreSQL"), _make_pending("Use Redis")]
        assert len(_deduplicate(nodes)) == 2

    def test_same_title_different_types_both_kept(self):
        nodes = [
            _make_pending("Use PostgreSQL", ConceptType.DECISION),
            _make_pending("Use PostgreSQL", ConceptType.RISK),
        ]
        assert len(_deduplicate(nodes)) == 2

    def test_collision_emits_warning_log(self):
        nodes = [_make_pending("Use PostgreSQL"), _make_pending("Use PostgreSQL")]
        with patch("seshat.pipeline.extraction.pending_node.logger") as mock_logger:
            _deduplicate(nodes)
        mock_logger.warning.assert_called_once()

    def test_collision_keeps_higher_heuristics_score(self):
        low = _make_pending("Use PostgreSQL", heuristics=0.3)
        high = _make_pending("Use PostgreSQL", heuristics=0.8)
        result = _deduplicate([low, high])
        assert len(result) == 1
        assert result[0].heuristics == 0.8

    def test_collision_result_is_order_independent(self):
        low = _make_pending("Use PostgreSQL", heuristics=0.3)
        high = _make_pending("Use PostgreSQL", heuristics=0.8)
        result_a = _deduplicate([low, high])
        result_b = _deduplicate([high, low])
        assert result_a[0].heuristics == result_b[0].heuristics == 0.8

    def test_collision_tie_broken_by_longer_description(self):
        short = _make_pending("Use PostgreSQL", heuristics=0.5, description="short")
        long = _make_pending("Use PostgreSQL", heuristics=0.5, description="much longer description here")
        result = _deduplicate([short, long])
        assert len(result) == 1
        assert result[0].description == "much longer description here"


class TestPendingNodeAssignStatus:
    def _pending_with_breakdown(
        self, heuristics: float, concept_type: ConceptType = ConceptType.DECISION
    ) -> _PendingNode:
        node = _make_pending("Use PostgreSQL", concept_type)
        node.breakdown = ConfidenceBreakdown(heuristics=heuristics)
        return node

    def test_auto_mode_above_threshold_sets_auto_approved(self):
        node = self._pending_with_breakdown(0.8)
        node.assign_status(_make_config(auto_mode=True, confidence_threshold=0.5))
        assert node.status == NodeStatus.APPROVED
        assert node.approval_method == ApprovalMethod.AUTO
        assert node.approved_at is not None
        assert node.pending_reason is None

    def test_auto_mode_below_threshold_rejects_without_reason(self):
        node = self._pending_with_breakdown(0.3)
        node.assign_status(_make_config(auto_mode=True, confidence_threshold=0.7))
        assert node.status == NodeStatus.REJECTED
        assert node.approval_method is None
        assert node.pending_reason is None

    def test_above_threshold_sets_threshold_approval(self):
        node = self._pending_with_breakdown(0.8)
        node.assign_status(_make_config(confidence_threshold=0.5))
        assert node.status == NodeStatus.APPROVED
        assert node.approval_method == ApprovalMethod.THRESHOLD
        assert node.approved_at is not None
        assert node.pending_reason is None

    def test_below_threshold_sets_pending_review(self):
        node = self._pending_with_breakdown(0.3)
        node.assign_status(_make_config(confidence_threshold=0.7))
        assert node.status == NodeStatus.PENDING_REVIEW
        assert node.approval_method is None
        assert node.pending_reason == "heuristics 0.30 < threshold 0.70"

    def test_per_type_threshold_overrides_global(self):
        node = self._pending_with_breakdown(0.6, ConceptType.DECISION)
        config = _make_config(
            confidence_threshold=0.9,
            per_type_thresholds={ConceptType.DECISION: 0.5},
        )
        node.assign_status(config)
        assert node.status == NodeStatus.APPROVED

    def test_per_type_threshold_does_not_apply_to_other_type(self):
        node = self._pending_with_breakdown(0.6, ConceptType.RISK)
        config = _make_config(
            confidence_threshold=0.9,
            per_type_thresholds={ConceptType.DECISION: 0.5},
        )
        node.assign_status(config)
        assert node.status == NodeStatus.PENDING_REVIEW

    def _pending_with_grounding(self, heuristics: float, grounding_passed: bool | None) -> _PendingNode:
        node = _make_pending("Use PostgreSQL")
        node.breakdown = ConfidenceBreakdown(heuristics=heuristics, grounding_passed=grounding_passed)
        return node

    def test_manual_mode_grounding_failed_above_threshold_sets_pending_review(self):
        node = self._pending_with_grounding(0.8, False)
        node.assign_status(_make_config(auto_mode=False, confidence_threshold=0.5))
        assert node.status == NodeStatus.PENDING_REVIEW
        assert node.pending_reason == "grounding failed"
        assert node.approval_method is None

    def test_manual_mode_grounding_passed_above_threshold_approves(self):
        node = self._pending_with_grounding(0.8, True)
        node.assign_status(_make_config(auto_mode=False, confidence_threshold=0.5))
        assert node.status == NodeStatus.APPROVED

    def test_manual_mode_grounding_none_above_threshold_approves(self):
        node = self._pending_with_grounding(0.8, None)
        node.assign_status(_make_config(auto_mode=False, confidence_threshold=0.5))
        assert node.status == NodeStatus.APPROVED

    def test_auto_mode_grounding_failed_above_threshold_rejects(self):
        node = self._pending_with_grounding(0.8, False)
        node.assign_status(_make_config(auto_mode=True, confidence_threshold=0.5))
        assert node.status == NodeStatus.REJECTED


class TestPendingNodeBuild:
    def _built_node(self, concept_fields: dict | None = None, heuristics: float = 0.7):
        node = _make_pending("Use PostgreSQL")
        node.breakdown = ConfidenceBreakdown(heuristics=heuristics)
        if concept_fields is not None:
            node.concept_fields = concept_fields
        return node.build()

    def test_build_sets_core_fields(self):
        kb = self._built_node()
        assert kb.type == ConceptType.DECISION
        assert kb.title == "Use PostgreSQL"
        assert kb.description == "desc"
        assert kb.metadata.job_id == "job-1"
        assert kb.metadata.ingestion_source == IngestionSource.JOB

    def test_build_sets_confidence_from_heuristics(self):
        kb = self._built_node(heuristics=0.65)
        assert kb.confidence == 0.65

    def test_build_embeds_confidence_breakdown_in_metadata(self):
        kb = self._built_node(heuristics=0.42)
        assert kb.metadata.confidence_breakdown is not None
        assert kb.metadata.confidence_breakdown.heuristics == 0.42

    def test_build_sets_concept_fields_when_non_empty(self):
        kb = self._built_node(concept_fields={"rationale": "perf"})
        assert kb.metadata.concept_fields == {"rationale": "perf"}

    def test_build_sets_concept_fields_none_when_empty(self):
        kb = self._built_node(concept_fields={})
        assert kb.metadata.concept_fields is None

    def test_build_propagates_status_and_approval(self):
        node = _make_pending("Use PostgreSQL")
        node.breakdown = ConfidenceBreakdown(heuristics=0.8)
        node.assign_status(_make_config(confidence_threshold=0.5))
        kb = node.build()
        assert kb.status == NodeStatus.APPROVED
        assert kb.metadata.approval_method == ApprovalMethod.THRESHOLD

    def test_build_passes_quote_anchors(self):
        node = _make_pending("Use PostgreSQL")
        anchor = _make_anchor("use PostgreSQL")
        node.quote_anchors = [anchor]
        node.breakdown = ConfidenceBreakdown(heuristics=0.5)
        kb = node.build()
        assert kb.quote_anchors == [anchor]


class TestPendingNodeBuilder:
    def test_from_anchored_sets_title_and_description(self):
        builder = _make_builder()
        anchored = _make_anchored("Use PostgreSQL")
        node = builder.from_anchored(anchored)
        assert node.title == "Use PostgreSQL"
        assert node.description == "A decision."

    def test_from_anchored_strips_base_fields_from_concept_fields(self):
        builder = _make_builder()
        anchored = _make_anchored("Use PostgreSQL")
        node = builder.from_anchored(anchored)
        # quote, title, description must not appear in concept_fields
        assert "quote" not in node.concept_fields
        assert "title" not in node.concept_fields
        assert "description" not in node.concept_fields

    def test_from_anchored_retains_type_specific_fields(self):
        builder = _make_builder()
        anchored = _make_anchored("Use PostgreSQL")
        node = builder.from_anchored(anchored)
        assert "decision" in node.concept_fields
        assert "rationale" in node.concept_fields

    def test_from_anchored_sets_anchor_when_present(self):
        builder = _make_builder()
        anchored = _make_anchored("Use PostgreSQL", quote="use PostgreSQL")
        node = builder.from_anchored(anchored)
        assert len(node.quote_anchors) == 1

    def test_from_anchored_empty_anchors_when_none(self):
        builder = _make_builder()
        item = _make_decision("Use PostgreSQL")
        anchored = AnchoredConcept(item=item, quote_anchor=None)
        node = builder.from_anchored(anchored)
        assert node.quote_anchors == []

    def test_from_anchored_calls_scorer_with_quote_text(self):
        scorer = MagicMock()
        scorer.score = MagicMock(return_value=0.6)
        builder = _make_builder(scorer)
        anchored = _make_anchored("Use PostgreSQL", quote="use PostgreSQL")
        node = builder.from_anchored(anchored)
        scorer.score.assert_called_once()
        assert node.heuristics == 0.6

    def test_from_group_uses_group_title_and_description(self):
        builder = _make_builder()
        members = [_make_anchored("Use PostgreSQL"), _make_anchored("Use Redis")]
        group = ConceptGroup(group_title="DB choices", group_description="Database decisions", members=members)
        node = builder.from_group(group)
        assert node.title == "DB choices"
        assert node.description == "Database decisions"

    def test_from_group_collects_anchors_from_all_members(self):
        builder = _make_builder()
        a1 = AnchoredConcept(item=_make_decision("A"), quote_anchor=_make_anchor("use PostgreSQL"))
        a2 = AnchoredConcept(item=_make_decision("B"), quote_anchor=None)
        group = ConceptGroup(group_title="G", group_description="desc", members=[a1, a2])
        node = builder.from_group(group)
        assert len(node.quote_anchors) == 1

    def test_from_group_concept_fields_contains_members(self):
        builder = _make_builder()
        members = [_make_anchored("Use PostgreSQL")]
        group = ConceptGroup(group_title="G", group_description="desc", members=members)
        node = builder.from_group(group)
        assert "members" in node.concept_fields
        assert isinstance(node.concept_fields["members"], list)

    def test_build_all_dispatches_anchored_and_group(self):
        builder = _make_builder()
        anchored = _make_anchored("Use PostgreSQL")
        group = ConceptGroup(
            group_title="DB",
            group_description="DB decisions",
            members=[_make_anchored("Use Redis")],
        )
        nodes = builder.build_all([anchored, group])
        assert len(nodes) == 2
        assert nodes[0].title == "Use PostgreSQL"
        assert nodes[1].title == "DB"
