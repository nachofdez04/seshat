from seshat.eval.identification.matcher import QUOTE_MATCH_THRESHOLD, MatchMethod, match_nodes
from seshat.eval.models import IdentificationCorpusNode
from seshat.models.enums import ConceptType, NodeStatus
from tests.helpers import make_node
from tests.unit.eval.identification.helpers import corpus_node

TRANSCRIPT = (
    "We decided to use PostgreSQL for all operational data. "
    "There is a risk that replication lag could affect reads. "
    "Alice will benchmark the replica setup by Friday."
)


class TestMatchNodes:
    def test_exact_match(self):
        quote = "We decided to use PostgreSQL for all operational data."
        predicted = [make_node(quote=quote, transcript=TRANSCRIPT, type=ConceptType.DECISION)]
        expected = [corpus_node(quote, ConceptType.DECISION)]
        result = match_nodes(TRANSCRIPT, expected, predicted)
        assert len(result.matched) == 1
        assert result.matched[0].match_score >= QUOTE_MATCH_THRESHOLD
        assert len(result.missed) == 0
        assert len(result.spurious) == 0

    def test_type_mismatch_is_not_a_match(self):
        quote = "We decided to use PostgreSQL for all operational data."
        predicted = [make_node(quote=quote, transcript=TRANSCRIPT, type=ConceptType.RISK)]
        expected = [corpus_node(quote, ConceptType.DECISION)]
        result = match_nodes(TRANSCRIPT, expected, predicted)
        assert len(result.matched) == 0
        assert len(result.missed) == 1
        assert len(result.spurious) == 1

    def test_spurious_node(self):
        quote = "Alice will benchmark the replica setup by Friday."
        predicted = [make_node(quote=quote, transcript=TRANSCRIPT, type=ConceptType.ACTION_ITEM)]
        result = match_nodes(TRANSCRIPT, [], predicted)
        assert len(result.spurious) == 1
        assert len(result.matched) == 0

    def test_missed_node(self):
        expected = [corpus_node("There is a risk that replication lag could affect reads.", ConceptType.RISK)]
        result = match_nodes(TRANSCRIPT, expected, [])
        assert len(result.missed) == 1
        assert result.missed[0].quote == expected[0].quote

    def test_both_empty_returns_empty_result(self):
        result = match_nodes(TRANSCRIPT, [], [])
        assert result.matched == []
        assert result.missed == []
        assert result.spurious == []

    def test_greedy_claims_best_pair_first(self):
        # Two expected nodes with distinct quotes; two predicted nodes where the
        # first predicted matches the first expected with a higher score than the
        # second predicted would. Greedy selection must assign each node once only.
        quote_a = "We decided to use PostgreSQL for all operational data."
        quote_b = "There is a risk that replication lag could affect reads."
        expected = [
            corpus_node(quote_a, ConceptType.DECISION),
            corpus_node(quote_b, ConceptType.RISK),
        ]
        predicted = [
            make_node(quote=quote_a, transcript=TRANSCRIPT, type=ConceptType.DECISION),
            make_node(quote=quote_b, transcript=TRANSCRIPT, type=ConceptType.RISK),
        ]
        result = match_nodes(TRANSCRIPT, expected, predicted)
        assert len(result.matched) == 2
        assert len(result.missed) == 0
        assert len(result.spurious) == 0


class TestTitleFallback:
    def test_semantically_matching_node_scores_above_threshold(self):
        """A node with no anchors but matching title and description should match."""
        node = make_node(
            title="Use PostgreSQL for operational data",
            description="The team chose PostgreSQL for all operational data storage.",
            type=ConceptType.DECISION,
            status=NodeStatus.PENDING_REVIEW,
            quote_anchors=[],
        )
        expected = [
            IdentificationCorpusNode(
                quote="We decided to use PostgreSQL for all operational data.",
                type=ConceptType.DECISION,
                title="Use PostgreSQL for operational data",
                description="The team chose PostgreSQL for all operational data storage.",
            )
        ]
        result = match_nodes(TRANSCRIPT, expected, [node])
        assert len(result.matched) == 1
        assert result.matched[0].matched_by == MatchMethod.TITLE_FALLBACK
        assert result.matched[0].match_score >= QUOTE_MATCH_THRESHOLD

    def test_semantically_unrelated_node_scores_below_threshold(self):
        """A node with no anchors and unrelated title+description should not match."""
        node = make_node(
            title="Migrate to Kubernetes",
            description="The team decided to move all services to Kubernetes for orchestration.",
            type=ConceptType.DECISION,
            status=NodeStatus.PENDING_REVIEW,
            quote_anchors=[],
        )
        expected = [
            IdentificationCorpusNode(
                quote="We decided to use PostgreSQL for all operational data.",
                type=ConceptType.DECISION,
                title="Use PostgreSQL for operational data",
                description="The team chose PostgreSQL for all operational data storage.",
            )
        ]
        result = match_nodes(TRANSCRIPT, expected, [node])
        assert len(result.matched) == 0
        assert len(result.missed) == 1
        assert len(result.spurious) == 1
