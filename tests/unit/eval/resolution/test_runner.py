from seshat.eval.models import ResolutionCorpusExample, ResolutionCorpusNode
from seshat.eval.resolution.runner import _is_same_type
from seshat.models.enums import ConceptType


def _node(slug: str, concept_type: ConceptType) -> ResolutionCorpusNode:
    return ResolutionCorpusNode(
        id=slug,
        type=concept_type,
        title="t",
        description="d",
        quote="q",
    )


def _example(source_type: ConceptType, kb_type: ConceptType) -> ResolutionCorpusExample:
    return ResolutionCorpusExample(
        corpus_id="test",
        description="test",
        source_nodes=[_node("src", source_type)],
        kb_nodes=[_node("kb", kb_type)],
        expected_relations=[],
    )


class TestIsSameType:
    def test_same_type_returns_true(self):
        assert _is_same_type(_example(ConceptType.DECISION, ConceptType.DECISION)) is True

    def test_different_types_returns_false(self):
        assert _is_same_type(_example(ConceptType.DECISION, ConceptType.RISK)) is False
