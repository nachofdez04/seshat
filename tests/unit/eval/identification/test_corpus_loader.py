import pytest

from seshat.config.settings import EvalConfig
from seshat.eval.identification.corpus_loader import load_corpus
from seshat.models.enums import ConceptType
from tests.unit.eval.conftest import TagFilterContractTests


@pytest.fixture(scope="class")
def examples(eval_test_corpus: EvalConfig):
    return load_corpus(eval_test_corpus.identification_corpus_dir)


class TestCorpusLoader:
    def test_loads_examples(self, examples):
        assert len(examples) > 0

    def test_corpus_examples_have_valid_content(self, examples):
        for ex in examples:
            assert ex.transcript.strip()
            for node in ex.expected_nodes:
                assert node.quote.strip()
                assert isinstance(node.type, ConceptType)


class TestProductionCorpus(TagFilterContractTests):
    load_corpus = staticmethod(load_corpus)
    corpus_dir_attr = "identification_corpus_dir"
    tag_key = "tier"

    def test_all_files_load_and_have_valid_content(self, eval_corpus: EvalConfig):
        examples = load_corpus(eval_corpus.identification_corpus_dir)
        assert len(examples) > 0

        for ex in examples:
            assert ex.corpus_id
            assert ex.transcript.strip()
            for node in ex.expected_nodes:
                assert node.quote.strip()
                assert isinstance(node.type, ConceptType)
                assert node.title.strip()
                assert node.description.strip()
