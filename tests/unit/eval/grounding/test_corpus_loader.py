import pytest

from seshat.config.eval_settings import EvalConfig
from seshat.eval.grounding.corpus_loader import load_corpus
from tests.unit.eval.conftest import TagFilterContractTests


@pytest.fixture(scope="class")
def examples(eval_test_corpus: EvalConfig):
    return load_corpus(eval_test_corpus.grounding_corpus_dir)


class TestCorpusLoader:
    def test_loads_examples(self, examples):
        assert len(examples) > 0

    def test_corpus_examples_have_valid_content(self, examples):
        for ex in examples:
            assert ex.corpus_id
            assert ex.description.strip()
            for node in ex.nodes:
                assert node.title.strip()
                assert node.description.strip()
                assert node.quote.strip()
                assert isinstance(node.expected_supported, bool)

    def test_transcript_optional(self, examples):
        # test corpus file has no transcript — should be None
        ex = examples[0]
        assert ex.transcript is None


class TestProductionCorpus(TagFilterContractTests):
    load_corpus = staticmethod(load_corpus)
    corpus_dir_attr = "grounding_corpus_dir"
    tag_key = "tier"

    def test_all_files_load_and_have_valid_content(self, eval_corpus: EvalConfig):
        examples = load_corpus(eval_corpus.grounding_corpus_dir)
        assert len(examples) > 0

        for ex in examples:
            assert ex.corpus_id
            assert ex.description.strip()
            for node in ex.nodes:
                assert node.title.strip()
                assert node.quote.strip()
                assert isinstance(node.expected_supported, bool)
