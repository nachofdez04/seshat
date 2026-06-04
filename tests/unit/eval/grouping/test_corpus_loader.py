import pytest

from seshat.config.settings import EvalConfig
from seshat.eval.grouping.corpus_loader import load_corpus
from tests.unit.eval.conftest import TagFilterContractTests


@pytest.fixture(scope="class")
def examples(eval_test_corpus: EvalConfig):
    return load_corpus(eval_test_corpus.grouping_corpus_dir)


class TestCorpusLoader:
    def test_loads_examples(self, examples):
        assert len(examples) > 0

    def test_corpus_examples_have_valid_content(self, examples):
        for ex in examples:
            assert ex.corpus_id
            assert ex.description.strip()
            assert len(ex.items) >= 1
            for item in ex.items:
                assert item.id
                assert item.title.strip()
                assert item.description.strip()

    def test_expected_groups_cover_all_items(self, examples):
        for ex in examples:
            all_item_ids = {item.id for item in ex.items}
            grouped_ids = {item_id for group in ex.expected_groups for item_id in group}
            assert grouped_ids == all_item_ids, f"{ex.corpus_id}: expected_groups do not cover all items"


class TestProductionCorpus(TagFilterContractTests):
    load_corpus = staticmethod(load_corpus)
    corpus_dir_attr = "grouping_corpus_dir"
    tag_key = "concept_type"

    def test_all_files_load_and_have_valid_content(self, eval_corpus: EvalConfig):
        examples = load_corpus(eval_corpus.grouping_corpus_dir)
        assert len(examples) > 0

        for ex in examples:
            assert ex.corpus_id
            assert ex.description.strip()
            for item in ex.items:
                assert item.id
                assert item.title.strip()
