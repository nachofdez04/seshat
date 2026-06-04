import uuid

import pytest

from seshat.config.settings import EvalConfig
from seshat.eval.resolution.corpus_loader import build_kb_nodes, load_corpus
from tests.unit.eval.conftest import TagFilterContractTests


@pytest.fixture(scope="class")
def examples(eval_test_corpus: EvalConfig):
    return load_corpus(eval_test_corpus.resolution_corpus_dir)


class TestCorpusLoader:
    def test_loads_example(self, examples):
        assert len(examples) > 0

    def test_build_kb_nodes_with_slug_map(self, examples):
        ex = examples[0]
        all_slugs = [n.id for n in ex.source_nodes + ex.kb_nodes]
        kb_nodes, slug_map = build_kb_nodes(ex)

        for slug in all_slugs:
            assert slug in slug_map
            assert slug in kb_nodes

        for uid in slug_map.values():
            assert isinstance(uid, uuid.UUID)


class TestProductionCorpus(TagFilterContractTests):
    load_corpus = staticmethod(load_corpus)
    corpus_dir_attr = "resolution_corpus_dir"
    tag_key = "tier"

    def test_all_files_load_and_slugs_resolve(self, eval_corpus: EvalConfig):
        examples = load_corpus(eval_corpus.resolution_corpus_dir)
        assert len(examples) > 0

        for ex in examples:
            _, slug_map = build_kb_nodes(ex)
            all_slugs = set(slug_map.keys())
            for r in ex.expected_relations:
                assert r.source in all_slugs, f"{ex.corpus_id}: unknown source slug {r.source!r}"
                assert r.target in all_slugs, f"{ex.corpus_id}: unknown target slug {r.target!r}"
