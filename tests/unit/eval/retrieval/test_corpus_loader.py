import pytest

from seshat.config.settings import EvalConfig
from seshat.eval.retrieval.corpus_loader import build_kb_nodes, load_corpus


@pytest.fixture(scope="class")
def examples(eval_test_corpus: EvalConfig):
    return load_corpus(eval_test_corpus.retrieval_corpus_dir)


class TestCorpusLoader:
    def test_loads_example(self, examples):
        assert len(examples) > 0


class TestProductionCorpus:
    def test_all_files_load_and_ids_resolve(self, eval_corpus: EvalConfig):
        examples = load_corpus(eval_corpus.retrieval_corpus_dir)
        assert len(examples) > 0

        for ex in examples:
            _, _, slug_map = build_kb_nodes(ex)
            candidate_ids = {cn.id for cn in ex.candidate_nodes}
            for rel_id in ex.expected_relevant_ids:
                assert rel_id in candidate_ids, f"{ex.corpus_id}: unknown relevant id {rel_id!r}"
            assert ex.query_node.id in slug_map, f"{ex.corpus_id}: query node id missing from slug_map"
