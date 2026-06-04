from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from seshat.config.settings import EvalConfig

if TYPE_CHECKING:
    from collections.abc import Callable


pytestmark = pytest.mark.eval

_EVAL_DIR = Path(__file__).parent


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        if Path(item.fspath).is_relative_to(_EVAL_DIR):
            item.add_marker(pytest.mark.eval)


_DATA_ROOT = Path(__file__).parent.parent.parent.parent / "data" / "eval"
_DUMMY_GATE = _DATA_ROOT / "gate.json"


@pytest.fixture(scope="session")
def eval_corpus() -> EvalConfig:
    return EvalConfig(corpus_base_dir=_DATA_ROOT / "corpus", gate_path=_DUMMY_GATE)


@pytest.fixture(scope="session")
def eval_test_corpus() -> EvalConfig:
    return EvalConfig(corpus_base_dir=_DATA_ROOT / "test_corpus", gate_path=_DUMMY_GATE)


class TagFilterContractTests:
    """Mixin that asserts the tag-filter contract for any corpus loader.

    Subclasses must define:
      - load_corpus: callable with signature (corpus_dir, tag_filter=None)
      - corpus_dir_attr: name of the EvalConfig attribute for the corpus dir
      - tag_key: the tag key expected to be present in the production corpus
    """

    load_corpus: Callable[..., list[Any]]
    corpus_dir_attr: str
    tag_key: str

    def _corpus_dir(self, eval_corpus: EvalConfig) -> Path:
        return getattr(eval_corpus, self.corpus_dir_attr)

    def test_tags_are_parsed(self, eval_corpus: EvalConfig) -> None:
        examples = self.load_corpus(self._corpus_dir(eval_corpus))
        tagged = [ex for ex in examples if ex.tags]
        assert tagged, "expected at least one production corpus file to have tags"

    def test_tag_filter_includes_matching(self, eval_corpus: EvalConfig) -> None:
        all_examples = self.load_corpus(self._corpus_dir(eval_corpus))
        values = {ex.tags.get(self.tag_key) for ex in all_examples if self.tag_key in ex.tags}
        assert values, f"expected at least one example with a {self.tag_key!r} tag"

        value = next(iter(values))
        filtered = self.load_corpus(self._corpus_dir(eval_corpus), tag_filter={self.tag_key: value})
        assert all(ex.tags.get(self.tag_key) == value for ex in filtered)
        assert len(filtered) < len(all_examples)

    def test_tag_filter_excludes_non_matching(self, eval_corpus: EvalConfig) -> None:
        filtered = self.load_corpus(self._corpus_dir(eval_corpus), tag_filter={self.tag_key: "__nonexistent__"})
        assert filtered == []

    def test_tag_filter_none_returns_all(self, eval_corpus: EvalConfig) -> None:
        all_examples = self.load_corpus(self._corpus_dir(eval_corpus))
        filtered = self.load_corpus(self._corpus_dir(eval_corpus), tag_filter=None)
        assert len(filtered) == len(all_examples)
