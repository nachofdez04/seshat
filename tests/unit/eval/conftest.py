from pathlib import Path

import pytest

from seshat.config.settings import EvalConfig

_DATA_ROOT = Path(__file__).parent.parent.parent.parent / "data" / "eval"
_DUMMY_GATE = _DATA_ROOT / "gate.json"


@pytest.fixture(scope="session")
def eval_corpus() -> EvalConfig:
    return EvalConfig(corpus_base_dir=_DATA_ROOT / "corpus", gate_path=_DUMMY_GATE)


@pytest.fixture(scope="session")
def eval_test_corpus() -> EvalConfig:
    return EvalConfig(corpus_base_dir=_DATA_ROOT / "test_corpus", gate_path=_DUMMY_GATE)
