import pytest
import yaml

from seshat.core.config.eval_settings import EvalConfig
from seshat.eval.transcription.corpus_loader import load_corpus
from tests.unit.eval.conftest import TagFilterContractTests


@pytest.fixture(scope="class")
def examples(eval_test_corpus: EvalConfig):
    return load_corpus(eval_test_corpus.transcription_corpus_dir)


def _write_example(corpus_dir, name: str, data: dict) -> None:
    corpus_dir.mkdir(parents=True, exist_ok=True)
    (corpus_dir / f"{name}.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")


class TestCorpusLoader:
    def test_loads_examples(self, examples):
        assert len(examples) > 0

    def test_corpus_examples_have_valid_content(self, examples):
        for ex in examples:
            assert ex.corpus_id
            assert ex.reference.strip()
            assert ex.resolved_audio_path.is_file()

    def test_audio_sha256_is_computed_from_the_file(self, examples):
        import hashlib

        ex = examples[0]
        assert ex.audio_sha256 == hashlib.sha256(ex.resolved_audio_path.read_bytes()).hexdigest()

    def test_missing_audio_file_raises_at_load_time(self, tmp_path):
        _write_example(
            tmp_path,
            "broken",
            {"audio_file": "data/fixtures/audio/does_not_exist.mp3", "reference": "hello"},
        )
        with pytest.raises(FileNotFoundError, match="missing audio file"):
            load_corpus(tmp_path)

    def test_corpus_id_comes_from_the_filename(self, examples):
        assert all(ex.corpus_id == ex.corpus_id.strip() for ex in examples)
        assert "001_basic" in {ex.corpus_id for ex in examples}


class TestProductionCorpus(TagFilterContractTests):
    load_corpus = staticmethod(load_corpus)
    corpus_dir_attr = "transcription_corpus_dir"
    tag_key = "background_noise"

    def test_all_files_load_and_have_valid_content(self, eval_corpus: EvalConfig):
        examples = load_corpus(eval_corpus.transcription_corpus_dir)
        assert len(examples) > 0

        for ex in examples:
            assert ex.corpus_id
            assert ex.reference.strip()
            assert ex.audio_sha256
            assert ex.resolved_audio_path.is_file()

    def test_tag_values_are_strings_so_cli_tag_filters_match(self, eval_corpus: EvalConfig):
        # The CLI can only produce string tag values and matches_tags compares by equality,
        # so a YAML bool would make `--tag background_noise=true` silently match nothing.
        examples = load_corpus(eval_corpus.transcription_corpus_dir)
        for ex in examples:
            for key, value in ex.tags.items():
                assert isinstance(value, str), f"{ex.corpus_id}: tag {key!r} is {type(value).__name__}, not str"
