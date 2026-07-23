from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from seshat.core.config.eval_settings import EvalConfig
from seshat.core.config.settings import TranscriptionConfig
from seshat.core.models.enums import TranscriptionProvider
from seshat.eval.cache import build_cache_fp
from seshat.eval.gate import write_gate
from seshat.eval.models import GateResult, MetricEntry, TranscriptionCorpusExample, TranscriptionPrediction
from seshat.eval.transcription.runner import (
    TranscriptionEvalRunner,
    _aggregate_metrics,
    _build_breakdown,
    _build_dataframe,
)

# ── helpers ───────────────────────────────────────────────────────────────────


class FakeTranscriber:
    """In-memory transcriber: returns a canned hypothesis and counts calls. Costs nothing."""

    def __init__(self, hypothesis: str = "") -> None:
        self._hypothesis = hypothesis
        self.calls = 0

    async def transcribe(self, audio_bytes: bytes, extension: str) -> str:
        self.calls += 1
        return self._hypothesis

    async def ping(self) -> None:
        return None


def _make_example(
    corpus_id: str = "ex_001",
    reference: str = "we need a database",
    audio_sha256: str = "a" * 64,
    tags: dict | None = None,
) -> TranscriptionCorpusExample:
    return TranscriptionCorpusExample(
        corpus_id=corpus_id,
        audio_file="data/fixtures/audio/example.mp3",
        reference=reference,
        audio_sha256=audio_sha256,
        tags=tags or {},
    )


def _make_runner(
    eval_config: EvalConfig,
    provider: TranscriptionProvider = TranscriptionProvider.ASSEMBLYAI,
    model: str | None = None,
    language: str = "en",
    transcriber: FakeTranscriber | None = None,
) -> TranscriptionEvalRunner:
    transcription_config = TranscriptionConfig(provider=provider, model=model, language=language)
    return TranscriptionEvalRunner(
        transcriber=transcriber or FakeTranscriber(),  # type: ignore[arg-type]
        transcription_config=transcription_config,
        config=eval_config,
    )


def _cache_path(runner: TranscriptionEvalRunner, example: TranscriptionCorpusExample):
    return build_cache_fp(runner._config.transcription_cache_dir, example, agent_hash=runner._provider_hash)


# ── cache fingerprint ─────────────────────────────────────────────────────────


class TestCacheFingerprint:
    def test_provider_change_invalidates_the_entry(self, eval_test_corpus: EvalConfig):
        example = _make_example()
        assemblyai = _make_runner(eval_test_corpus, provider=TranscriptionProvider.ASSEMBLYAI)
        openai = _make_runner(eval_test_corpus, provider=TranscriptionProvider.OPENAI)
        assert _cache_path(assemblyai, example) != _cache_path(openai, example)

    def test_model_change_invalidates_the_entry(self, eval_test_corpus: EvalConfig):
        example = _make_example()
        default = _make_runner(eval_test_corpus)
        pinned = _make_runner(eval_test_corpus, model="best-v2")
        assert _cache_path(default, example) != _cache_path(pinned, example)

    def test_language_change_invalidates_the_entry(self, eval_test_corpus: EvalConfig):
        example = _make_example()
        english = _make_runner(eval_test_corpus, language="en")
        spanish = _make_runner(eval_test_corpus, language="es")
        assert _cache_path(english, example) != _cache_path(spanish, example)

    def test_regenerated_audio_invalidates_the_entry(self, eval_test_corpus: EvalConfig):
        runner = _make_runner(eval_test_corpus)
        original = _make_example(audio_sha256="a" * 64)
        regenerated = _make_example(audio_sha256="b" * 64)
        assert _cache_path(runner, original) != _cache_path(runner, regenerated)

    def test_identical_inputs_reuse_the_same_entry(self, eval_test_corpus: EvalConfig):
        first = _make_runner(eval_test_corpus)
        second = _make_runner(eval_test_corpus)
        assert _cache_path(first, _make_example()) == _cache_path(second, _make_example())


# ── _build_dataframe ──────────────────────────────────────────────────────────


def test_build_dataframe_one_row_per_example():
    df = _build_dataframe([_make_example("ex1"), _make_example("ex2")])
    assert len(df) == 2


def test_build_dataframe_has_required_columns():
    df = _build_dataframe([_make_example()])
    assert set(df.columns) == {"inputs", "expectations", "tags"}


def test_build_dataframe_carries_corpus_id_and_reference():
    df = _build_dataframe([_make_example("abc", reference="the spoken words")])
    assert df.iloc[0]["inputs"]["corpus_id"] == "abc"
    assert df.iloc[0]["expectations"]["reference"] == "the spoken words"


def test_build_dataframe_tags_prefixed_with_corpus_and_stringified():
    df = _build_dataframe([_make_example(tags={"background_noise": "true", "speakers": 4})])
    tags = df.iloc[0]["tags"]
    assert tags["corpus.background_noise"] == "true"
    assert tags["corpus.speakers"] == "4"


def test_build_dataframe_empty_input():
    df = _build_dataframe([])
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0


# ── _aggregate_metrics ────────────────────────────────────────────────────────


def _eval_result(macro: float | None = None) -> SimpleNamespace:
    metrics = {} if macro is None else {"transcription.wer/mean": macro}
    return SimpleNamespace(metrics=metrics)


def test_aggregate_metrics_pooled_wer_is_length_weighted():
    # 1-word example fully wrong, 9-word example perfect → pooled 1/10, macro 0.5.
    long_ref = "one two three four five six seven eight nine"
    examples = [_make_example("short", reference="alpha"), _make_example("long", reference=long_ref)]
    cache = {
        "short": TranscriptionPrediction(text="beta"),
        "long": TranscriptionPrediction(text=long_ref),
    }
    metrics = _aggregate_metrics(examples, cache, _eval_result(macro=0.5))
    assert metrics["wer"] == pytest.approx(0.1)
    assert metrics["wer_macro"] == pytest.approx(0.5)


def test_aggregate_metrics_perfect_transcription_is_zero():
    examples = [_make_example("ex1", reference="we need a database")]
    cache = {"ex1": TranscriptionPrediction(text="We need a database.")}
    assert _aggregate_metrics(examples, cache, _eval_result())["wer"] == pytest.approx(0.0)


def test_aggregate_metrics_omits_macro_when_mlflow_did_not_report_it():
    examples = [_make_example("ex1")]
    cache = {"ex1": TranscriptionPrediction(text="we need a database")}
    assert "wer_macro" not in _aggregate_metrics(examples, cache, _eval_result())


def test_aggregate_metrics_skips_examples_without_a_prediction():
    examples = [_make_example("ex1", reference="hello world"), _make_example("missing", reference="ignored entirely")]
    cache = {"ex1": TranscriptionPrediction(text="hello world")}
    assert _aggregate_metrics(examples, cache, _eval_result())["wer"] == pytest.approx(0.0)


# ── _build_breakdown ──────────────────────────────────────────────────────────


def test_build_breakdown_keys_match_corpus_ids():
    examples = [_make_example("alpha"), _make_example("beta")]
    cache = {"alpha": TranscriptionPrediction(text="x"), "beta": TranscriptionPrediction(text="y")}
    assert set(_build_breakdown(examples, cache)) == {"alpha", "beta"}


def test_build_breakdown_records_per_example_wer_and_texts():
    ex = _make_example("ex1", reference="we need a database", tags={"background_noise": "true"})
    cache = {"ex1": TranscriptionPrediction(text="we need a warehouse")}
    entry = _build_breakdown([ex], cache)["ex1"]

    assert entry["wer"] == pytest.approx(0.25)
    assert entry["reference"] == "we need a database"
    assert entry["hypothesis"] == "we need a warehouse"
    assert entry["tags"] == {"background_noise": "true"}


def test_build_breakdown_missing_prediction_yields_none_values():
    entry = _build_breakdown([_make_example("ex1")], {})["ex1"]
    assert entry["wer"] is None
    assert entry["hypothesis"] is None


# ── run (whole pipeline, no paid calls) ───────────────────────────────────────


@pytest.fixture(scope="module")
def local_mlflow(tmp_path_factory):
    """Point MLflow at a throwaway sqlite store so evaluate() runs without a tracking server."""
    import mlflow

    store = tmp_path_factory.mktemp("mlflow")
    mlflow.set_tracking_uri("sqlite:///" + str(store / "mlflow.db"))
    mlflow.set_experiment("transcription-runner-unit")


@pytest.fixture
def isolated_config(tmp_path, monkeypatch, eval_test_corpus: EvalConfig) -> EvalConfig:
    """Test corpus, with the gate file and prediction cache redirected into tmp_path."""
    monkeypatch.setattr(EvalConfig, "_cache_dir", tmp_path / "eval_cache")
    return EvalConfig(corpus_base_dir=eval_test_corpus.corpus_base_dir, gate_path=tmp_path / "eval_gate.json")


def _corpus_reference(config: EvalConfig) -> str:
    from seshat.eval.transcription.corpus_loader import load_corpus

    return load_corpus(config.transcription_corpus_dir)[0].reference


async def _run_in_own_mlflow_run(runner: TranscriptionEvalRunner, **kwargs):
    """Each run needs its own MLflow run: evaluate() reuses the active one, and two harness
    runs sharing a run collide when logging the same param keys with different values."""
    import mlflow

    with mlflow.start_run():
        return await runner.run(**kwargs)


class TestTranscriptionEvalRunner:
    async def test_run_scores_the_corpus_and_writes_the_gate(self, local_mlflow, isolated_config):
        transcriber = FakeTranscriber(_corpus_reference(isolated_config))
        runner = _make_runner(isolated_config, transcriber=transcriber)

        gate = await _run_in_own_mlflow_run(runner)

        assert transcriber.calls == 1
        assert gate.transcription_metrics is not None
        # a perfect hypothesis scores 0.0 and clears the upper-bound threshold
        assert gate.transcription_metrics["wer"].value == pytest.approx(0.0)
        assert gate.transcription_metrics["wer"].passed is True
        assert gate.transcription_metrics["wer_macro"].gated is False
        assert isolated_config.gate_path.exists()

    async def test_run_reuses_the_cached_hypothesis(self, local_mlflow, isolated_config):
        reference = _corpus_reference(isolated_config)
        await _run_in_own_mlflow_run(_make_runner(isolated_config, transcriber=FakeTranscriber(reference)))

        second = FakeTranscriber(reference)
        await _run_in_own_mlflow_run(_make_runner(isolated_config, transcriber=second))

        assert second.calls == 0

    async def test_run_without_gate_update_leaves_the_gate_file_alone(self, local_mlflow, isolated_config):
        reference = _corpus_reference(isolated_config)
        await _run_in_own_mlflow_run(_make_runner(isolated_config, transcriber=FakeTranscriber(reference)))
        before = isolated_config.gate_path.read_text(encoding="utf-8")

        # A comparison run for another provider: scored and returned, but never written to the gate.
        comparison = _make_runner(
            isolated_config,
            provider=TranscriptionProvider.OPENAI,
            transcriber=FakeTranscriber("something else entirely"),
        )
        gate = await _run_in_own_mlflow_run(comparison, update_gate=False)

        assert isolated_config.gate_path.read_text(encoding="utf-8") == before
        assert gate.transcription_metrics is not None
        assert gate.transcription_metrics["wer"].value > 0.0

    async def test_comparison_logs_the_persisted_overall_gate_verdict(
        self, local_mlflow, isolated_config, monkeypatch
    ):
        write_gate(
            GateResult(
                run_id="existing",
                identification_metrics={
                    "decision.precision": MetricEntry(value=0.0, gated=True, passed=False),
                },
            ),
            isolated_config.gate_path,
        )
        logged: dict = {}
        monkeypatch.setattr(
            "seshat.eval.transcription.runner.log_eval_run_metadata",
            lambda **kwargs: logged.update(kwargs),
        )
        reference = _corpus_reference(isolated_config)
        runner = _make_runner(isolated_config, transcriber=FakeTranscriber(reference))

        comparison = await _run_in_own_mlflow_run(runner, update_gate=False)

        assert comparison.harness_passed("transcription") is True
        assert logged["harness_passed"] is True
        assert logged["gate_passed"] is False

    async def test_comparison_without_a_persisted_gate_logs_false(
        self, local_mlflow, isolated_config, monkeypatch
    ):
        logged: dict = {}
        monkeypatch.setattr(
            "seshat.eval.transcription.runner.log_eval_run_metadata",
            lambda **kwargs: logged.update(kwargs),
        )
        reference = _corpus_reference(isolated_config)
        runner = _make_runner(isolated_config, transcriber=FakeTranscriber(reference))

        comparison = await _run_in_own_mlflow_run(runner, update_gate=False)

        assert not isolated_config.gate_path.exists()
        assert comparison.harness_passed("transcription") is True
        assert logged["harness_passed"] is True
        assert logged["gate_passed"] is False

    async def test_run_without_gate_update_and_empty_filter_does_not_create_gate(self, isolated_config):
        runner = _make_runner(isolated_config)

        gate = await runner.run(tag_filter={"language": "__nonexistent__"}, update_gate=False)

        assert not isolated_config.gate_path.exists()
        assert gate.run_id == "transcription-no-corpus"
