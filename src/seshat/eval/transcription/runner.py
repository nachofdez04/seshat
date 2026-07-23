from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import mlflow
import mlflow.genai
import pandas as pd

from seshat.app.platform.observability.usage_tracker import track_eval_usage
from seshat.core.utils.hashing import fingerprint
from seshat.core.utils.log import set_task_num
from seshat.eval.cache import build_cache_fp, read_or_run, sweep_stale_entries
from seshat.eval.gate import read_gate, transcription_entries, upsert_gate
from seshat.eval.mlflow_logging import log_eval_run_metadata
from seshat.eval.models import GateResult, TranscriptionPrediction
from seshat.eval.transcription.corpus_loader import load_corpus
from seshat.eval.transcription.scorers import pooled_wer, scorer, word_error_rate

if TYPE_CHECKING:
    from pathlib import Path

    from mlflow.genai.evaluation.entities import EvaluationResult

    from seshat.app.transcription.base import AbstractTranscriber
    from seshat.core.config.eval_settings import EvalConfig
    from seshat.core.config.settings import TranscriptionConfig
    from seshat.eval.corpus_tags import CorpusTagFilter
    from seshat.eval.models import TranscriptionCorpusExample


class TranscriptionEvalRunner:
    """Eval runner for the transcription pass — WER of a provider against reference transcripts."""

    def __init__(
        self,
        transcriber: AbstractTranscriber,
        transcription_config: TranscriptionConfig,
        config: EvalConfig,
    ) -> None:
        self._transcriber = transcriber
        self._transcription_config = transcription_config
        self._config = config
        # Provider settings that change the hypothesis. The audio bytes are covered separately,
        # via the corpus example's audio_sha256 (part of build_cache_fp's content fingerprint).
        self._provider_hash = fingerprint(
            "|".join(
                [
                    str(transcription_config.provider),
                    transcription_config.model or "default",
                    transcription_config.language,
                ]
            )
        )

    async def run(
        self,
        tag_filter: CorpusTagFilter | None = None,
        model_id: str | None = None,
        update_gate: bool = True,
    ) -> GateResult:
        """Transcribe the corpus and score it.

        `update_gate=False` is the provider-comparison path: metrics are logged to MLflow but
        the shared gate file is left untouched, so a comparison run cannot silently own the gate.
        """
        examples = load_corpus(self._config.transcription_corpus_dir, tag_filter=tag_filter)
        if not examples:
            if update_gate:
                return upsert_gate(self._config.gate_path, run_id="transcription-no-corpus")

            return GateResult(run_id="transcription-no-corpus")

        result_cache, touched, cache_hits = await self._run_all_predictions(examples)

        def _predict(corpus_id: str, _audio_file: str) -> dict:
            if corpus_id not in result_cache:
                raise KeyError(f"corpus_id {corpus_id!r} not found in result cache — mlflow unpacking mismatch")

            return {"hypothesis": result_cache[corpus_id].text}

        df = _build_dataframe(examples)
        eval_result = mlflow.genai.evaluate(data=df, predict_fn=_predict, scorers=[scorer], model_id=model_id)

        run_id = eval_result.run_id
        transcription_metrics = _aggregate_metrics(examples, result_cache, eval_result)

        if update_gate:
            gate = upsert_gate(self._config.gate_path, run_id=run_id, transcription_metrics=transcription_metrics)
            gate_passed = gate.passed
        else:
            gate = GateResult(run_id=run_id, transcription_metrics=transcription_entries(transcription_metrics))
            gate_passed = read_gate(self._config.gate_path).passed if self._config.gate_path.exists() else False

        log_eval_run_metadata(
            run_id=run_id,
            harness="transcription",
            gate_passed=gate_passed,
            harness_passed=gate.harness_passed("transcription"),
            corpus_dir=self._config.transcription_corpus_dir,
            corpus_examples=examples,
            breakdown_artifact=_build_breakdown(examples, result_cache),
            tag_filter=tag_filter,
            cache_hits=cache_hits,
            total_predictions=len(examples),
            extra_params={
                # These three are what make a cross-provider comparison readable in the MLflow UI.
                "transcription.provider": str(self._transcription_config.provider),
                "transcription.model": self._transcription_config.model or "default",
                "transcription.language": self._transcription_config.language,
                "transcription.gate_updated": str(update_gate).lower(),
            },
        )

        sweep_stale_entries(
            self._config.transcription_cache_dir,
            corpus_ids=[ex.corpus_id for ex in examples],
            touched=touched,
            # Scoped to this provider so a comparison run does not evict the other providers'
            # hypotheses and force them to be re-purchased.
            agent_hash=self._provider_hash,
        )
        return gate

    @track_eval_usage("transcription")
    async def _run_all_predictions(
        self, examples: list[TranscriptionCorpusExample]
    ) -> tuple[dict[str, TranscriptionPrediction], set[Path], int]:
        sem = asyncio.Semaphore(self._config.max_concurrent_predictions)

        async def _run_one(
            task_idx: int, ex: TranscriptionCorpusExample
        ) -> tuple[str, TranscriptionPrediction, Path, bool]:
            set_task_num(task_idx)
            cache_fp = build_cache_fp(self._config.transcription_cache_dir, ex, agent_hash=self._provider_hash)

            async with sem:
                result, used, was_cached = await read_or_run(cache_fp, TranscriptionPrediction, self._transcribe(ex))
            return ex.corpus_id, result, used, was_cached

        quads = await asyncio.gather(*(_run_one(i, ex) for i, ex in enumerate(examples)))
        results = {corpus_id: result for corpus_id, result, _, _ in quads}
        touched = {used for _, _, used, _ in quads}
        cache_hits = sum(1 for _, _, _, was_cached in quads if was_cached)
        return results, touched, cache_hits

    async def _transcribe(self, example: TranscriptionCorpusExample) -> TranscriptionPrediction:
        audio_path = example.resolved_audio_path
        text = await self._transcriber.transcribe(audio_path.read_bytes(), extension=audio_path.suffix)
        return TranscriptionPrediction(text=text)


def _build_dataframe(examples: list[TranscriptionCorpusExample]) -> pd.DataFrame:
    rows = []
    for ex in examples:
        rows.append(
            {
                "inputs": {"corpus_id": ex.corpus_id, "_audio_file": str(ex.audio_file)},
                "expectations": {"reference": ex.reference},
                "tags": {f"corpus.{k}": str(v) for k, v in ex.tags.items()},
            }
        )
    return pd.DataFrame(rows)


def _aggregate_metrics(
    examples: list[TranscriptionCorpusExample],
    result_cache: dict[str, TranscriptionPrediction],
    eval_result: EvaluationResult,
) -> dict[str, float]:
    """Pooled (length-weighted) WER is the gated headline; the macro mean is informational.

    Pooled WER is not a mean of the per-row scores, so it is computed from the pairs directly
    rather than read back out of the MLflow aggregate.
    """
    pairs = [(ex.reference, result_cache[ex.corpus_id].text) for ex in examples if ex.corpus_id in result_cache]
    metrics = {"wer": pooled_wer(pairs)}

    macro = eval_result.metrics.get("transcription.wer/mean")
    if macro is not None:
        metrics["wer_macro"] = float(macro)

    return metrics


def _build_breakdown(
    examples: list[TranscriptionCorpusExample],
    result_cache: dict[str, TranscriptionPrediction],
) -> dict:
    breakdown: dict = {}
    for ex in examples:
        prediction = result_cache.get(ex.corpus_id)
        breakdown[ex.corpus_id] = {
            "tags": ex.tags,
            "audio_file": str(ex.audio_file),
            "wer": word_error_rate(ex.reference, prediction.text) if prediction else None,
            "reference": ex.reference,
            "hypothesis": prediction.text if prediction else None,
        }
    return breakdown
