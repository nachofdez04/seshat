from __future__ import annotations

from typing import TYPE_CHECKING

import mlflow

from seshat.app.transcription.factory import get_transcriber
from seshat.core.config.settings import TranscriptionConfig
from seshat.core.utils.log import get_logger
from seshat.eval.transcription.runner import TranscriptionEvalRunner

if TYPE_CHECKING:
    from seshat.core.config.eval_settings import EvalConfig
    from seshat.core.config.settings import SeshatConfig
    from seshat.core.models.enums import TranscriptionProvider
    from seshat.eval.corpus_tags import CorpusTagFilter


logger = get_logger(__name__)


async def run(
    eval_config: EvalConfig,
    seshat_config: SeshatConfig,
    tag_filter: CorpusTagFilter | None = None,
    providers: list[TranscriptionProvider] | None = None,
) -> None:
    """Run the transcription WER harness, optionally across several providers for comparison.

    Without `providers`, the configured provider runs and owns the gate. With `providers`, each
    one runs in its own nested MLflow run and only the configured default updates the gate file —
    otherwise the last provider in the list would silently own it.
    """
    if not providers:
        await _run_one(eval_config, seshat_config, tag_filter, update_gate=True)
        return

    default_provider = seshat_config.transcription.provider
    for provider in providers:
        provider_config = _config_for_provider(seshat_config, provider)
        with mlflow.start_run(run_name=f"seshat-eval-transcription-{provider}", nested=True):
            await _run_one(eval_config, provider_config, tag_filter, update_gate=provider == default_provider)


async def _run_one(
    eval_config: EvalConfig,
    seshat_config: SeshatConfig,
    tag_filter: CorpusTagFilter | None,
    update_gate: bool,
) -> None:
    transcription_config = seshat_config.transcription
    logger.info(
        "Transcription provider=%r model=%r language=%r (gate update: %s)",
        str(transcription_config.provider),
        transcription_config.model or "default",
        transcription_config.language,
        update_gate,
    )

    runner = TranscriptionEvalRunner(
        transcriber=get_transcriber(seshat_config),
        transcription_config=transcription_config,
        config=eval_config,
    )
    gate = await runner.run(tag_filter=tag_filter, update_gate=update_gate)

    wer = gate.transcription_metrics["wer"].value if gate.transcription_metrics else None
    logger.info("transcription eval [%s]: wer=%s passed=%s", transcription_config.provider, wer, gate.passed)


def _config_for_provider(seshat_config: SeshatConfig, provider: TranscriptionProvider) -> SeshatConfig:
    """Rebuild the transcription config for `provider`, keeping the shared limits and language.

    `model` and `api_key_secret_key` are dropped rather than copied so the new provider's own
    defaults re-resolve — a rebuild (not model_copy) is required for that validator to fire.
    """
    if provider == seshat_config.transcription.provider:
        return seshat_config

    shared = seshat_config.transcription.model_dump(exclude={"provider", "model", "api_key_secret_key"})
    return seshat_config._with(transcription=TranscriptionConfig(provider=provider, **shared))
