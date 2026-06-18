from __future__ import annotations

from typing import TYPE_CHECKING

from seshat.models.enums import TranscriptionProvider
from seshat.observability.usage_tracker import TrackingTranscriber
from seshat.secrets.factory import get_secrets_resolver
from seshat.utils.log import get_logger

if TYPE_CHECKING:
    from seshat.config.settings import SeshatConfig
    from seshat.transcription.base import AbstractTranscriber

logger = get_logger(__name__)


def get_transcriber(
    config: SeshatConfig,
) -> AbstractTranscriber:
    secrets = get_secrets_resolver(config)
    api_key = secrets.get_secret(config.transcription.api_key_secret_key)  # type: ignore[arg-type]

    logger.debug("Initialising transcriber: %s", config.transcription.provider)

    raw: AbstractTranscriber
    match config.transcription.provider:
        case TranscriptionProvider.ASSEMBLYAI:
            from seshat.transcription.assemblyai_transcriber import AssemblyAITranscriber

            raw = AssemblyAITranscriber(config.transcription, api_key)
        case TranscriptionProvider.OPENAI:
            from seshat.transcription.openai_transcriber import OpenAITranscriber

            raw = OpenAITranscriber(config.transcription, api_key)
        case _:
            raise ValueError(f"Unsupported transcription provider: {config.transcription.provider}")

    return TrackingTranscriber(raw)
