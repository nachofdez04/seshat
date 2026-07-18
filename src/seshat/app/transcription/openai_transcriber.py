from __future__ import annotations

from typing import TYPE_CHECKING

from openai import AsyncOpenAI

from seshat.app.transcription.base import AbstractTranscriber
from seshat.core.utils.log import get_logger

if TYPE_CHECKING:
    from seshat.core.config.settings import TranscriptionConfig


_DEFAULT_TRANSCRIPTION_MODEL = "whisper-1"


logger = get_logger(__name__)


class OpenAITranscriber(AbstractTranscriber):
    def __init__(self, config: TranscriptionConfig, api_key: str) -> None:
        self._client = AsyncOpenAI(
            api_key=api_key,
            max_retries=config.max_retries,
            timeout=config.timeout_seconds,
        )
        self._config = config

    async def transcribe(self, audio_bytes: bytes, extension: str) -> str:
        extension = extension.lstrip(".")
        model_name = self._config.model or _DEFAULT_TRANSCRIPTION_MODEL
        logger.debug("Transcribing %s audio (%d bytes) using model %s", extension, len(audio_bytes), model_name)

        response = await self._client.audio.transcriptions.create(
            model=model_name,
            file=(f"audio.{extension}", audio_bytes),
            language=self._config.language,
            response_format="verbose_json",
        )

        segments = response.segments or []
        return "\n".join(seg.text.strip() for seg in segments)

    async def ping(self) -> None:
        await self._client.models.list()
