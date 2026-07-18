from __future__ import annotations

import io
from typing import TYPE_CHECKING, Any

import assemblyai as aai

from seshat.app.transcription.base import AbstractTranscriber
from seshat.core.utils.concurrency import run_in_thread
from seshat.core.utils.log import get_logger

if TYPE_CHECKING:
    from seshat.core.config.settings import TranscriptionConfig

logger = get_logger(__name__)


class AssemblyAITranscriber(AbstractTranscriber):
    def __init__(self, config: TranscriptionConfig, api_key: str) -> None:
        client_kwargs: dict[str, Any] = {"api_key": api_key}
        if config.timeout_seconds is not None:
            client_kwargs["http_timeout"] = config.timeout_seconds
        transcription_client_settings = aai.Settings(**client_kwargs)

        self._client = aai.Client(settings=transcription_client_settings)
        self._transcriber = aai.Transcriber(
            client=self._client,
            config=aai.TranscriptionConfig(language_code=config.language, speaker_labels=True),
        )
        self._config = config

    async def transcribe(self, audio_bytes: bytes, extension: str) -> str:
        extension = extension.lstrip(".")
        logger.debug("Transcribing %s audio (%d bytes) via AssemblyAI", extension, len(audio_bytes))
        transcript = await run_in_thread(self._transcriber.transcribe, io.BytesIO(audio_bytes))
        if transcript.status == aai.TranscriptStatus.error:
            raise RuntimeError(f"AssemblyAI transcription failed: {transcript.error}")

        utterances = transcript.utterances or []
        return "\n".join(utt.text for utt in utterances)

    async def ping(self) -> None:
        response = await run_in_thread(self._client.http_client.get, "/v2")
        response.raise_for_status()
