from __future__ import annotations

import os

import pytest

from seshat.app.transcription.assemblyai_transcriber import AssemblyAITranscriber
from seshat.app.transcription.openai_transcriber import OpenAITranscriber
from seshat.core.config.settings import TranscriptionConfig
from seshat.core.models.enums import TranscriptionProvider
from tests.integration.conftest import SKIP_IF_NO_ASSEMBLYAI_API, SKIP_IF_NO_OPENAI_TRANSCRIPTION_API

pytestmark = [pytest.mark.integration, pytest.mark.llm, pytest.mark.transcription]


class TestAssemblyAITranscriber:
    @SKIP_IF_NO_ASSEMBLYAI_API
    async def test_transcribe_returns_string(self, short_audio_bytes: bytes):
        config = TranscriptionConfig(provider=TranscriptionProvider.ASSEMBLYAI)
        service = AssemblyAITranscriber(config, api_key=os.environ["ASSEMBLYAI_API_KEY"])
        result = await service.transcribe(audio_bytes=short_audio_bytes, extension=".mp3")
        assert isinstance(result, str)
        assert len(result) > 0

    @SKIP_IF_NO_ASSEMBLYAI_API
    async def test_ping_does_not_raise(self):
        config = TranscriptionConfig(provider=TranscriptionProvider.ASSEMBLYAI)
        service = AssemblyAITranscriber(config, api_key=os.environ["ASSEMBLYAI_API_KEY"])
        await service.ping()


class TestOpenAITranscriber:
    @SKIP_IF_NO_OPENAI_TRANSCRIPTION_API
    async def test_transcribe_returns_string(self, short_audio_bytes: bytes):
        config = TranscriptionConfig(provider=TranscriptionProvider.OPENAI)
        service = OpenAITranscriber(config, api_key=os.environ["OPENAI_API_KEY"])
        result = await service.transcribe(audio_bytes=short_audio_bytes, extension=".mp3")
        assert isinstance(result, str)
        assert len(result) > 0

    @SKIP_IF_NO_OPENAI_TRANSCRIPTION_API
    async def test_ping_does_not_raise(self):
        config = TranscriptionConfig(provider=TranscriptionProvider.OPENAI)
        service = OpenAITranscriber(config, api_key=os.environ["OPENAI_API_KEY"])
        await service.ping()
