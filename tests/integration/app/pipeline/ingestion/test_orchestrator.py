from __future__ import annotations

from datetime import date
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml

from seshat.app.pipeline.ingestion.audio_validator import (
    AudioValidationError,
    FileTooLargeError,
    UnsupportedFormatError,
)
from seshat.app.pipeline.ingestion.orchestrator import IngestionOrchestrator
from seshat.app.pipeline.ingestion.text_validator import TextValidationError
from seshat.core.config.settings import TranscriptionConfig
from seshat.core.models.transcript import TranscriptMetadata
from tests.integration.conftest import SKIP_IF_NO_LOCALSTACK

pytestmark = [pytest.mark.integration, SKIP_IF_NO_LOCALSTACK]


@pytest.fixture
def mock_transcriber():
    transcriber = MagicMock()
    transcriber.transcribe = AsyncMock(return_value="We decided to use PostgreSQL.")
    return transcriber


def _build_orchestrator(mock_transcriber, blob_store, transcription_config=None):
    return IngestionOrchestrator(mock_transcriber, blob_store, transcription_config or TranscriptionConfig())


@pytest.fixture
def orchestrator(mock_transcriber, blob_store):
    return _build_orchestrator(mock_transcriber, blob_store)


class TestIngestionOrchestratorAudio:
    async def test_ingest_valid_mp3(self, orchestrator, short_audio_bytes):
        metadata = TranscriptMetadata(meeting_date=date(2026, 4, 21))
        doc = await orchestrator.ingest_audio(short_audio_bytes, date(2026, 4, 21), "job-audio-1", metadata)
        assert doc.source_type == "audio"
        assert doc.blob_key == "jobs/2026-04-21/job-audio-1/raw/transcript.txt"

    async def test_ingest_with_matching_filename(self, orchestrator, short_audio_bytes):
        metadata = TranscriptMetadata(meeting_date=date(2026, 4, 21))
        doc = await orchestrator.ingest_audio(
            short_audio_bytes, date(2026, 4, 21), "job-audio-2", metadata, filename="recording.mp3"
        )
        assert doc.blob_key == "jobs/2026-04-21/job-audio-2/raw/transcript.txt"

    async def test_ingest_mismatched_filename_raises(self, orchestrator, short_audio_bytes):
        metadata = TranscriptMetadata(meeting_date=date(2026, 4, 21))
        with pytest.raises(AudioValidationError, match="mismatch"):
            await orchestrator.ingest_audio(
                short_audio_bytes, date(2026, 4, 21), "job-mismatch", metadata, filename="recording.wav"
            )


class TestIngestionOrchestratorText:
    async def test_ingest_valid_yaml(self, orchestrator):
        raw = yaml.dump(
            {
                "date": "2026-04-21",
                "content": "We decided to use PostgreSQL.",
                "participants": ["Alice"],
            }
        ).encode()
        doc = await orchestrator.ingest_text(raw, date(2026, 4, 21), "job-text-1", "meeting.yaml")
        assert doc.source_type == "text"
        assert doc.blob_key == "jobs/2026-04-21/job-text-1/raw/transcript.txt"
        assert doc.metadata.participants == ["Alice"]

    async def test_ingest_text_meeting_date_mismatch_raises(self, orchestrator):
        raw = yaml.dump({"date": "2026-04-21", "content": "Notes."}).encode()
        with pytest.raises(TextValidationError, match="mismatch"):
            await orchestrator.ingest_text(raw, date(2026, 1, 1), "job-text-2", "meeting.yaml")


class TestIngestionOrchestratorValidate:
    # async def (with no internal await): sync test defs bypass pytest-asyncio's
    # loop-factory hook entirely, so their async fixtures fall back to the platform
    # default loop (ProactorEventLoop on Windows), which psycopg async rejects.
    async def test_valid_audio_passes(self, orchestrator, short_audio_bytes):
        orchestrator.validate(short_audio_bytes, "audio", date(2026, 4, 21), "recording.mp3")

    async def test_oversized_audio_raises_file_too_large(self, mock_transcriber, blob_store, short_audio_bytes):
        orchestrator = _build_orchestrator(mock_transcriber, blob_store, TranscriptionConfig(max_file_bytes=10))
        with pytest.raises(FileTooLargeError, match="exceeds maximum"):
            orchestrator.validate(short_audio_bytes, "audio", date(2026, 4, 21), "recording.mp3")

    async def test_mismatched_extension_raises_unsupported_format(self, orchestrator, short_audio_bytes):
        with pytest.raises(UnsupportedFormatError, match="mismatch"):
            orchestrator.validate(short_audio_bytes, "audio", date(2026, 4, 21), "recording.wav")

    async def test_valid_text_passes(self, orchestrator):
        raw = yaml.dump({"date": "2026-04-21", "content": "Notes."}).encode()
        orchestrator.validate(raw, "text", date(2026, 4, 21), "meeting.yaml")

    async def test_text_meeting_date_mismatch_raises(self, orchestrator):
        raw = yaml.dump({"date": "2026-04-21", "content": "Notes."}).encode()
        with pytest.raises(TextValidationError, match="mismatch"):
            orchestrator.validate(raw, "text", date(2026, 1, 1), "meeting.yaml")
