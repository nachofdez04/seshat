from __future__ import annotations

from abc import ABC, abstractmethod


class AbstractTranscriber(ABC):
    @abstractmethod
    async def transcribe(self, audio_bytes: bytes, extension: str) -> str:
        """Transcribe raw audio bytes and return plain text."""

    @abstractmethod
    async def ping(self) -> None:
        """Verify connectivity to the transcription provider. Raises on failure."""
