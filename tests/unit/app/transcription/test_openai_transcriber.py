from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from seshat.app.transcription.openai_transcriber import OpenAITranscriber
from seshat.core.config.settings import TranscriptionConfig


class TestOpenAITranscriberPing:
    async def test_reachable_does_not_raise(self):
        with patch("seshat.app.transcription.openai_transcriber.AsyncOpenAI") as mock_client_cls:
            mock_client_cls.return_value.models.list = AsyncMock()
            transcriber = OpenAITranscriber(TranscriptionConfig(), api_key="key")
            await transcriber.ping()  # must not raise
        mock_client_cls.return_value.models.list.assert_awaited_once()

    async def test_unreachable_raises(self):
        with patch("seshat.app.transcription.openai_transcriber.AsyncOpenAI") as mock_client_cls:
            mock_client_cls.return_value.models.list = AsyncMock(side_effect=RuntimeError("unreachable"))
            transcriber = OpenAITranscriber(TranscriptionConfig(), api_key="key")
            with pytest.raises(RuntimeError, match="unreachable"):
                await transcriber.ping()
