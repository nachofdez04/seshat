from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from seshat.app.transcription.assemblyai_transcriber import AssemblyAITranscriber
from seshat.core.config.settings import TranscriptionConfig


class TestAssemblyAITranscriberPing:
    async def test_reachable_does_not_raise(self):
        with patch("assemblyai.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.http_client.get.return_value = MagicMock(raise_for_status=MagicMock())
            mock_client_cls.return_value = mock_client
            transcriber = AssemblyAITranscriber(TranscriptionConfig(), api_key="key")
            await transcriber.ping()  # must not raise
        mock_client.http_client.get.assert_called_once_with("/v2")

    async def test_unreachable_raises(self):
        with patch("assemblyai.Client") as mock_client_cls:
            mock_client = MagicMock()
            response = MagicMock()
            response.raise_for_status.side_effect = RuntimeError("unreachable")
            mock_client.http_client.get.return_value = response
            mock_client_cls.return_value = mock_client
            transcriber = AssemblyAITranscriber(TranscriptionConfig(), api_key="key")
            with pytest.raises(RuntimeError, match="unreachable"):
                await transcriber.ping()
