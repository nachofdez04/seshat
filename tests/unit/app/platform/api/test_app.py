from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from seshat.app.platform.api.app import (
    _check_eval_gate,
    _emit_config_warnings,
    _ping_embedding_providers,
    _ping_external_model_providers,
    _ping_reranking_providers,
    _ping_transcription_providers,
)


class TestCheckEvalGate:
    def test_skip_eval_gate_returns_immediately(self, caplog):
        config = MagicMock()
        config.skip_eval_gate = True
        _check_eval_gate(config)  # must not raise

    def test_missing_gate_file_raises_system_exit(self, tmp_path):
        config = MagicMock()
        config.skip_eval_gate = False
        config.eval_gate_path = tmp_path / "nonexistent.json"
        with pytest.raises(SystemExit):
            _check_eval_gate(config)

    def test_gate_not_passed_raises_system_exit(self, tmp_path):
        gate_path = tmp_path / "eval_gate.json"
        gate_path.write_text(json.dumps({"passed": False}))
        config = MagicMock()
        config.skip_eval_gate = False
        config.eval_gate_path = gate_path
        with pytest.raises(SystemExit):
            _check_eval_gate(config)

    def test_gate_passed_does_not_raise(self, tmp_path):
        gate_path = tmp_path / "eval_gate.json"
        gate_path.write_text(json.dumps({"passed": True}))
        config = MagicMock()
        config.skip_eval_gate = False
        config.eval_gate_path = gate_path
        _check_eval_gate(config)  # must not raise


class TestEmitConfigWarnings:
    def test_no_grounding_emits_warning(self, caplog):
        import logging

        config = MagicMock()
        config.extraction.grounding = None
        with caplog.at_level(logging.WARNING):
            _emit_config_warnings(config)
        assert any("grounding=None" in r.message for r in caplog.records)

    def test_with_grounding_no_warning(self, caplog):
        import logging

        config = MagicMock()
        config.extraction.grounding = MagicMock()
        with caplog.at_level(logging.WARNING):
            _emit_config_warnings(config)
        assert not any("grounding" in r.message for r in caplog.records)


class TestPingExternalModelProviders:
    async def test_skip_external_provider_ping_returns_immediately(self):
        config = MagicMock()
        config.api.skip_external_provider_ping = True
        await _ping_external_model_providers(config)  # must not raise

    async def test_all_providers_reachable_does_not_raise(self):
        config = MagicMock()
        config.api.skip_external_provider_ping = False
        with (
            patch("seshat.app.platform.api.app._ping_llm_providers", new=AsyncMock(return_value=[])),
            patch("seshat.app.platform.api.app._ping_embedding_providers", new=AsyncMock(return_value=[])),
            patch("seshat.app.platform.api.app._ping_transcription_providers", new=AsyncMock(return_value=[])),
            patch("seshat.app.platform.api.app._ping_reranking_providers", new=AsyncMock(return_value=[])),
        ):
            await _ping_external_model_providers(config)  # must not raise

    async def test_faulty_provider_raises_system_exit(self):
        config = MagicMock()
        config.api.skip_external_provider_ping = False
        with (
            patch("seshat.app.platform.api.app._ping_llm_providers", new=AsyncMock(return_value=["anthropic"])),
            patch("seshat.app.platform.api.app._ping_embedding_providers", new=AsyncMock(return_value=[])),
            patch("seshat.app.platform.api.app._ping_transcription_providers", new=AsyncMock(return_value=[])),
            patch("seshat.app.platform.api.app._ping_reranking_providers", new=AsyncMock(return_value=[])),
            pytest.raises(SystemExit),
        ):
            await _ping_external_model_providers(config)


class TestPingEmbeddingProviders:
    async def test_reachable_returns_empty_list(self):
        config = MagicMock()
        config.vector_index.embedding_provider = "openai"
        with patch("seshat.app.platform.api.app._build_embeddings") as mock_build:
            mock_build.return_value.aembed_query = AsyncMock(return_value=[0.1])
            result = await _ping_embedding_providers(config)
        assert result == []

    async def test_unreachable_returns_provider_name(self):
        config = MagicMock()
        config.vector_index.embedding_provider = "openai"
        with patch("seshat.app.platform.api.app._build_embeddings") as mock_build:
            mock_build.return_value.aembed_query = AsyncMock(side_effect=RuntimeError("boom"))
            result = await _ping_embedding_providers(config)
        assert result == ["openai"]


class TestPingTranscriptionProviders:
    async def test_reachable_returns_empty_list(self):
        config = MagicMock()
        config.transcription.provider = "assemblyai"
        with patch("seshat.app.platform.api.app.get_transcriber") as mock_get:
            mock_get.return_value.ping = AsyncMock()
            result = await _ping_transcription_providers(config)
        assert result == []

    async def test_unreachable_returns_provider_name(self):
        config = MagicMock()
        config.transcription.provider = "assemblyai"
        with patch("seshat.app.platform.api.app.get_transcriber") as mock_get:
            mock_get.return_value.ping = AsyncMock(side_effect=RuntimeError("boom"))
            result = await _ping_transcription_providers(config)
        assert result == ["assemblyai"]


class TestPingRerankingProviders:
    async def test_no_reranker_configured_returns_empty_list(self):
        config = MagicMock()
        config.rag.reranker = None
        with patch("seshat.app.platform.api.app._get_reranker", return_value=None):
            result = await _ping_reranking_providers(config)
        assert result == []

    async def test_reachable_returns_empty_list(self):
        config = MagicMock()
        config.rag.reranker.provider = "cohere"
        with patch("seshat.app.platform.api.app._get_reranker") as mock_get_reranker:
            mock_get_reranker.return_value.ping = AsyncMock()
            result = await _ping_reranking_providers(config)
        assert result == []

    async def test_unreachable_returns_provider_name(self):
        config = MagicMock()
        config.rag.reranker.provider = "cohere"
        with patch("seshat.app.platform.api.app._get_reranker") as mock_get_reranker:
            mock_get_reranker.return_value.ping = AsyncMock(side_effect=RuntimeError("boom"))
            result = await _ping_reranking_providers(config)
        assert result == ["cohere"]
