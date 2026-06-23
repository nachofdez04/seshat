from unittest.mock import MagicMock, patch

import pytest

from seshat.config.settings import ExtractionConfig, GroundingLLMConfig, IdentificationLLMConfig, SeshatConfig
from seshat.models.enums import LLMProvider
from seshat.pipeline.llm_factory import _build_llm


@pytest.mark.usefixtures("mocked_secrets_resolver")
class TestBuildLlm:
    def test_calls_init_chat_model_with_provider_and_model(self, minimal_config):
        llm_cfg = IdentificationLLMConfig(provider=LLMProvider.ANTHROPIC, model="claude-haiku-4-5-20251001")
        stub = MagicMock()

        with patch("seshat.pipeline.llm_factory.init_chat_model", return_value=stub) as mock_init:
            result = _build_llm(llm_cfg, minimal_config)

        mock_init.assert_called_once()
        call_kwargs = mock_init.call_args
        assert call_kwargs.args[0] == "claude-haiku-4-5-20251001"
        assert call_kwargs.kwargs["model_provider"] == LLMProvider.ANTHROPIC
        assert result is stub

    def test_uses_api_key_secret_key(self, minimal_config, mocked_secrets_resolver):
        llm_cfg = IdentificationLLMConfig(provider=LLMProvider.ANTHROPIC, api_key_secret_key="my_custom_key")

        with patch("seshat.pipeline.llm_factory.init_chat_model"):
            _build_llm(llm_cfg, minimal_config)

        mocked_secrets_resolver.get_secret.assert_called_once_with("my_custom_key")

    def test_default_api_key_secret_key_derived_from_provider(self, minimal_config, mocked_secrets_resolver):
        llm_cfg = IdentificationLLMConfig(provider=LLMProvider.OPENAI, model="gpt-4o-mini")

        with patch("seshat.pipeline.llm_factory.init_chat_model"):
            _build_llm(llm_cfg, minimal_config)

        mocked_secrets_resolver.get_secret.assert_called_once_with("openai_api_key")

    def test_bedrock_converse_skips_api_key(self, minimal_config, mocked_secrets_resolver):
        llm_cfg = IdentificationLLMConfig(provider=LLMProvider.BEDROCK_CONVERSE, model="anthropic.claude-sonnet-4-5")

        with patch("seshat.pipeline.llm_factory.init_chat_model") as mock_init:
            _build_llm(llm_cfg, minimal_config)

        mocked_secrets_resolver.get_secret.assert_not_called()
        call_kwargs = mock_init.call_args.kwargs
        assert "api_key" not in call_kwargs

    def test_anthropic_provider_sends_prompt_caching_header(self, minimal_config):
        llm_cfg = IdentificationLLMConfig(provider=LLMProvider.ANTHROPIC, model="claude-haiku-4-5-20251001")

        with patch("seshat.pipeline.llm_factory.init_chat_model") as mock_init:
            _build_llm(llm_cfg, minimal_config)

        call_kwargs = mock_init.call_args.kwargs
        assert call_kwargs["model_kwargs"]["extra_headers"]["anthropic-beta"] == "prompt-caching-2024-07-31"

    def test_bedrock_converse_does_not_send_prompt_caching_header(self, minimal_config):
        llm_cfg = IdentificationLLMConfig(provider=LLMProvider.BEDROCK_CONVERSE, model="anthropic.claude-sonnet-4-5")

        with patch("seshat.pipeline.llm_factory.init_chat_model") as mock_init:
            _build_llm(llm_cfg, minimal_config)

        call_kwargs = mock_init.call_args.kwargs
        assert "model_kwargs" not in call_kwargs or "extra_headers" not in call_kwargs.get("model_kwargs", {})

    def test_azure_openai_does_not_send_prompt_caching_header(self, minimal_config):
        llm_cfg = IdentificationLLMConfig(provider=LLMProvider.AZURE_OPENAI, model="gpt-4o")

        with patch("seshat.pipeline.llm_factory.init_chat_model") as mock_init:
            _build_llm(llm_cfg, minimal_config)

        call_kwargs = mock_init.call_args.kwargs
        assert "model_kwargs" not in call_kwargs

    def test_get_grounding_llm_raises_value_error_when_not_configured(self):
        from seshat.pipeline.llm_factory import get_grounding_llm

        cfg = SeshatConfig(
            _env_file=None,  # type: ignore[call-arg]
            extraction=ExtractionConfig(
                identification=IdentificationLLMConfig(provider=LLMProvider.ANTHROPIC),
                grounding=None,
            ),
        )

        with pytest.raises(ValueError, match="grounding is not configured"):
            get_grounding_llm(cfg)

    def test_grounding_llm_uses_its_own_secret_key(self, mocked_secrets_resolver):
        from seshat.pipeline.llm_factory import get_grounding_llm

        grd_cfg = GroundingLLMConfig(provider=LLMProvider.OPENAI, api_key_secret_key="openai_grounding_key")
        cfg = SeshatConfig(
            _env_file=None,  # type: ignore[call-arg]
            extraction=ExtractionConfig(
                identification=IdentificationLLMConfig(provider=LLMProvider.ANTHROPIC),
                grounding=grd_cfg,
            ),
        )

        with patch("seshat.pipeline.llm_factory.init_chat_model"):
            get_grounding_llm(cfg)

        mocked_secrets_resolver.get_secret.assert_called_once_with("openai_grounding_key")
