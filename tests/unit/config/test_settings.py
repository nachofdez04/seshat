import pytest
from pydantic import ValidationError

from seshat.config.settings import (
    ExtractionConfig,
    GroundingLLMConfig,
    IdentificationLLMConfig,
    ReflectiveLLMConfig,
    SecretsConfig,
    SeshatConfig,
    SeshatConfigOverride,
    TranscriptionConfig,
    get_request_settings,
)
from seshat.models.enums import LLMProvider, SecretsProvider


class TestReflectiveLLMConfig:
    def test_defaults_to_disabled(self):
        cfg = ReflectiveLLMConfig()
        assert cfg.enabled is False
        assert cfg.llm is None

    def test_enabled_with_defaults(self):
        cfg = ReflectiveLLMConfig(enabled=True)
        assert cfg.enabled is True

    def test_llm_override_accepted(self):
        llm = IdentificationLLMConfig()
        cfg = ReflectiveLLMConfig(enabled=True, llm=llm)
        assert cfg.llm is llm


class TestGroundingModelValidator:
    def test_same_provider_raises(self):
        with pytest.raises(ValidationError, match=r"grounding.provider"):
            ExtractionConfig(
                identification=IdentificationLLMConfig(provider=LLMProvider.ANTHROPIC),
                grounding=GroundingLLMConfig(
                    provider=LLMProvider.ANTHROPIC,
                    model="claude-haiku-4-5-20251001",
                ),
            )


class TestGetRequestSettings:
    def test_none_override_returns_singleton(self, monkeypatch, minimal_config: SeshatConfig):
        monkeypatch.setattr("seshat.config.settings._config", minimal_config)
        assert get_request_settings(None) is minimal_config

    def test_override_applies_and_preserves_unset_fields(self, monkeypatch, minimal_config: SeshatConfig):
        monkeypatch.setattr("seshat.config.settings._config", minimal_config)
        result = get_request_settings(SeshatConfigOverride(extraction=ExtractionConfig(confidence_threshold=0.5)))
        assert result.extraction.confidence_threshold == 0.5
        assert result.extraction.identification.provider == minimal_config.extraction.identification.provider
        assert result.extraction.identification.model == minimal_config.extraction.identification.model

    def test_partial_override_preserves_non_default_base_values(self, monkeypatch):
        """Overriding one extraction field must not revert sibling fields to their
        defaults when the base config carries non-default values for those fields."""
        monkeypatch.setenv("postgres_url", "postgresql://seshat:seshat@localhost:5432/seshat")
        # Base has a non-default max_output_tokens on the identification LLM config.
        base = SeshatConfig(
            _env_file=None,  # type: ignore[call-arg]
            secrets=SecretsConfig(provider=SecretsProvider.ENV),
            transcription=TranscriptionConfig(max_retries=1),
            extraction=ExtractionConfig(identification=IdentificationLLMConfig(max_output_tokens=1024)),
            max_jobs_per_user_per_hour=5,
        )
        monkeypatch.setattr("seshat.config.settings._config", base)

        # Override only confidence_threshold, the other fields should survive unchanged:
        # - transcription.max_retries (another config class)
        # - extraction.identification.max_output_tokens (same config class)
        # - max_jobs_per_user_per_hour (top level field)
        result = get_request_settings(SeshatConfigOverride(extraction=ExtractionConfig(confidence_threshold=0.5)))

        # should be overridden
        assert result.extraction.confidence_threshold == 0.5
        # must not revert to default
        assert result.transcription.max_retries == 1
        assert result.extraction.identification.max_output_tokens == 1024
        assert result.max_jobs_per_user_per_hour == 5
