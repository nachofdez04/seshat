import pytest
from pydantic import ValidationError

from seshat.config.settings import (
    ConfidenceWeights,
    ExtractionConfig,
    LLMConfig,
    SecretsConfig,
    SeshatConfig,
    SeshatConfigOverride,
    TranscriptionConfig,
    VerificationConfig,
    get_request_settings,
)
from seshat.models.enums import LLMProvider, SecretsProvider


class TestConfidenceWeightsRedistribute:
    def test_logprobs_disabled_remaining_sum_to_one(self):
        weights = ConfidenceWeights()
        result = weights.redistribute({"logprobs"})
        total = result.logprobs + result.verification + result.heuristics
        assert abs(total - 1.0) < 1e-9
        assert result.logprobs == 0.0

    def test_verification_disabled_remaining_sum_to_one(self):
        weights = ConfidenceWeights()
        result = weights.redistribute({"verification"})
        total = result.logprobs + result.verification + result.heuristics
        assert abs(total - 1.0) < 1e-9
        assert result.verification == 0.0

    def test_both_optional_signals_disabled_heuristics_gets_full_weight(self):
        weights = ConfidenceWeights()
        result = weights.redistribute({"logprobs", "verification"})
        assert result.heuristics == pytest.approx(1.0)
        assert result.logprobs == 0.0
        assert result.verification == 0.0

    def test_heuristics_disabled_raises(self):
        with pytest.raises(ValueError, match="non-disableable"):
            ConfidenceWeights().redistribute({"heuristics"})

    def test_unknown_signal_raises(self):
        with pytest.raises(ValueError, match="non-disableable"):
            ConfidenceWeights().redistribute({"typo_signal"})


class TestVerificationModelValidator:
    def test_same_provider_raises(self):
        with pytest.raises(ValidationError, match=r"verification.provider"):
            ExtractionConfig(
                llm=LLMConfig(provider=LLMProvider.ANTHROPIC),
                verification=VerificationConfig(
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
        assert result.extraction.llm.provider == minimal_config.extraction.llm.provider
        assert result.extraction.llm.model == minimal_config.extraction.llm.model

    def test_partial_override_preserves_non_default_base_values(self, monkeypatch):
        """Overriding one extraction field must not revert sibling fields to their
        defaults when the base config carries non-default values for those fields."""
        monkeypatch.setenv("postgres_url", "postgresql://seshat:seshat@localhost:5432/seshat")
        # Base has a non-default max_chunk_count (10 vs default 50).
        base = SeshatConfig(
            _env_file=None,  # type: ignore[call-arg]
            secrets=SecretsConfig(provider=SecretsProvider.ENV),
            transcription=TranscriptionConfig(max_retries=1),
            extraction=ExtractionConfig(max_chunk_count=10),
            max_jobs_per_user_per_hour=5,
        )
        monkeypatch.setattr("seshat.config.settings._config", base)

        # Override only confidence_threshold, the other fields should survive unchanged:
        # - transcription.provider (another config class)
        # - extraction.max_chunk_count (same config class)
        # - max_jobs_per_user_per_hour (top level field)
        result = get_request_settings(SeshatConfigOverride(extraction=ExtractionConfig(confidence_threshold=0.5)))

        # should be overridden
        assert result.extraction.confidence_threshold == 0.5
        # must not revert to default
        assert result.transcription.max_retries == 1
        assert result.extraction.max_chunk_count == 10
        assert result.max_jobs_per_user_per_hour == 5
