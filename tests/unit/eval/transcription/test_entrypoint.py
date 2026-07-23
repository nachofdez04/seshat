from seshat.core.config.settings import SeshatConfig, TranscriptionConfig
from seshat.core.models.enums import TranscriptionProvider
from seshat.eval.transcription.entrypoint import _config_for_provider


def _config(**kwargs) -> SeshatConfig:
    return SeshatConfig(transcription=TranscriptionConfig(**kwargs))


class TestConfigForProvider:
    def test_default_provider_returns_the_config_untouched(self):
        config = _config(provider=TranscriptionProvider.ASSEMBLYAI, model="pinned-model")
        assert _config_for_provider(config, TranscriptionProvider.ASSEMBLYAI) is config

    def test_switching_provider_drops_the_pinned_model(self):
        config = _config(provider=TranscriptionProvider.ASSEMBLYAI, model="assemblyai-only-model")
        switched = _config_for_provider(config, TranscriptionProvider.OPENAI)

        assert switched.transcription.provider == TranscriptionProvider.OPENAI
        assert switched.transcription.model is None

    def test_switching_provider_re_resolves_the_api_key_secret(self):
        # model_copy would skip the validator and leave the old provider's secret key in place.
        config = _config(provider=TranscriptionProvider.ASSEMBLYAI)
        switched = _config_for_provider(config, TranscriptionProvider.OPENAI)
        assert switched.transcription.api_key_secret_key == "openai_api_key"

    def test_switching_provider_keeps_the_shared_settings(self):
        config = _config(provider=TranscriptionProvider.ASSEMBLYAI, language="es", max_retries=7)
        switched = _config_for_provider(config, TranscriptionProvider.OPENAI)

        assert switched.transcription.language == "es"
        assert switched.transcription.max_retries == 7

    def test_switching_provider_leaves_the_original_config_intact(self):
        config = _config(provider=TranscriptionProvider.ASSEMBLYAI)
        _config_for_provider(config, TranscriptionProvider.OPENAI)
        assert config.transcription.provider == TranscriptionProvider.ASSEMBLYAI
