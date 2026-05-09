import pytest


class TestEnvSecretsResolver:
    def test_get_existing_secret(self, monkeypatch, env_secrets_resolver):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-abc")
        assert env_secrets_resolver.get_secret("ANTHROPIC_API_KEY") == "test-key-abc"

    def test_get_missing_secret_raises(self, monkeypatch, env_secrets_resolver):
        monkeypatch.delenv("NONEXISTENT_KEY", raising=False)
        with pytest.raises(KeyError, match="NONEXISTENT_KEY"):
            env_secrets_resolver.get_secret("NONEXISTENT_KEY")

    def test_empty_secret_raises(self, monkeypatch, env_secrets_resolver):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "")
        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
            env_secrets_resolver.get_secret("ANTHROPIC_API_KEY")
