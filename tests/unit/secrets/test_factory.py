from unittest.mock import patch

import pytest

from seshat.config.settings import SecretsConfig, SeshatConfig
from seshat.models.enums import SecretsProvider
from seshat.secrets.aws_resolver import AWSSecretsResolver
from seshat.secrets.env_resolver import EnvSecretsResolver
from seshat.secrets.factory import _cached_resolver, get_secrets_resolver


def _config_with(provider: SecretsProvider) -> SeshatConfig:
    return SeshatConfig(_env_file=None, secrets=SecretsConfig(provider=provider))  # type: ignore[call-arg]


class TestGetSecretsResolver:
    def test_env_provider_returns_env_resolver(self):
        resolver = get_secrets_resolver(_config_with(SecretsProvider.ENV))
        assert isinstance(resolver, EnvSecretsResolver)

    def test_aws_provider_returns_aws_resolver(self):
        with patch("boto3.client"):
            resolver = get_secrets_resolver(_config_with(SecretsProvider.AWS))
        assert isinstance(resolver, AWSSecretsResolver)

    def test_unknown_provider_raises(self):
        secrets_config = SecretsConfig.model_construct(provider="unknown")
        with pytest.raises(ValueError, match="unknown"):
            _cached_resolver(secrets_config)  # type: ignore[arg-type]
