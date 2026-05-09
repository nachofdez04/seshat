import pytest

from seshat.config.settings import SecretsConfig
from seshat.models.enums import SecretsProvider
from seshat.secrets.aws_resolver import AWSSecretsResolver
from seshat.secrets.env_resolver import EnvSecretsResolver


@pytest.fixture
def aws_secrets_config() -> SecretsConfig:
    return SecretsConfig(provider=SecretsProvider.AWS)


@pytest.fixture
def env_secrets_config() -> SecretsConfig:
    return SecretsConfig(provider=SecretsProvider.ENV)


@pytest.fixture
def aws_secrets_resolver(aws_secrets_config) -> AWSSecretsResolver:
    return AWSSecretsResolver(aws_secrets_config)


@pytest.fixture
def env_secrets_resolver(env_secrets_config) -> EnvSecretsResolver:
    return EnvSecretsResolver(env_secrets_config)
