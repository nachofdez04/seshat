import boto3
import pytest

from seshat.secrets.aws_resolver import AWSSecretsResolver
from tests.integration.conftest import LOCALSTACK_REGION, SKIP_IF_NO_LOCALSTACK

pytestmark = [pytest.mark.integration, SKIP_IF_NO_LOCALSTACK]


@pytest.fixture
def secrets_config(localstack_secretsmanager_url):
    from seshat.config.settings import SecretsConfig

    return SecretsConfig(
        region=LOCALSTACK_REGION,
        secret_path_prefix="seshat",
        endpoint_url=localstack_secretsmanager_url,
    )


@pytest.fixture
def aws_secrets_resolver(monkeypatch, secrets_config):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test")
    return AWSSecretsResolver(secrets_config)


@pytest.fixture
def seeded_secret(localstack_secretsmanager_url):
    secrets_manager = boto3.client(
        "secretsmanager", region_name=LOCALSTACK_REGION, endpoint_url=localstack_secretsmanager_url
    )

    key, value = "e2e-key", "e2e-value"
    secrets_manager.create_secret(Name=f"seshat/{key}", SecretString=value)
    yield key, value
    secrets_manager.delete_secret(SecretId=f"seshat/{key}", ForceDeleteWithoutRecovery=True)


class TestAWSSecretsResolverLocalStack:
    def test_get_secret_end_to_end(self, aws_secrets_resolver, seeded_secret):
        key, value = seeded_secret
        assert aws_secrets_resolver.get_secret(key) == value

    def test_get_missing_secret_raises_key_error(self, aws_secrets_resolver):
        with pytest.raises(KeyError, match="seshat/no-such-key"):
            aws_secrets_resolver.get_secret("no-such-key")
