from unittest.mock import MagicMock

import pytest
from botocore.exceptions import ClientError


class TestAWSSecretsResolver:
    def test_get_secret_returns_value(self, aws_secrets_resolver):
        client = MagicMock()
        client.get_secret_value.return_value = {"SecretString": "my-secret"}
        aws_secrets_resolver._client = client

        assert aws_secrets_resolver.get_secret("my-key") == "my-secret"
        client.get_secret_value.assert_called_once_with(SecretId="seshat/my-key")

    def test_get_secret_missing_raises_key_error(self, aws_secrets_resolver):
        client = MagicMock()
        client.get_secret_value.side_effect = ClientError(
            error_response={"Error": {"Code": "ResourceNotFoundException", "Message": "not found"}},
            operation_name="GetSecretValue",
        )
        aws_secrets_resolver._client = client

        with pytest.raises(KeyError, match="seshat/missing-key"):
            aws_secrets_resolver.get_secret("missing-key")
