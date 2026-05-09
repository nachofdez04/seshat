from __future__ import annotations

from typing import TYPE_CHECKING

import boto3
from botocore.exceptions import ClientError

from seshat.secrets.base_resolver import AbstractSecretsResolver

if TYPE_CHECKING:
    from seshat.config.settings import SecretsConfig


class AWSSecretsResolver(AbstractSecretsResolver):
    def __init__(self, config: SecretsConfig) -> None:
        super().__init__(config)
        self._client = boto3.client("secretsmanager", region_name=config.region, endpoint_url=config.endpoint_url)
        self._prefix = config.secret_path_prefix

    def _fetch_secret(self, key: str) -> str:
        full_key = f"{self._prefix}/{key}"
        try:
            response = self._client.get_secret_value(SecretId=full_key)
        except ClientError as exc:
            raise KeyError(full_key) from exc
        return response["SecretString"]
