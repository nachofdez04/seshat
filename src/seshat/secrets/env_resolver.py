import os

from seshat.secrets.base_resolver import AbstractSecretsResolver


class EnvSecretsResolver(AbstractSecretsResolver):
    def _fetch_secret(self, key: str) -> str:
        value = os.environ.get(key)
        if value is None:
            raise KeyError(key)
        if not value:
            raise ValueError(f"Secret {key!r} is set but empty — check your environment")
        return value
