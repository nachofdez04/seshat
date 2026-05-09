from __future__ import annotations

import logging
from functools import lru_cache
from typing import TYPE_CHECKING

from seshat.models.enums import SecretsProvider

if TYPE_CHECKING:
    from seshat.config.settings import SecretsConfig, SeshatConfig
    from seshat.secrets.base_resolver import AbstractSecretsResolver


logger = logging.getLogger(__name__)


def get_secrets_resolver(config: SeshatConfig) -> AbstractSecretsResolver:
    return _cached_resolver(config.secrets)  # type: ignore[arg-type]


@lru_cache(maxsize=1)
def _cached_resolver(config: SecretsConfig) -> AbstractSecretsResolver:
    logger.debug("Initialising secrets resolver: %s", config.provider)
    match config.provider:
        case SecretsProvider.ENV:
            from seshat.secrets.env_resolver import EnvSecretsResolver

            return EnvSecretsResolver(config)
        case SecretsProvider.AWS:
            from seshat.secrets.aws_resolver import AWSSecretsResolver

            return AWSSecretsResolver(config)
        case _:
            raise ValueError(f"Unknown secrets provider: {config.provider!r}")
