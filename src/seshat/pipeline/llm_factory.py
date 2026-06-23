from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain.chat_models import init_chat_model
from pydantic import SecretStr

from seshat.models.enums import LLMProvider
from seshat.secrets.factory import get_secrets_resolver
from seshat.utils.log import get_logger

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

    from seshat.config.settings import SeshatConfig, _LLMConfig

logger = get_logger(__name__)


# TODO: add a _ping_llm(llm) startup health check (minimal single-token call) once the app entrypoint exists
def get_identification_llm(config: SeshatConfig) -> BaseChatModel:
    return _build_llm(config.extraction.identification, config)


def get_grounding_llm(config: SeshatConfig) -> BaseChatModel:
    if config.extraction.grounding is None:
        raise ValueError("grounding is not configured")
    return _build_llm(config.extraction.grounding, config)


def get_resolution_llm(config: SeshatConfig) -> BaseChatModel:
    return _build_llm(config.extraction.resolution, config)


_PROVIDERS_WITHOUT_API_KEY = frozenset({LLMProvider.BEDROCK_CONVERSE})
_PROMPT_CACHING_PROVIDERS = frozenset({LLMProvider.ANTHROPIC})


def _build_llm(llm: _LLMConfig, config: SeshatConfig) -> BaseChatModel:
    kwargs: dict[str, Any] = {
        "model_provider": llm.provider,
        "temperature": llm.temperature,
        "timeout": llm.timeout_seconds,
    }

    if llm.max_output_tokens is not None:
        kwargs["max_tokens"] = llm.max_output_tokens

    if llm.provider not in _PROVIDERS_WITHOUT_API_KEY:
        secrets = get_secrets_resolver(config)
        kwargs["api_key"] = SecretStr(secrets.get_secret(llm.api_key_secret_key))  # type: ignore[arg-type]

    if llm.provider in _PROMPT_CACHING_PROVIDERS:
        # user-supplied model_kwargs win; caching header is the default and is overridden if explicitly set
        model_kwargs = kwargs.get("model_kwargs", {})
        kwargs["model_kwargs"] = {"extra_headers": {"anthropic-beta": "prompt-caching-2024-07-31"}} | model_kwargs

    logger.debug("Building LLM: provider=%s model=%s", llm.provider, llm.model)
    return init_chat_model(llm.model, **kwargs)
