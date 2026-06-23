import pytest
from langchain_core.language_models import BaseChatModel

from seshat.config.settings import ExtractionConfig, GroundingLLMConfig, ResolutionLLMConfig
from tests.integration.helpers import (
    cheap_grounding_config,
    cheap_identification_config,
    cheap_resolution_config,
    make_cheap_llm,
)


@pytest.fixture
def cheap_llm() -> BaseChatModel:
    return make_cheap_llm()


@pytest.fixture
def extraction_config() -> ExtractionConfig:
    return ExtractionConfig(identification=cheap_identification_config())


@pytest.fixture
def resolution_config() -> ResolutionLLMConfig:
    return cheap_resolution_config()


@pytest.fixture
def grounding_config() -> GroundingLLMConfig:
    return cheap_grounding_config()
