import pytest
from langchain_openai import AzureOpenAIEmbeddings

from seshat.app.platform.observability.usage_tracker import TrackingEmbeddings
from seshat.core.config.settings import IdentificationLLMConfig
from tests.integration._probes import _openai_reachable
from tests.integration.helpers import cheap_identification_config


@pytest.fixture(scope="module")
def identification_config() -> IdentificationLLMConfig:
    return cheap_identification_config()


@pytest.fixture(scope="module")
def azure_embeddings() -> TrackingEmbeddings:
    if not _openai_reachable():
        pytest.skip("OpenAI API not reachable — OPENAI_API_KEY not set or network issue")
    return TrackingEmbeddings(AzureOpenAIEmbeddings())
