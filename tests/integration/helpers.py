import os
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from seshat.config.settings import (
    GroundingLLMConfig,
    IdentificationLLMConfig,
    ResolutionLLMConfig,
)
from seshat.models.enums import LLMProvider, RelationshipType
from seshat.models.nodes import KBNode, KBRelationship
from tests.integration.conftest import _BEDROCK_PROFILE, _anthropic_reachable, _azure_available, _bedrock_available

_PROVIDER2CHEAP_MODEL_MAPPING: dict[LLMProvider, str] = {
    LLMProvider.BEDROCK_CONVERSE: "eu.anthropic.claude-haiku-4-5-20251001-v1:0",
    LLMProvider.AZURE_OPENAI: "gpt-5-nano",
    LLMProvider.ANTHROPIC: "claude-haiku-4-5-20251001",
    LLMProvider.OPENAI: "gpt-5-nano",
}


def _pick_provider() -> LLMProvider:
    if _bedrock_available(profile_name=_BEDROCK_PROFILE):
        return LLMProvider.BEDROCK_CONVERSE
    if _azure_available():
        return LLMProvider.AZURE_OPENAI
    if os.environ.get("ANTHROPIC_API_KEY"):
        return LLMProvider.ANTHROPIC
    return LLMProvider.OPENAI


def _get_cheap_model(provider: LLMProvider) -> str:
    if provider not in _PROVIDER2CHEAP_MODEL_MAPPING:
        raise ValueError(f"No cheap model configured for provider {provider}")
    return _PROVIDER2CHEAP_MODEL_MAPPING[provider]


def _pick_grounding_provider() -> LLMProvider:
    # Must differ in provider family from _pick_provider() (ExtractionConfig validator).
    if _anthropic_reachable():
        return LLMProvider.AZURE_OPENAI if _azure_available() else LLMProvider.OPENAI
    return LLMProvider.BEDROCK_CONVERSE if _bedrock_available(profile_name=_BEDROCK_PROFILE) else LLMProvider.ANTHROPIC


def make_cheap_llm() -> BaseChatModel:
    provider = _pick_provider()
    kwargs: dict[str, Any] = {
        "model_provider": provider,
        "temperature": 0.0,
    }
    if provider == LLMProvider.BEDROCK_CONVERSE:
        kwargs["credentials_profile_name"] = _BEDROCK_PROFILE

    return init_chat_model(model=_get_cheap_model(provider), **kwargs)


def cheap_identification_config() -> IdentificationLLMConfig:
    provider = _pick_provider()
    return IdentificationLLMConfig(provider=provider, model=_get_cheap_model(provider))


def cheap_resolution_config() -> ResolutionLLMConfig:
    provider = _pick_provider()
    return ResolutionLLMConfig(provider=provider, model=_get_cheap_model(provider))


def cheap_grounding_config() -> GroundingLLMConfig:
    provider = _pick_grounding_provider()
    return GroundingLLMConfig(provider=provider, model=_get_cheap_model(provider), max_retries=1)


async def upload_transcript(blob_store, content: str) -> str:
    blob_key = f"transcripts/{uuid4()}.txt"
    await blob_store.put(blob_key, content.encode())
    return blob_key


async def seed_node(
    node,
    kb_store,
    vector_store,
    *,
    job_id: str | None = None,
    write_to_kb: bool = True,
    write_to_vector: bool = True,
) -> None:
    stored = (
        node
        if job_id is None
        else node.model_copy(update={"metadata": node.metadata.model_copy(update={"job_id": job_id})})
    )
    if write_to_kb:
        await kb_store.write_node(stored)
    if write_to_vector:
        metadata = {"node_type": node.type.value, "confidence": node.confidence}
        if job_id is not None:
            metadata["job_id"] = job_id
        await vector_store.upsert(str(node.id), f"{node.title} {node.description}", metadata)


def make_relationship(
    src: KBNode,
    tgt: KBNode,
    rel_type: RelationshipType = RelationshipType.SUPERSEDES,
    job_id: str = "job-1",
) -> KBRelationship:
    return KBRelationship(
        source_id=src.id,
        target_id=tgt.id,
        rel_type=rel_type,
        job_id=job_id,
        created_at=datetime.now(UTC),
    )
