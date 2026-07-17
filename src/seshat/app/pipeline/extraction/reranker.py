from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import cohere
import voyageai

from seshat.app.platform.observability.usage_tracker import get_run_tracker
from seshat.core.models.enums import RerankerProvider
from seshat.core.utils.log import get_logger

if TYPE_CHECKING:
    from cohere import V2RerankResponse
    from voyageai.object import RerankingObject

    from seshat.core.config.settings import RerankerConfig
    from seshat.core.models.nodes import KBNode

logger = get_logger(__name__)


class AbstractReranker(ABC):
    def __init__(self, config: RerankerConfig, api_key: str) -> None:
        self._config = config
        self._max_retries = config.max_retries
        self._timeout = config.timeout_seconds
        self._api_key = api_key

    @abstractmethod
    async def _rerank(self, query: str, nodes: list[KBNode]) -> list[KBNode]: ...

    async def rerank(self, query: str, nodes: list[KBNode]) -> list[KBNode]:
        nodes = await self._rerank(query, nodes)
        result = nodes[: self._config.top_n] if self._config.top_n is not None else nodes
        logger.debug("rerank done: %d -> %d nodes (top_n=%s)", len(nodes), len(result), self._config.top_n)
        return result


class CohereReranker(AbstractReranker):
    def __init__(self, config: RerankerConfig, api_key: str) -> None:
        super().__init__(config, api_key)
        self._client = cohere.AsyncClientV2(api_key=api_key, timeout=self._timeout, max_retries=self._max_retries)

    async def _rerank(self, query: str, nodes: list[KBNode]) -> list[KBNode]:
        logger.debug("cohere rerank: query=%r nodes=%d model=%s", query[:60], len(nodes), self._config.model)
        docs = [n.vector_store_text for n in nodes]
        response = await self._client.rerank(
            model=self._config.model,
            query=query,
            documents=docs,
            top_n=len(docs),
        )

        await self._record_reranker_cost(response)
        return [nodes[item.index] for item in response.results]

    async def _record_reranker_cost(self, response: V2RerankResponse) -> None:  # type: ignore
        if response.meta and response.meta.tokens:
            input_tokens = response.meta.tokens.input_tokens
        else:
            logger.warning("Cohere reranker response missing token usage info; cannot record reranker cost")
            return

        callback = get_run_tracker()
        if callback is None:
            logger.warning("No active token budget tracker found in context; reranker tokens will not be tracked")
            return
        await callback.tracker.add(reranker_input_tokens=input_tokens)  # type: ignore


class VoyageReranker(AbstractReranker):
    def __init__(self, config: RerankerConfig, api_key: str) -> None:
        super().__init__(config, api_key)
        self._client = voyageai.AsyncClient(api_key=api_key, timeout=self._timeout, max_retries=self._max_retries)

    async def _rerank(self, query: str, nodes: list[KBNode]) -> list[KBNode]:
        logger.debug("voyage rerank: query=%r nodes=%d model=%s", query[:60], len(nodes), self._config.model)
        docs = [n.vector_store_text for n in nodes]
        response = await self._client.rerank(
            query=query,
            documents=docs,
            model=self._config.model,
            top_k=len(docs),
        )

        await self._record_reranker_cost(response)
        return [nodes[item.index] for item in response.results]

    async def _record_reranker_cost(self, response: RerankingObject) -> None:
        callback = get_run_tracker()
        if callback is None:
            logger.warning("No active token budget tracker found in context; reranker tokens will not be tracked")
            return
        await callback.tracker.add(reranker_input_tokens=response.total_tokens)


def reranker_factory(config: RerankerConfig, api_key: str) -> AbstractReranker:
    match config.provider:
        case RerankerProvider.COHERE:
            return CohereReranker(config, api_key)
        case RerankerProvider.VOYAGE:
            return VoyageReranker(config, api_key)
        case _:
            raise ValueError(f"Unsupported reranker provider: {config.provider}")
