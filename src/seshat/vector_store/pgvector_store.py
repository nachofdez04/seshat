from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from langchain_core.documents import Document
from langchain_postgres import PGVector

from seshat.utils.db import ensure_psycopg_scheme
from seshat.vector_store.base_store import AbstractVectorStore

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings

    from seshat.config.settings import VectorIndexConfig, VectorStoreConfig
    from seshat.models.api import NodeFilter, SearchResult

logger = logging.getLogger(__name__)


class PGVectorStore(AbstractVectorStore):
    def __init__(
        self, config: VectorStoreConfig, index: VectorIndexConfig, embeddings: Embeddings, connection_string: str
    ) -> None:
        self._config = config
        self._index = index
        self._connection_string = self._validate_connection_string(connection_string)
        self._store = PGVector(
            embeddings=embeddings, collection_name=index.collection, connection=self._connection_string, async_mode=True
        )

    @staticmethod
    def _validate_connection_string(connection_string: str) -> str:
        return ensure_psycopg_scheme(
            connection_string,
            warn_msg=(
                "Unexpected driver %r in vector store connection string; "
                "replacing with '+psycopg' for langchain-postgres compatibility."
            ),
        )

    @staticmethod
    def get_supported_filter_fields() -> frozenset[str]:
        return frozenset({"node_type", "min_confidence", "ingestion_source"})

    async def upsert(self, node_id: str, text: str, metadata: dict) -> None:
        doc = Document(page_content=text, metadata={**metadata, "node_id": node_id})
        await self._store.aadd_documents([doc], ids=[node_id])

    async def search(
        self,
        query: str,
        top_k: int,
        node_filter: NodeFilter | None = None,
    ) -> list[SearchResult]:
        from seshat.models.api import SearchResult

        filter_dict = _build_filter(node_filter)
        results = await self._store.asimilarity_search_with_relevance_scores(query, k=top_k, filter=filter_dict)
        return [SearchResult(node_id=doc.metadata["node_id"], score=score) for doc, score in results]

    async def delete(self, node_id: str) -> None:
        await self._store.adelete(ids=[node_id])


def _build_filter(node_filter: NodeFilter | None) -> dict | None:
    if node_filter is None:
        return None

    unsupported = {f for f in node_filter.model_fields_set if f not in PGVectorStore.get_supported_filter_fields()}
    if unsupported:
        raise NotImplementedError(
            f"PGVector metadata filter does not support: {sorted(unsupported)}. "
            "Use PostgresKBStore.query() for full NodeFilter support."
        )

    result: dict[str, Any] = {}
    if node_filter.node_type:
        result["node_type"] = node_filter.node_type.value
    if node_filter.min_confidence is not None:
        result["confidence"] = {"$gte": node_filter.min_confidence}
    if node_filter.ingestion_source:
        result["ingestion_source"] = node_filter.ingestion_source.value
    return result
