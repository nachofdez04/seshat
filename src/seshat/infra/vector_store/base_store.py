from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from seshat.core.models.enums import SearchMode

if TYPE_CHECKING:
    from seshat.core.models.api_graph import NodeFilter, SearchResult


class AbstractVectorStore(ABC):
    @staticmethod
    @abstractmethod
    def get_supported_filter_fields() -> frozenset[str]:
        """Return the NodeFilter field names this implementation can express as metadata filters."""
        ...

    @abstractmethod
    async def upsert(self, node_id: str, text: str, metadata: dict) -> None: ...

    @abstractmethod
    async def search_dense(
        self,
        query: str,
        top_k: int,
        node_filter: NodeFilter | None = None,
        exclude_job_id: str | None = None,
        score_threshold: float | None = None,
    ) -> list[SearchResult]: ...

    @abstractmethod
    async def search_sparse(
        self,
        query: str,
        top_k: int,
        node_filter: NodeFilter | None = None,
        exclude_job_id: str | None = None,
    ) -> list[SearchResult]: ...

    async def search(
        self,
        query: str,
        top_k: int,
        node_filter: NodeFilter | None = None,
        exclude_job_id: str | None = None,
        score_threshold: float | None = None,
        mode: SearchMode = SearchMode.SEMANTIC,
    ) -> list[SearchResult]:
        """Convenience dispatcher for callers that receive mode at runtime."""
        match mode:
            case SearchMode.SEMANTIC:
                return await self.search_dense(query, top_k, node_filter, exclude_job_id, score_threshold)
            case SearchMode.KEYWORD:
                return await self.search_sparse(query, top_k, node_filter, exclude_job_id)
            case _:
                raise ValueError(f"Unsupported search mode: {mode!r}")

    @abstractmethod
    async def update_metadata(self, node_id: str, patch: dict) -> None: ...

    @abstractmethod
    async def delete(self, node_id: str) -> None: ...
