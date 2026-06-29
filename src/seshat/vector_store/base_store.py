from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from seshat.models.enums import SearchMode

if TYPE_CHECKING:
    from seshat.models.api_graph import NodeFilter, SearchResult


class AbstractVectorStore(ABC):
    @staticmethod
    @abstractmethod
    def get_supported_filter_fields() -> frozenset[str]:
        """Return the NodeFilter field names this implementation can express as metadata filters."""
        ...

    @abstractmethod
    async def upsert(self, node_id: str, text: str, metadata: dict) -> None: ...

    @abstractmethod
    async def search(
        self,
        query: str,
        top_k: int,
        node_filter: NodeFilter | None = None,
        exclude_job_id: str | None = None,
        score_threshold: float | None = None,
        mode: SearchMode = SearchMode.SEMANTIC,
    ) -> list[SearchResult]:
        # RAG contract (spec §5): callers SHOULD set node_filter.node_type before calling
        # search to restrict results to one concept type and avoid over-broad context.
        # Only fields returned by get_supported_filter_fields() are usable; others raise NotImplementedError.
        ...

    @abstractmethod
    async def delete(self, node_id: str) -> None: ...
