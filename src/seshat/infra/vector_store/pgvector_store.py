from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any
from uuid import UUID

import sqlalchemy as sa
from langchain_core.documents import Document
from langchain_postgres import PGVector
from sqlalchemy import Float, cast, func, select, text
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR

from seshat.core.models.api_graph import SearchResult
from seshat.core.utils.db import ensure_psycopg_scheme
from seshat.core.utils.log import get_logger
from seshat.infra.vector_store.base_store import AbstractVectorStore

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings
    from sqlalchemy.ext.asyncio import AsyncEngine

    from seshat.core.config.settings import VectorIndexConfig, VectorStoreConfig
    from seshat.core.models.api_graph import NodeFilter

logger = get_logger(__name__)

# langchain_pg_embedding is created lazily by LangChain on first connection, so
# we cannot rely on Alembic running this before the table exists.  The DDL is
# idempotent (ADD COLUMN IF NOT EXISTS / CREATE INDEX IF NOT EXISTS) and applied
# lazily before the first sparse or hybrid search.
_ENSURE_TS_CONTENT = text("""
    ALTER TABLE langchain_pg_embedding
    ADD COLUMN IF NOT EXISTS ts_content tsvector
    GENERATED ALWAYS AS (to_tsvector('english', document)) STORED;

    CREATE INDEX IF NOT EXISTS langchain_pg_embedding_ts_content_gin
    ON langchain_pg_embedding USING gin(ts_content);
""")


class PGVectorStore(AbstractVectorStore):
    def __init__(
        self,
        config: VectorStoreConfig,
        index: VectorIndexConfig,
        embeddings: Embeddings,
        connection_string: str,
    ) -> None:
        self._config = config
        self._index = index
        self._connection_string = self._validate_connection_string(connection_string)
        self._store = PGVector(
            embeddings=embeddings, collection_name=index.collection, connection=self._connection_string, async_mode=True
        )
        self._collection_id: str | None = None
        self._ts_content_ready = False

    @property
    def _engine(self) -> AsyncEngine:
        assert self._store._async_engine is not None, "PGVector async engine is not initialized"
        return self._store._async_engine

    @property
    def _ts_content(self) -> sa.ColumnClause[str]:
        # ts_content is a generated column not in the LangChain ORM model.
        return sa.column("ts_content", TSVECTOR)

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
        return frozenset(
            {"node_type", "min_confidence", "ingestion_source", "meeting_date_from", "meeting_date_to", "state"}
        )

    async def _ensure_ts_content(self) -> None:
        if self._ts_content_ready:
            return

        # With async_mode=True, PGVector skips __post_init__ and lazily calls __apost_init__
        # on the first operation; trigger it explicitly so EmbeddingStore/CollectionStore
        # are available on self._store before we build SQL statements against them.
        await self._store.__apost_init__()

        async with self._engine.begin() as conn:
            await conn.execute(_ENSURE_TS_CONTENT)

        self._ts_content_ready = True

    async def upsert(self, node_id: str, text: str, metadata: dict) -> None:
        logger.debug("Upserting vector for node_id=%s", node_id)
        doc = Document(page_content=text, metadata={**metadata, "node_id": node_id})
        await self._store.aadd_documents([doc], ids=[node_id])

    async def search_dense(
        self,
        query: str,
        top_k: int,
        node_filter: NodeFilter | None = None,
        exclude_job_id: str | None = None,
        score_threshold: float | None = None,
    ) -> list[SearchResult]:
        results = await self._similarity_search(
            query, top_k=top_k, node_filter=node_filter, exclude_job_id=exclude_job_id, score_threshold=score_threshold
        )
        return [SearchResult(node_id=doc.metadata["node_id"], score=score) for doc, score in results]

    async def search_sparse(
        self,
        query: str,
        top_k: int,
        node_filter: NodeFilter | None = None,
        exclude_job_id: str | None = None,
    ) -> list[SearchResult]:
        sparse = await self._sparse_search(query, top_k=top_k, node_filter=node_filter, exclude_job_id=exclude_job_id)
        return [SearchResult(node_id=UUID(node_id), score=score) for node_id, score in sparse]

    async def _sparse_search(
        self,
        query: str,
        top_k: int,
        node_filter: NodeFilter | None,
        exclude_job_id: str | None,
    ) -> list[tuple[str, float]]:
        if not query.strip():
            return []

        await self._ensure_ts_content()
        collection_id = await self._get_collection_id()

        ts_query_expr = func.to_tsquery("english", " | ".join(re.findall(r"\w+", query)))
        ts_rank = func.ts_rank_cd(self._ts_content, ts_query_expr)

        stmt = (
            select(
                self._store.EmbeddingStore.cmetadata["node_id"].as_string().label("node_id"),
                ts_rank.label("rank"),
            )
            .where(
                self._store.EmbeddingStore.collection_id == collection_id,
                self._ts_content.op("@@")(ts_query_expr),
            )
            .order_by(ts_rank.desc())
            .limit(top_k)
        )
        stmt = self._apply_sparse_filter(stmt, node_filter, exclude_job_id)

        async with self._engine.connect() as conn:
            result = await conn.execute(stmt)
            rows = result.fetchall()

        logger.debug(
            "sparse_search: query=%r collection_id=%s rows=%d",
            query[:60],
            collection_id,
            len(rows),
        )
        return [(row.node_id, float(row.rank)) for row in rows]

    async def _similarity_search(
        self,
        query: str,
        top_k: int,
        node_filter: NodeFilter | None,
        exclude_job_id: str | None,
        score_threshold: float | None,
    ) -> list[tuple[Document, float]]:
        semantic_filter = self._build_semantic_filter(node_filter, exclude_job_id)
        logger.debug(
            "similarity_search: query=%r top_k=%d score_threshold=%s filter=%r",
            query[:60],
            top_k,
            score_threshold,
            semantic_filter,
        )
        results = await self._store.asimilarity_search_with_relevance_scores(
            query, k=top_k, filter=semantic_filter, score_threshold=score_threshold
        )
        logger.debug("similarity_search: returned %d results", len(results))
        return results

    async def _get_collection_id(self) -> str:
        if self._collection_id is None:
            stmt = select(self._store.CollectionStore).where(self._store.CollectionStore.name == self._index.collection)
            async with self._engine.connect() as conn:
                result = await conn.execute(stmt)
                row = result.fetchone()

            if row is None:
                raise RuntimeError(f"Collection '{self._index.collection}' not found in langchain_pg_collection")

            self._collection_id = str(row.uuid)

        return self._collection_id

    def _apply_sparse_filter(self, stmt: Any, node_filter: NodeFilter | None, exclude_job_id: str | None) -> Any:
        if node_filter is not None:
            if node_filter.node_type:
                stmt = stmt.where(
                    self._store.EmbeddingStore.cmetadata["node_type"].as_string() == node_filter.node_type
                )
            if node_filter.min_confidence is not None:
                stmt = stmt.where(
                    cast(self._store.EmbeddingStore.cmetadata["confidence"].as_string(), Float)
                    >= node_filter.min_confidence
                )
            if node_filter.ingestion_source:
                stmt = stmt.where(
                    self._store.EmbeddingStore.cmetadata["ingestion_source"].as_string() == node_filter.ingestion_source
                )
            if node_filter.meeting_date_from is not None:
                stmt = stmt.where(
                    self._store.EmbeddingStore.cmetadata["meeting_date"].as_string()
                    >= str(node_filter.meeting_date_from)
                )
            if node_filter.meeting_date_to is not None:
                stmt = stmt.where(
                    self._store.EmbeddingStore.cmetadata["meeting_date"].as_string() <= str(node_filter.meeting_date_to)
                )
        if node_filter is not None and node_filter.state is not None:
            stmt = stmt.where(self._store.EmbeddingStore.cmetadata["state"].as_string() == node_filter.state)
        if exclude_job_id is not None:
            stmt = stmt.where(self._store.EmbeddingStore.cmetadata["job_id"].as_string() != exclude_job_id)

        return stmt

    def _build_semantic_filter(self, node_filter: NodeFilter | None, exclude_job_id: str | None = None) -> dict | None:
        if node_filter is None and exclude_job_id is None:
            return None

        result: dict[str, Any] = {}

        if node_filter is not None:
            unsupported = {f for f in node_filter.model_fields_set if f not in self.get_supported_filter_fields()}
            if unsupported:
                logger.warning(
                    "PGVector semantic filter ignoring unsupported fields %s (supported: %s) — "
                    "these filters are not applied at the vector-store layer.",
                    sorted(unsupported),
                    sorted(self.get_supported_filter_fields()),
                )
            if node_filter.node_type:
                result["node_type"] = node_filter.node_type
            if node_filter.min_confidence is not None:
                result["confidence"] = {"$gte": node_filter.min_confidence}
            if node_filter.ingestion_source:
                result["ingestion_source"] = node_filter.ingestion_source
            if node_filter.meeting_date_from is not None:
                result.setdefault("meeting_date", {})["$gte"] = str(node_filter.meeting_date_from)
            if node_filter.meeting_date_to is not None:
                result.setdefault("meeting_date", {})["$lte"] = str(node_filter.meeting_date_to)

        if node_filter is not None and node_filter.state is not None:
            result["state"] = node_filter.state

        if exclude_job_id is not None:
            result["job_id"] = {"$ne": exclude_job_id}

        return result

    async def update_metadata(self, node_id: str, patch: dict) -> None:
        # sa.literal with type_=JSONB tells psycopg3 to bind patch as a JSONB object;
        # passing json.dumps(patch) as a plain string would make || concatenate as an array.
        stmt = (
            sa.update(self._store.EmbeddingStore)
            .where(self._store.EmbeddingStore.cmetadata["node_id"].as_string() == node_id)
            .values(cmetadata=self._store.EmbeddingStore.cmetadata.op("||")(sa.literal(patch, type_=JSONB)))
        )
        async with self._engine.begin() as conn:
            await conn.execute(stmt)

    async def delete(self, node_id: str) -> None:
        logger.debug("Deleting vector for node_id=%s", node_id)
        await self._store.adelete(ids=[node_id])
