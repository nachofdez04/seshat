from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any
from uuid import UUID

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from seshat.app.agents.base import RetryExhaustedError, _BaseAgent
from seshat.core.models.api_graph import SearchResult
from seshat.core.models.enums import SearchMode
from seshat.core.utils.hashing import fingerprint
from seshat.core.utils.log import get_logger

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

    from seshat.core.config.settings import RAGConfig
    from seshat.infra.vector_store.base_store import AbstractVectorStore

logger = get_logger(__name__)


class _Keywords(BaseModel):
    keywords: list[str]


class _QueryVariants(BaseModel):
    variants: list[str]


_KEYWORD_EXTRACTION_PROMPT = """\
You are extracting search keywords for a knowledge base of meeting notes.
The KB contains four node types: decisions, risks, action items, and open questions.

Given a query node, extract the most discriminating search terms — words that would \
uniquely identify semantically related nodes in the KB. Prioritise:
- Proper nouns and named tools/systems (e.g. "Flyway", "PagerDuty", "Terraform")
- Domain-specific technical terms (e.g. "schema drift", "rollback", "SLO breach")
- The specific subject or object of the relationship (what was decided/risked/asked)

Avoid:
- Generic words that appear in most nodes ("service", "team", "issue", "change")
- Node-type words ("decision", "risk", "action item", "question")
- Stop words and filler

Return 3-6 keywords or short phrases as a JSON object with a "keywords" array.
"""

_MULTI_QUERY_PROMPT_TEMPLATE = """\
You are generating alternative search queries for a knowledge base lookup.
Given an input query, produce {{num_variants}} alternative phrasings that capture \
the same intent from different angles (e.g. more abstract, more specific, \
using synonyms, or from a different perspective).

Return exactly {{num_variants}} queries as a JSON object with a "variants" array.
"""


class _KeywordExtractionFailed(RetryExhaustedError): ...


class _MultiQueryFailed(RetryExhaustedError): ...


class _KeywordAgent(_BaseAgent):
    @property
    def _system_prompt(self) -> str:
        return _KEYWORD_EXTRACTION_PROMPT

    async def extract(self, query: str) -> str:
        messages = [SystemMessage(_KEYWORD_EXTRACTION_PROMPT), HumanMessage(query)]
        result: _Keywords = await self._retryable_structured_ainvoke(
            messages,
            _Keywords,
            raise_on_exhaustion=_KeywordExtractionFailed("keyword extraction exhausted retries"),
        )
        keywords = " ".join(result.keywords)
        logger.debug("keyword extraction: %r -> %r", query[:60], keywords[:60])
        return keywords


class _MultiQueryAgent(_BaseAgent):
    @property
    def _system_prompt(self) -> str:
        return _MULTI_QUERY_PROMPT_TEMPLATE

    async def generate(self, query: str, num_variants: int) -> list[str]:
        prompt = _MULTI_QUERY_PROMPT_TEMPLATE.replace("{{num_variants}}", str(num_variants))
        messages = [SystemMessage(prompt), HumanMessage(query)]
        result: _QueryVariants = await self._retryable_structured_ainvoke(
            messages,
            _QueryVariants,
            raise_on_exhaustion=_MultiQueryFailed("multi-query exhausted retries"),
        )
        variants = result.variants[:num_variants]
        logger.debug("multi-query: generated %d variants for %r", len(variants), query[:60])
        return variants


class SearchEngine:
    def __init__(
        self,
        rag_config: RAGConfig,
        vector_store: AbstractVectorStore,
        keyword_llm: BaseChatModel | None,
        multi_query_llm: BaseChatModel | None,
    ) -> None:
        self._rag_config = rag_config
        self._vs = vector_store
        self._keyword_agent = (
            _KeywordAgent(keyword_llm, rag_config.keyword_extraction_llm)
            if keyword_llm is not None and rag_config.keyword_extraction_llm is not None
            else None
        )
        self._multi_query_agent = (
            _MultiQueryAgent(multi_query_llm, rag_config.multi_query.llm)
            if multi_query_llm is not None and rag_config.multi_query is not None
            else None
        )

    @property
    def search_mode(self) -> SearchMode:
        return self._rag_config.search_mode

    async def search(
        self,
        query: str,
        *,
        node_filter: Any | None = None,
        exclude_job_id: str | None = None,
        top_k: int | None = None,
        score_threshold: float | None = None,
    ) -> list[SearchResult]:
        logger.debug("search: mode=%s query=%r", self.search_mode.value, query[:60])
        common_search_kwargs = {
            "node_filter": node_filter,
            "exclude_job_id": exclude_job_id,
            "top_k": top_k if top_k is not None else self._rag_config.top_k,
        }

        match self.search_mode:
            case SearchMode.SEMANTIC:
                results = await self._semantic_search(query, score_threshold=score_threshold, **common_search_kwargs)
            case SearchMode.KEYWORD:
                results = await self._keyword_search(query, **common_search_kwargs)
            case SearchMode.HYBRID:
                results = await self._hybrid_search(query, score_threshold=score_threshold, **common_search_kwargs)
            case _:
                raise ValueError(f"Unsupported search mode: {self.search_mode.value!r}")

        logger.debug("search: returned %d results", len(results))
        return results

    async def _semantic_search(self, query: str, **kwargs: Any) -> list[SearchResult]:
        variants = await self._generate_variants(query)
        if not variants:
            return await self._vs.search_dense(query, **kwargs)

        queries = [query, *variants]
        result_lists = await asyncio.gather(*[self._vs.search_dense(q, **kwargs) for q in queries])
        return _rrf(result_lists, [])

    async def _keyword_search(self, query: str, **kwargs: Any) -> list[SearchResult]:
        keywords = await self._extract_keywords(query)
        return await self._vs.search_sparse(query=(keywords or query), **kwargs)

    async def _hybrid_search(self, query: str, **kwargs: Any) -> list[SearchResult]:
        score_threshold = kwargs.pop("score_threshold", None)
        semantic, keyword = await asyncio.gather(
            self._semantic_search(query, score_threshold=score_threshold, **kwargs),
            self._keyword_search(query, **kwargs),
        )
        return _rrf([semantic], [keyword])

    async def _extract_keywords(self, query: str) -> str | None:
        if self._keyword_agent is not None:
            try:
                return await self._keyword_agent.extract(query)
            except Exception:
                logger.warning("keyword extraction LLM call failed; using original query")
                return None
        return query

    async def _generate_variants(self, query: str) -> list[str]:
        multi_cfg = self._rag_config.multi_query
        if multi_cfg is not None and self._multi_query_agent is not None:
            try:
                return await self._multi_query_agent.generate(query, multi_cfg.num_variants)
            except Exception:
                logger.warning("multi-query generation LLM call failed; using original query only")
                return []
        return []

    def fingerprint(self) -> str:
        """Stable hash over the retrieval config; used by eval to bust the result cache on any change."""
        parts = [
            self._rag_config.search_mode.value,
            self._rag_config.keyword_extraction_llm.model if self._rag_config.keyword_extraction_llm else "none",
            (
                f"{self._rag_config.multi_query.llm.model}:n{self._rag_config.multi_query.num_variants}"
                if self._rag_config.multi_query
                else "none"
            ),
            (
                f"{self._rag_config.reranker.provider.value}:{self._rag_config.reranker.model}"
                if self._rag_config.reranker
                else "none"
            ),
        ]
        return fingerprint(":".join(parts))

    def prompt_texts(self) -> dict[str, str]:
        """Returns the active prompt strings keyed by role; used by MLflow to log the prompts alongside run params."""
        texts: dict[str, str] = {}
        if self._keyword_agent is not None:
            texts["keyword_extraction"] = _KEYWORD_EXTRACTION_PROMPT
        if self._multi_query_agent is not None:
            texts["multi_query"] = _MULTI_QUERY_PROMPT_TEMPLATE
        return texts


def _rrf(
    dense_leg: list[list[SearchResult]],
    sparse_leg: list[list[SearchResult]],
    k: int = 60,
) -> list[SearchResult]:
    """Reciprocal Rank Fusion across multiple result lists. Score = sum(1 / (k + rank)) per node across all legs."""
    scores: dict[str, float] = {}
    for dense_results in dense_leg:
        for rank, result in enumerate(dense_results):
            node_id = str(result.node_id)
            scores[node_id] = scores.get(node_id, 0.0) + 1.0 / (k + rank)

    for sparse_results in sparse_leg:
        for rank, result in enumerate(sparse_results):
            node_id = str(result.node_id)
            scores[node_id] = scores.get(node_id, 0.0) + 1.0 / (k + rank)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [SearchResult(node_id=UUID(node_id), score=score) for node_id, score in ranked]
