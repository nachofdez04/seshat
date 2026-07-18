from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.embeddings import Embeddings

if TYPE_CHECKING:
    from pydantic import SecretStr


class CohereEmbeddings(Embeddings):
    """Cohere embeddings via the native `cohere` SDK.

    Not implemented through langchain-cohere: every released version of that
    package pins cohere<6.0, which conflicts with the cohere>=7.0.5 this project
    already depends on for CohereReranker's AsyncClientV2.
    """

    def __init__(self, model: str, api_key: SecretStr, *, timeout: float | None = None, max_retries: int = 3) -> None:
        import cohere

        self.model = model
        self._client = cohere.AsyncClientV2(
            api_key=api_key.get_secret_value(), timeout=timeout, max_retries=max_retries
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError("CohereEmbeddings only supports async use (aembed_documents/aembed_query).")

    def embed_query(self, text: str) -> list[float]:
        raise NotImplementedError("CohereEmbeddings only supports async use (aembed_documents/aembed_query).")

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return await self._embed(texts, input_type="search_document")

    async def aembed_query(self, text: str) -> list[float]:
        embeddings = await self._embed([text], input_type="search_query")
        return embeddings[0]

    async def _embed(self, texts: list[str], *, input_type: str) -> list[list[float]]:
        response = await self._client.embed(
            model=self.model,
            input_type=input_type,  # type: ignore[arg-type]
            texts=texts,
            embedding_types=["float"],
        )
        assert response.embeddings.float_ is not None, "Cohere embed response missing float embeddings"
        return response.embeddings.float_
