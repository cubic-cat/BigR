"""Embedding client backed by OpenAI-compatible APIs."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Protocol, Sequence

from configs import EmbeddingConfig, embedding_config

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None


class SupportsEmbedding(Protocol):
    """Protocol for embedding providers used by the retriever."""

    def embed_text(self, text: str) -> list[float]:
        """Embed a single text string."""

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of text strings."""


class EmbeddingClient:
    """Thin wrapper over an OpenAI-compatible embedding API."""

    def __init__(self, config: EmbeddingConfig | None = None) -> None:
        self.config = config or embedding_config
        self._client: OpenAI | None = None

    @property
    def client(self) -> OpenAI:
        """Create the API client lazily so imports stay cheap."""
        if self._client is None:
            if OpenAI is None:
                raise ImportError(
                    "openai package is required for EmbeddingClient. "
                    "Install it with `pip install openai`."
                )
            if not self.config.api_key:
                raise ValueError(
                    "Embedding API key is missing. Set EMBEDDING_API_KEY or "
                    "the provider-specific key in `.env`."
                )

            client_kwargs: dict[str, object] = {
                "api_key": self.config.api_key,
                "timeout": self.config.request_timeout,
            }
            if self.config.base_url:
                client_kwargs["base_url"] = self.config.base_url
            self._client = OpenAI(**client_kwargs)
        return self._client

    def is_configured(self) -> bool:
        """Return whether the embedding client has a usable API key."""
        return bool(self.config.api_key)

    def embed_text(self, text: str) -> list[float]:
        """Embed a single text string."""
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of strings and return dense vectors."""
        if not texts:
            return []

        vectors: list[list[float]] = []
        for batch in _iter_batches(texts, self.config.batch_size):
            request_kwargs: dict[str, object] = {
                "model": self.config.model_name,
                "input": [self._prepare_text(text) for text in batch],
                "encoding_format": "float",
            }
            if self._supports_dimensions():
                request_kwargs["dimensions"] = self.config.dimensions

            response = self.client.embeddings.create(**request_kwargs)
            batch_vectors = [list(item.embedding) for item in response.data]

            if self.config.normalize:
                batch_vectors = [self._normalize_vector(vector) for vector in batch_vectors]

            vectors.extend(batch_vectors)
        return vectors

    def _supports_dimensions(self) -> bool:
        provider = self.config.provider.strip().lower()
        model_name = self.config.model_name.strip().lower()
        return provider == "openai" and model_name.startswith("text-embedding-3")

    @staticmethod
    def _prepare_text(text: str) -> str:
        cleaned = text.strip()
        return cleaned or " "

    @staticmethod
    def _normalize_vector(vector: Sequence[float]) -> list[float]:
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return [float(value) for value in vector]
        return [float(value) / norm for value in vector]


def _iter_batches(items: Sequence[str], batch_size: int) -> Sequence[Sequence[str]]:
    for start in range(0, len(items), max(batch_size, 1)):
        yield items[start : start + max(batch_size, 1)]


def main() -> None:
    """Build the local vector store from `data/processed`."""
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from core.rag_chain import RAGChain

    chain = RAGChain()
    if not chain.retriever.embedding_client.is_configured():
        raise ValueError(
            "Embedding API key is missing. Set EMBEDDING_API_KEY or the "
            "provider-specific key in `.env` before generating vectors."
        )

    documents = chain.build_knowledge_base(replace=True, persist=True)
    stats = chain.retriever.stats()
    print(f"Embedded {len(documents)} documents from {chain.knowledge_base_dir}.")
    print(f"Collection: {stats['collection_name']}")
    print(f"Store: {stats['store_path']}")
    print(f"Dimensions: {stats['dimensions']}")


if __name__ == "__main__":
    main()
