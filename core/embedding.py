"""Embedding clients: API-based (OpenAI-compatible) and local (sentence-transformers)."""

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
        return bool(self.config.api_key)

    def embed_text(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors: list[list[float]] = []
        for batch in _iter_batches(texts, self.config.batch_size):
            request_kwargs: dict[str, object] = {
                "model": self.config.model_name,
                "input": [self._prepare_text(t) for t in batch],
                "encoding_format": "float",
            }
            if self._supports_dimensions():
                request_kwargs["dimensions"] = self.config.dimensions
            response = self.client.embeddings.create(**request_kwargs)
            batch_vectors = [list(item.embedding) for item in response.data]
            if self.config.normalize:
                batch_vectors = [_normalize(v) for v in batch_vectors]
            vectors.extend(batch_vectors)
        return vectors

    def _supports_dimensions(self) -> bool:
        provider = self.config.provider.strip().lower()
        model_name = self.config.model_name.strip().lower()
        return provider == "openai" and model_name.startswith("text-embedding-3")

    @staticmethod
    def _prepare_text(text: str) -> str:
        return text.strip() or " "


class LocalEmbeddingClient:
    """Embedding client backed by a local sentence-transformers model.

    Supports any model on HuggingFace, e.g. BAAI/bge-m3, BAAI/bge-large-en-v1.5.
    Uses MPS on Apple Silicon, CUDA if available, otherwise CPU.

    Args:
        model_name: HuggingFace model ID or local path.
        batch_size: Texts per forward pass. Larger = faster but more RAM.
        normalize: L2-normalise output vectors (recommended for cosine similarity).
        device: Force a specific device ('cpu', 'cuda', 'mps'). Auto-detected if None.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        batch_size: int = 32,
        normalize: bool = True,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize = normalize
        self._device = device
        self._model = None

    @property
    def model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required for LocalEmbeddingClient. "
                    "Install with: pip install sentence-transformers"
                )
            device = self._device or _auto_device()
            print(f"[LocalEmbeddingClient] Loading {self.model_name} on {device}...")
            self._model = SentenceTransformer(self.model_name, device=device)
            dim = self._model.get_sentence_embedding_dimension()
            print(f"[LocalEmbeddingClient] Ready — dim={dim}, device={device}")
        return self._model

    def is_configured(self) -> bool:
        return True

    def embed_text(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        prepared = [t.strip() or " " for t in texts]
        vectors = self.model.encode(
            prepared,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
        )
        return [v.tolist() for v in vectors]


def build_embedding_client(
    model: str | None = None,
    batch_size: int | None = None,
) -> EmbeddingClient | LocalEmbeddingClient:
    """Factory: return a local client for known HF model IDs, API client otherwise.

    Args:
        model: Model name. HuggingFace IDs (containing '/') use LocalEmbeddingClient.
               API model names (e.g. 'text-embedding-v4') use EmbeddingClient.
               None → reads EMBEDDING_MODEL_NAME from env (defaults to API client).
        batch_size: Override default batch size.
    """
    import os
    name = model or os.getenv("EMBEDDING_MODEL_NAME", "")
    if "/" in name:
        bs = batch_size or int(os.getenv("LOCAL_EMBEDDING_BATCH_SIZE", "32"))
        return LocalEmbeddingClient(model_name=name, batch_size=bs)
    client = EmbeddingClient()
    if batch_size:
        # Can't mutate frozen config; wrap in a simple override
        object.__setattr__(client.config, "batch_size", batch_size)
    return client


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _auto_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def _normalize(vector: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0:
        return [float(v) for v in vector]
    return [float(v) / norm for v in vector]


def _iter_batches(items: Sequence[str], batch_size: int):
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

