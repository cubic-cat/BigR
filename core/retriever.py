"""Local vector store plus pluggable retrieval strategies."""

from __future__ import annotations

import importlib
import json
import os
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from configs import VectorDBConfig, vector_db_config

from .embedding import EmbeddingClient, SupportsEmbedding
from .reranker import (
    available_rerank_methods,
    get_default_rerank_candidate_top_k,
    get_default_rerank_enabled,
    get_default_rerank_method,
    resolve_reranker,
)
from .search_types import SearchResult, VectorDocument

DEFAULT_RETRIEVAL_METHOD = os.getenv("RETRIEVAL_METHOD", "dense").strip().lower() or "dense"

StrategyFactory = Callable[[], "RetrievalStrategy"]
_STRATEGY_FACTORIES: dict[str, StrategyFactory] = {}
_LAZY_STRATEGY_FACTORIES: dict[str, tuple[str, str]] = {}


class RetrievalStrategy(Protocol):
    """Protocol implemented by each retrieval strategy module."""

    name: str

    def search(
        self,
        *,
        query: str,
        records: Sequence[VectorDocument],
        embedding_client: SupportsEmbedding,
        top_k: int = 4,
        min_score: float | None = None,
        **kwargs: Any,
    ) -> list[SearchResult]:
        """Return ranked search results for the current strategy."""


def register_retrieval_strategy(name: str, factory: StrategyFactory) -> None:
    """Register a ready-to-use retrieval strategy factory."""
    normalized_name = _normalize_retrieval_method(name)
    _STRATEGY_FACTORIES[normalized_name] = factory


def register_lazy_retrieval_strategy(
    name: str,
    *,
    module_path: str,
    class_name: str,
) -> None:
    """Register a retrieval strategy that will be imported on demand."""
    normalized_name = _normalize_retrieval_method(name)
    _LAZY_STRATEGY_FACTORIES[normalized_name] = (module_path, class_name)


def available_retrieval_methods() -> list[str]:
    """Return all registered retrieval method names."""
    return sorted(set(_STRATEGY_FACTORIES) | set(_LAZY_STRATEGY_FACTORIES))


def get_default_retrieval_method() -> str:
    """Return the current module-level default retrieval method."""
    return DEFAULT_RETRIEVAL_METHOD


def set_default_retrieval_method(name: str) -> str:
    """Update the module-level default retrieval method."""
    global DEFAULT_RETRIEVAL_METHOD
    DEFAULT_RETRIEVAL_METHOD = _normalize_retrieval_method(name)
    return DEFAULT_RETRIEVAL_METHOD


def resolve_retrieval_strategy(name: str) -> RetrievalStrategy:
    """Instantiate the retrieval strategy matching the given name."""
    normalized_name = _normalize_retrieval_method(name)

    if normalized_name in _STRATEGY_FACTORIES:
        return _STRATEGY_FACTORIES[normalized_name]()

    if normalized_name not in _LAZY_STRATEGY_FACTORIES:
        available = ", ".join(available_retrieval_methods()) or "none"
        raise ValueError(
            f"Unknown retrieval method: {normalized_name}. "
            f"Available methods: {available}."
        )

    module_path, class_name = _LAZY_STRATEGY_FACTORIES[normalized_name]
    module = importlib.import_module(module_path)
    strategy_cls = getattr(module, class_name)
    register_retrieval_strategy(normalized_name, strategy_cls)
    return resolve_retrieval_strategy(normalized_name)


class LocalVectorRetriever:
    """Local vector store wrapper with switchable retrieval strategies."""

    def __init__(
        self,
        embedding_client: SupportsEmbedding | None = None,
        config: VectorDBConfig | None = None,
        *,
        retrieval_method: str | None = None,
        rerank_method: str | None = None,
    ) -> None:
        self.embedding_client = embedding_client or EmbeddingClient()
        self.config = config or vector_db_config
        self.retrieval_method = _normalize_retrieval_method(
            retrieval_method or DEFAULT_RETRIEVAL_METHOD
        )
        self.rerank_method = _normalize_rerank_method(
            rerank_method or get_default_rerank_method()
        )
        self._records: list[VectorDocument] | None = None
        self._strategy_cache: dict[str, RetrievalStrategy] = {}
        self._reranker_cache: dict[str, Any] = {}

    @property
    def collection_dir(self) -> Path:
        return Path(self.config.storage_path) / self.config.collection_name

    @property
    def store_path(self) -> Path:
        return self.collection_dir / "store.json"

    @classmethod
    def available_retrieval_methods(cls) -> list[str]:
        """Expose registered retrieval methods to callers."""
        return available_retrieval_methods()

    def set_retrieval_method(self, name: str) -> str:
        """Change the default retrieval method for this retriever instance."""
        self.retrieval_method = _normalize_retrieval_method(name)
        return self.retrieval_method

    @classmethod
    def available_rerank_methods(cls) -> list[str]:
        """Expose registered rerank methods to callers."""
        return available_rerank_methods()

    def set_rerank_method(self, name: str) -> str:
        """Change the default rerank method for this retriever instance."""
        self.rerank_method = _normalize_rerank_method(name)
        return self.rerank_method

    def get_retrieval_strategy(self, retrieval_method: str | None = None) -> RetrievalStrategy:
        """Return the active retrieval strategy, caching instances per method."""
        method = _normalize_retrieval_method(retrieval_method or self.retrieval_method)
        if method not in self._strategy_cache:
            self._strategy_cache[method] = resolve_retrieval_strategy(method)
        return self._strategy_cache[method]

    def get_reranker(self, rerank_method: str | None = None) -> Any:
        """Return the active reranker, caching instances per method."""
        method = _normalize_rerank_method(rerank_method or self.rerank_method)
        if method not in self._reranker_cache:
            self._reranker_cache[method] = resolve_reranker(method)
        return self._reranker_cache[method]

    def add_documents(
        self,
        documents: Sequence[str | Mapping[str, Any] | Any],
        *,
        persist: bool = True,
    ) -> list[VectorDocument]:
        """Embed and append documents to the local store."""
        prepared = [self._coerce_document(document) for document in documents]
        vectors = self.embedding_client.embed_texts([item.text for item in prepared])

        for item, vector in zip(prepared, vectors, strict=True):
            item.vector = [float(value) for value in vector]

        records = self._load_records()
        records.extend(prepared)
        self._records = records

        if persist:
            self._persist_records(records)
        return prepared

    def replace_documents(
        self,
        documents: Sequence[str | Mapping[str, Any] | Any],
        *,
        persist: bool = True,
    ) -> list[VectorDocument]:
        """Overwrite the collection with a new document set."""
        self._records = []
        if self.store_path.exists():
            self.store_path.unlink()
        return self.add_documents(documents, persist=persist)

    def similarity_search(
        self,
        query: str,
        *,
        top_k: int = 4,
        min_score: float | None = None,
        rerank: bool | None = None,
        retrieval_method: str | None = None,
        rerank_method: str | None = None,
        candidate_top_k: int | None = None,
        **strategy_kwargs: Any,
    ) -> list[SearchResult]:
        """Search documents using the currently selected retrieval strategy."""
        records = self._load_records()
        if not records:
            return []

        active_rerank = get_default_rerank_enabled() if rerank is None else rerank
        effective_top_k = max(top_k, 0)
        candidate_limit = _resolve_candidate_top_k(
            requested_top_k=effective_top_k,
            candidate_top_k=candidate_top_k,
            rerank_enabled=active_rerank,
        )
        strategy = self.get_retrieval_strategy(retrieval_method)
        results = strategy.search(
            query=query,
            records=records,
            embedding_client=self.embedding_client,
            top_k=candidate_limit,
            min_score=None if active_rerank else min_score,
            **strategy_kwargs,
        )

        if not active_rerank:
            return results[:effective_top_k]

        reranker = self.get_reranker(rerank_method)
        return reranker.rerank(
            query=query,
            results=results,
            top_k=effective_top_k,
            min_score=min_score,
        )

    def build_context(
        self,
        results: Sequence[SearchResult],
        *,
        max_chars: int = 4000,
    ) -> str:
        """Convert retrieval results into a prompt-friendly context string."""
        blocks: list[str] = []
        current_chars = 0

        for index, result in enumerate(results, start=1):
            source = (
                result.metadata.get("source")
                or result.metadata.get("file_name")
                or result.metadata.get("title")
                or result.id
            )
            block = f"[{index}] source: {source}\n{result.text.strip()}"
            if blocks and current_chars + len(block) > max_chars:
                break
            blocks.append(block)
            current_chars += len(block) + 2

        return "\n\n".join(blocks)

    def list_documents(self) -> list[VectorDocument]:
        """Return all persisted records."""
        return list(self._load_records())

    def clear(self) -> None:
        """Delete all records in the local collection."""
        self._records = []
        if self.store_path.exists():
            self.store_path.unlink()

    def stats(self) -> dict[str, Any]:
        """Return collection statistics plus retrieval mode metadata."""
        records = self._load_records()
        dimensions = len(records[0].vector) if records and records[0].vector else 0
        return {
            "collection_name": self.config.collection_name,
            "store_path": str(self.store_path),
            "document_count": len(records),
            "dimensions": dimensions,
            "retrieval_method": self.retrieval_method,
            "available_retrieval_methods": self.available_retrieval_methods(),
            "rerank_method": self.rerank_method,
            "rerank_enabled": get_default_rerank_enabled(),
            "available_rerank_methods": self.available_rerank_methods(),
        }

    def _load_records(self) -> list[VectorDocument]:
        if self._records is not None:
            return self._records

        if not self.store_path.exists():
            self._records = []
            return self._records

        payload = json.loads(self.store_path.read_text(encoding="utf-8"))
        self._records = [
            VectorDocument(
                id=item["id"],
                text=item["text"],
                metadata=dict(item.get("metadata", {})),
                vector=[float(value) for value in item.get("vector", [])],
            )
            for item in payload.get("records", [])
        ]
        return self._records

    def _persist_records(self, records: Sequence[VectorDocument]) -> None:
        self.collection_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "collection_name": self.config.collection_name,
            "distance_metric": self.config.distance_metric,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "records": [asdict(record) for record in records],
        }
        self.store_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @staticmethod
    def _coerce_document(document: str | Mapping[str, Any] | Any) -> VectorDocument:
        if isinstance(document, str):
            return VectorDocument(
                id=uuid.uuid4().hex,
                text=document,
                metadata={},
            )

        if isinstance(document, Mapping):
            text = (
                document.get("text")
                or document.get("page_content")
                or document.get("content")
                or ""
            )
            metadata = _json_safe_dict(document.get("metadata", {}))
            document_id = str(
                document.get("id")
                or metadata.get("id")
                or uuid.uuid4().hex
            )
            return VectorDocument(
                id=document_id,
                text=str(text),
                metadata=metadata,
            )

        text = getattr(document, "page_content", None) or getattr(document, "text", "")
        metadata = _json_safe_dict(getattr(document, "metadata", {}))
        document_id = str(
            getattr(document, "id", None)
            or metadata.get("id")
            or uuid.uuid4().hex
        )
        return VectorDocument(
            id=document_id,
            text=str(text),
            metadata=metadata,
        )


def _normalize_retrieval_method(name: str) -> str:
    normalized = (name or "").strip().lower()
    if not normalized:
        raise ValueError("Retrieval method cannot be empty.")
    return normalized


def _normalize_rerank_method(name: str) -> str:
    normalized = (name or "").strip().lower()
    if not normalized:
        raise ValueError("Rerank method cannot be empty.")
    return normalized


def _resolve_candidate_top_k(
    *,
    requested_top_k: int,
    candidate_top_k: int | None,
    rerank_enabled: bool,
) -> int:
    if requested_top_k <= 0:
        return 0

    if candidate_top_k is not None:
        return max(candidate_top_k, requested_top_k)

    if not rerank_enabled:
        return requested_top_k

    return max(get_default_rerank_candidate_top_k(), requested_top_k)


def _json_safe_dict(data: Mapping[str, Any] | Any) -> dict[str, Any]:
    if not isinstance(data, Mapping):
        return {}
    return {str(key): _json_safe_value(value) for key, value in data.items()}


def _json_safe_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe_value(item) for item in value]
    return str(value)


register_lazy_retrieval_strategy(
    "dense",
    module_path=f"{__package__}.dense_retrieval",
    class_name="DenseRetrievalStrategy",
)
