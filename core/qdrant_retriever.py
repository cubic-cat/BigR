"""Qdrant-backed vector retriever.

Implements the same public interface as LocalVectorRetriever so it can be
used as a drop-in replacement in RAGChain.  Requires `qdrant-client` package.

Dense search uses Qdrant's native vector index.
Sparse search uses BM25 client-side (same as sparse_retrieval.py) because
  Qdrant's sparse vector support requires a dedicated sparse collection — kept
  optional to avoid configuration complexity.
Hybrid search: RRF fusion of dense + sparse results.
"""

from __future__ import annotations

import os
import uuid
from typing import Any, Mapping, Sequence

from configs import VectorDBConfig, vector_db_config
from core.embedding import EmbeddingClient, SupportsEmbedding
from core.reranker import (
    available_rerank_methods,
    get_default_rerank_candidate_top_k,
    get_default_rerank_enabled,
    get_default_rerank_method,
    resolve_reranker,
)
from core.retriever import (
    _normalize_rerank_method,
    _normalize_retrieval_method,
    _resolve_candidate_top_k,
)
from core.search_types import SearchResult, VectorDocument

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qmodels
    _QDRANT_AVAILABLE = True
except ImportError:  # pragma: no cover
    _QDRANT_AVAILABLE = False
    QdrantClient = None  # type: ignore[assignment,misc]
    qmodels = None  # type: ignore[assignment]


QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_GRPC_PORT = int(os.getenv("QDRANT_GRPC_PORT", "6334"))
QDRANT_URL = os.getenv("QDRANT_URL", "")  # e.g. https://xyz.cloud.qdrant.io:6333
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
QDRANT_PREFER_GRPC = os.getenv("QDRANT_PREFER_GRPC", "false").lower() == "true"

# Embedding dimension (must match your embedding model)
QDRANT_VECTOR_SIZE = int(os.getenv("QDRANT_VECTOR_SIZE", "1024"))
QDRANT_DISTANCE = os.getenv("QDRANT_DISTANCE", "Cosine")  # Cosine | Dot | Euclid


class QdrantRetriever:
    """Qdrant-backed retriever with dense / sparse / hybrid search."""

    def __init__(
        self,
        embedding_client: SupportsEmbedding | None = None,
        config: VectorDBConfig | None = None,
        *,
        retrieval_method: str | None = None,
        rerank_method: str | None = None,
    ) -> None:
        if not _QDRANT_AVAILABLE:
            raise ImportError(
                "qdrant-client is required. Install with: pip install qdrant-client"
            )
        self.embedding_client = embedding_client or EmbeddingClient()
        self.config = config or vector_db_config
        self.retrieval_method = _normalize_retrieval_method(
            retrieval_method or os.getenv("RETRIEVAL_METHOD", "dense")
        )
        self.rerank_method = _normalize_rerank_method(
            rerank_method or get_default_rerank_method()
        )
        self._client: QdrantClient | None = None
        self._reranker_cache: dict[str, Any] = {}
        self._collection_ready = False

    # ------------------------------------------------------------------
    # Qdrant client & collection lifecycle
    # ------------------------------------------------------------------

    @property
    def client(self) -> QdrantClient:
        if self._client is None:
            kwargs: dict[str, Any] = {"prefer_grpc": QDRANT_PREFER_GRPC}
            if QDRANT_URL:
                kwargs["url"] = QDRANT_URL
            else:
                kwargs["host"] = QDRANT_HOST
                kwargs["port"] = QDRANT_PORT
                kwargs["grpc_port"] = QDRANT_GRPC_PORT
            if QDRANT_API_KEY:
                kwargs["api_key"] = QDRANT_API_KEY
            self._client = QdrantClient(**kwargs)
        return self._client

    @property
    def collection_name(self) -> str:
        return self.config.collection_name

    def _ensure_collection(self, vector_size: int | None = None) -> None:
        """Create the Qdrant collection if it doesn't exist yet."""
        if self._collection_ready:
            return
        size = vector_size or QDRANT_VECTOR_SIZE
        distance_map = {
            "cosine": qmodels.Distance.COSINE,
            "dot": qmodels.Distance.DOT,
            "euclid": qmodels.Distance.EUCLID,
        }
        distance = distance_map.get(QDRANT_DISTANCE.lower(), qmodels.Distance.COSINE)

        existing = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in existing:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=qmodels.VectorParams(size=size, distance=distance),
            )
        self._collection_ready = True

    # ------------------------------------------------------------------
    # Document ingestion
    # ------------------------------------------------------------------

    def add_documents(
        self,
        documents: Sequence[str | Mapping[str, Any] | Any],
        *,
        batch_size: int = 64,
        persist: bool = True,  # ignored — Qdrant always persists
    ) -> list[VectorDocument]:
        """Embed and upsert documents into Qdrant."""
        prepared = [_coerce_document(doc) for doc in documents]
        texts = [d.text for d in prepared]
        vectors = self.embedding_client.embed_texts(texts)

        vector_size = len(vectors[0]) if vectors else QDRANT_VECTOR_SIZE
        self._ensure_collection(vector_size)

        for doc, vec in zip(prepared, vectors):
            doc.vector = [float(v) for v in vec]

        # Upsert in batches
        for start in range(0, len(prepared), batch_size):
            batch = prepared[start : start + batch_size]
            points = [
                qmodels.PointStruct(
                    id=_doc_id_to_qdrant(doc.id),
                    vector=[float(v) for v in doc.vector],
                    payload={"text": doc.text, "metadata": doc.metadata, "original_id": doc.id},
                )
                for doc in batch
            ]
            self.client.upsert(collection_name=self.collection_name, points=points)

        return prepared

    def replace_documents(
        self,
        documents: Sequence[str | Mapping[str, Any] | Any],
        *,
        persist: bool = True,
    ) -> list[VectorDocument]:
        """Delete the collection and re-ingest all documents."""
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:  # noqa: BLE001
            pass
        self._collection_ready = False
        return self.add_documents(documents, persist=persist)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

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
        **kwargs: Any,
    ) -> list[SearchResult]:
        method = _normalize_retrieval_method(retrieval_method or self.retrieval_method)
        active_rerank = get_default_rerank_enabled() if rerank is None else rerank
        candidate_limit = _resolve_candidate_top_k(
            requested_top_k=max(top_k, 0),
            candidate_top_k=candidate_top_k,
            rerank_enabled=active_rerank,
        )

        if method == "dense":
            results = self._dense_search(query, top_k=candidate_limit, min_score=None if active_rerank else min_score)
        elif method == "sparse":
            results = self._sparse_search(query, top_k=candidate_limit, min_score=None if active_rerank else min_score)
        elif method == "hybrid":
            results = self._hybrid_search(query, top_k=candidate_limit, min_score=None if active_rerank else min_score)
        else:
            results = self._dense_search(query, top_k=candidate_limit, min_score=None if active_rerank else min_score)

        if not active_rerank:
            return results[:top_k]

        reranker = self._get_reranker(rerank_method)
        return reranker.rerank(query=query, results=results, top_k=top_k, min_score=min_score)

    def _dense_search(
        self, query: str, top_k: int, min_score: float | None
    ) -> list[SearchResult]:
        query_vector = self.embedding_client.embed_text(query)
        kwargs: dict = dict(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
            with_payload=True,
        )
        if min_score is not None:
            kwargs["score_threshold"] = min_score
        response = self.client.query_points(**kwargs)
        hits = response.points if hasattr(response, "points") else response
        results = []
        for hit in hits:
            payload = hit.payload or {}
            results.append(
                SearchResult(
                    id=str(payload.get("original_id", hit.id)),
                    text=payload.get("text", ""),
                    metadata=payload.get("metadata", {}),
                    score=float(hit.score),
                    retrieval_score=float(hit.score),
                    vector_score=float(hit.score),
                    retrieval_method="dense",
                )
            )
        return results

    def _sparse_search(
        self, query: str, top_k: int, min_score: float | None
    ) -> list[SearchResult]:
        # Fall back to client-side BM25 via the existing SparseRetrievalStrategy
        from core.sparse_retrieval import SparseRetrievalStrategy
        from core.retriever import _LAZY_STRATEGY_FACTORIES

        # Fetch candidate pool from Qdrant by dense search, then BM25 re-score
        # For a true sparse-only path we scroll and apply BM25 in memory.
        # This is acceptable at moderate scales; for full 79 GB use Qdrant's
        # native sparse vectors (qdrant_client >= 1.7) configured separately.
        records = self._scroll_all_as_documents(limit=max(top_k * 20, 200))
        strategy = SparseRetrievalStrategy()
        return strategy.search(
            query=query,
            records=records,
            embedding_client=self.embedding_client,
            top_k=top_k,
            min_score=min_score,
        )

    def _hybrid_search(
        self, query: str, top_k: int, min_score: float | None
    ) -> list[SearchResult]:
        k_rrf = 20.0
        dense_results = self._dense_search(query, top_k=top_k * 2, min_score=None)
        sparse_results = self._sparse_search(query, top_k=top_k * 2, min_score=None)

        scores: dict[str, float] = {}
        by_id: dict[str, SearchResult] = {}

        for rank, r in enumerate(dense_results, start=1):
            scores[r.id] = scores.get(r.id, 0.0) + 1.0 / (k_rrf + rank)
            by_id[r.id] = r

        for rank, r in enumerate(sparse_results, start=1):
            scores[r.id] = scores.get(r.id, 0.0) + 1.0 / (k_rrf + rank)
            if r.id not in by_id:
                by_id[r.id] = r

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        results = []
        for doc_id, rrf_score in ranked:
            base = by_id[doc_id]
            if min_score is not None and rrf_score < min_score:
                continue
            results.append(
                SearchResult(
                    id=base.id,
                    text=base.text,
                    metadata=base.metadata,
                    score=rrf_score,
                    retrieval_score=rrf_score,
                    retrieval_method="hybrid",
                )
            )
        return results

    def _scroll_all_as_documents(self, limit: int = 200) -> list[VectorDocument]:
        """Scroll Qdrant and return at most *limit* records as VectorDocument."""
        records: list[VectorDocument] = []
        offset = None
        while len(records) < limit:
            batch_size = min(64, limit - len(records))
            result, offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=batch_size,
                offset=offset,
                with_vectors=True,
                with_payload=True,
            )
            if not result:
                break
            for point in result:
                payload = point.payload or {}
                records.append(
                    VectorDocument(
                        id=str(payload.get("original_id", point.id)),
                        text=payload.get("text", ""),
                        metadata=payload.get("metadata", {}),
                        vector=list(point.vector) if point.vector else [],
                    )
                )
            if offset is None:
                break
        return records

    # ------------------------------------------------------------------
    # Context assembly (same as LocalVectorRetriever)
    # ------------------------------------------------------------------

    def build_context(
        self,
        results: Sequence[SearchResult],
        *,
        max_chars: int = 4000,
    ) -> str:
        blocks: list[str] = []
        current_chars = 0
        for index, result in enumerate(results, start=1):
            meta = result.metadata
            source = (
                meta.get("title")
                or meta.get("source")
                or meta.get("file_name")
                or result.id
            )
            section = meta.get("section", "")
            header = f"{source} — {section}" if section else source
            block = f"[{index}] {header}\n{result.text.strip()}"
            if blocks and current_chars + len(block) > max_chars:
                break
            blocks.append(block)
            current_chars += len(block) + 2
        return "\n\n".join(blocks)

    # ------------------------------------------------------------------
    # Utility methods matching LocalVectorRetriever's interface
    # ------------------------------------------------------------------

    def list_documents(self) -> list[VectorDocument]:
        return self._scroll_all_as_documents(limit=10_000)

    def clear(self) -> None:
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:  # noqa: BLE001
            pass
        self._collection_ready = False

    def stats(self) -> dict[str, Any]:
        try:
            info = self.client.get_collection(self.collection_name)
            count = info.points_count or 0
            dim = (
                info.config.params.vectors.size
                if hasattr(info.config.params.vectors, "size")
                else QDRANT_VECTOR_SIZE
            )
        except Exception:  # noqa: BLE001
            count = 0
            dim = QDRANT_VECTOR_SIZE
        return {
            "collection_name": self.collection_name,
            "backend": "qdrant",
            "host": QDRANT_URL or f"{QDRANT_HOST}:{QDRANT_PORT}",
            "document_count": count,
            "dimensions": dim,
            "retrieval_method": self.retrieval_method,
            "rerank_method": self.rerank_method,
            "rerank_enabled": get_default_rerank_enabled(),
            "available_rerank_methods": available_rerank_methods(),
        }

    def set_retrieval_method(self, name: str) -> str:
        self.retrieval_method = _normalize_retrieval_method(name)
        return self.retrieval_method

    def set_rerank_method(self, name: str) -> str:
        self.rerank_method = _normalize_rerank_method(name)
        return self.rerank_method

    @classmethod
    def available_retrieval_methods(cls) -> list[str]:
        return ["dense", "sparse", "hybrid"]

    @classmethod
    def available_rerank_methods(cls) -> list[str]:
        return available_rerank_methods()

    def _get_reranker(self, rerank_method: str | None = None) -> Any:
        method = _normalize_rerank_method(rerank_method or self.rerank_method)
        if method not in self._reranker_cache:
            self._reranker_cache[method] = resolve_reranker(method)
        return self._reranker_cache[method]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _doc_id_to_qdrant(doc_id: str) -> str | int:
    """Qdrant accepts string UUIDs or unsigned integers as point IDs."""
    try:
        val = int(doc_id)
        if val >= 0:
            return val
    except (ValueError, TypeError):
        pass
    # Ensure it's a valid UUID string; fall back to generating one
    try:
        import uuid as _uuid
        _uuid.UUID(doc_id)
        return doc_id
    except (ValueError, AttributeError):
        return uuid.uuid4().hex


def _coerce_document(document: str | Mapping[str, Any] | Any) -> VectorDocument:
    if isinstance(document, str):
        return VectorDocument(id=uuid.uuid4().hex, text=document, metadata={})
    if isinstance(document, Mapping):
        text = (
            document.get("text") or document.get("page_content") or document.get("content") or ""
        )
        metadata = dict(document.get("metadata", {}))
        doc_id = str(document.get("id") or metadata.get("id") or uuid.uuid4().hex)
        return VectorDocument(id=doc_id, text=str(text), metadata=metadata)
    text = getattr(document, "page_content", None) or getattr(document, "text", "")
    metadata = dict(getattr(document, "metadata", {}))
    doc_id = str(getattr(document, "id", None) or metadata.get("id") or uuid.uuid4().hex)
    return VectorDocument(id=doc_id, text=str(text), metadata=metadata)
