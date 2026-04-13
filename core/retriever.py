"""Local vector retrieval with persistence, similarity search, and reranking."""

from __future__ import annotations

import json
import math
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from configs import VectorDBConfig, vector_db_config

from .embedding import EmbeddingClient, SupportsEmbedding

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")


@dataclass(slots=True)
class VectorDocument:
    """A text chunk plus its dense vector representation."""

    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    vector: list[float] = field(default_factory=list)


@dataclass(slots=True)
class SearchResult:
    """A retrieved chunk with retrieval scores."""

    id: str
    text: str
    metadata: dict[str, Any]
    score: float
    vector_score: float
    rerank_score: float


class LocalVectorRetriever:
    """Simple local vector store backed by a JSON file."""

    def __init__(
        self,
        embedding_client: SupportsEmbedding | None = None,
        config: VectorDBConfig | None = None,
    ) -> None:
        self.embedding_client = embedding_client or EmbeddingClient()
        self.config = config or vector_db_config
        self._records: list[VectorDocument] | None = None

    @property
    def collection_dir(self) -> Path:
        return Path(self.config.storage_path) / self.config.collection_name

    @property
    def store_path(self) -> Path:
        return self.collection_dir / "store.json"

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
        rerank: bool = True,
    ) -> list[SearchResult]:
        """Search by vector similarity and optionally apply lightweight reranking."""
        records = self._load_records()
        if not records:
            return []

        query_vector = self.embedding_client.embed_text(query)
        query_tokens = _tokenize(query)
        results: list[SearchResult] = []

        for record in records:
            vector_score = _cosine_similarity(query_vector, record.vector)
            rerank_score = _keyword_overlap_score(query_tokens, record.text)
            score = vector_score if not rerank else 0.85 * vector_score + 0.15 * rerank_score

            if min_score is not None and score < min_score:
                continue

            results.append(
                SearchResult(
                    id=record.id,
                    text=record.text,
                    metadata=dict(record.metadata),
                    score=score,
                    vector_score=vector_score,
                    rerank_score=rerank_score,
                )
            )

        results.sort(key=lambda item: (item.score, item.vector_score), reverse=True)
        return results[: max(top_k, 0)]

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
        """Return basic collection statistics."""
        records = self._load_records()
        dimensions = len(records[0].vector) if records and records[0].vector else 0
        return {
            "collection_name": self.config.collection_name,
            "store_path": str(self.store_path),
            "document_count": len(records),
            "dimensions": dimensions,
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
            text = document
            metadata: dict[str, Any] = {}
            document_id = uuid.uuid4().hex
            return VectorDocument(id=document_id, text=text, metadata=metadata)

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
            return VectorDocument(id=document_id, text=str(text), metadata=metadata)

        text = getattr(document, "page_content", None) or getattr(document, "text", "")
        metadata = _json_safe_dict(getattr(document, "metadata", {}))
        document_id = str(getattr(document, "id", None) or metadata.get("id") or uuid.uuid4().hex)
        return VectorDocument(id=document_id, text=str(text), metadata=metadata)


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0

    dot_product = sum(left_value * right_value for left_value, right_value in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))

    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot_product / (left_norm * right_norm)


def _tokenize(text: str) -> set[str]:
    return {token.lower() for token in TOKEN_PATTERN.findall(text or "")}


def _keyword_overlap_score(query_tokens: set[str], text: str) -> float:
    if not query_tokens:
        return 0.0

    candidate_tokens = _tokenize(text)
    if not candidate_tokens:
        return 0.0

    overlap = len(query_tokens & candidate_tokens)
    return overlap / len(query_tokens)


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
