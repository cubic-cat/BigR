"""Hybrid (RRF) retrieval strategy combining dense and sparse signals."""

from __future__ import annotations

from typing import Any, Sequence

from .dense_retrieval import DenseRetrievalStrategy, cosine_similarity
from .embedding import SupportsEmbedding
from .retriever import RetrievalStrategy
from .search_types import SearchResult, VectorDocument
from .sparse_retrieval import SparseRetrievalStrategy


class HybridRetrievalStrategy(RetrievalStrategy):
    """Reciprocal Rank Fusion of dense and sparse retrieval results."""

    name = "hybrid"

    def __init__(self, k: float = 20.0) -> None:
        self.k = k
        self._dense = DenseRetrievalStrategy()
        self._sparse = SparseRetrievalStrategy()

    def search(
        self,
        *,
        query: str,
        records: Sequence[VectorDocument],
        embedding_client: SupportsEmbedding,
        top_k: int = 4,
        min_score: float | None = None,
        **_: Any,
    ) -> list[SearchResult]:
        """Rank records by fusing dense and sparse results via RRF."""
        if not records:
            return []

        dense_results = self._dense.search(
            query=query,
            records=records,
            embedding_client=embedding_client,
            top_k=len(records),
        )
        sparse_results = self._sparse.search(
            query=query,
            records=records,
            embedding_client=embedding_client,
            top_k=len(records),
        )

        rrf_scores: dict[str, float] = {}
        dense_score_map: dict[str, float] = {}
        sparse_score_map: dict[str, float] = {}

        for rank, result in enumerate(dense_results, start=1):
            rrf_scores[result.id] = rrf_scores.get(result.id, 0.0) + 1.0 / (self.k + rank)
            dense_score_map[result.id] = result.score

        for rank, result in enumerate(sparse_results, start=1):
            rrf_scores[result.id] = rrf_scores.get(result.id, 0.0) + 1.0 / (self.k + rank)
            sparse_score_map[result.id] = result.score

        record_map = {record.id: record for record in records}
        results: list[SearchResult] = []

        for doc_id, rrf_score in rrf_scores.items():
            if min_score is not None and rrf_score < min_score:
                continue

            record = record_map[doc_id]
            results.append(
                SearchResult(
                    id=record.id,
                    text=record.text,
                    metadata=dict(record.metadata),
                    score=rrf_score,
                    retrieval_score=rrf_score,
                    vector_score=dense_score_map.get(doc_id, 0.0),
                    retrieval_method=self.name,
                    rerank_method="",
                    details={
                        "rrf_k": self.k,
                        "dense_score": dense_score_map.get(doc_id, 0.0),
                        "sparse_score": sparse_score_map.get(doc_id, 0.0),
                        "rerank_enabled": False,
                    },
                )
            )

        results.sort(key=lambda item: item.score, reverse=True)
        return results[: max(top_k, 0)]
