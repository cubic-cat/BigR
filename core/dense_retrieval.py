"""Pure dense retrieval strategy."""

from __future__ import annotations

import math
from typing import Any, Sequence

from .embedding import SupportsEmbedding
from .retriever import RetrievalStrategy
from .search_types import SearchResult, VectorDocument


class DenseRetrievalStrategy(RetrievalStrategy):
    """Exact dense retrieval over local vectors."""

    name = "dense"

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
        """Rank records by dense vector similarity only."""
        if not records:
            return []

        query_vector = embedding_client.embed_text(query)
        results: list[SearchResult] = []

        for record in records:
            vector_score = cosine_similarity(query_vector, record.vector)
            if min_score is not None and vector_score < min_score:
                continue

            results.append(
                SearchResult(
                    id=record.id,
                    text=record.text,
                    metadata=dict(record.metadata),
                    score=vector_score,
                    retrieval_score=vector_score,
                    vector_score=vector_score,
                    retrieval_method=self.name,
                    rerank_method="",
                    details={
                        "rerank_enabled": False,
                    },
                )
            )

        results.sort(key=lambda item: (item.score, item.vector_score), reverse=True)
        return results[: max(top_k, 0)]


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Compute cosine similarity between two dense vectors."""
    if not left or not right or len(left) != len(right):
        return 0.0

    dot_product = sum(left_value * right_value for left_value, right_value in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))

    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot_product / (left_norm * right_norm)
