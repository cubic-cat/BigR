"""Dense retrieval strategy with optional lexical reranking."""

from __future__ import annotations

import math
import os
import re
from typing import Any, Sequence

from .embedding import SupportsEmbedding
from .retriever import RetrievalStrategy, SearchResult, VectorDocument

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")


class DenseRetrievalStrategy(RetrievalStrategy):
    """Exact dense retrieval over local vectors plus optional keyword reranking."""

    name = "dense"

    def __init__(
        self,
        *,
        vector_weight: float | None = None,
        keyword_weight: float | None = None,
        default_rerank: bool | None = None,
    ) -> None:
        self.vector_weight = vector_weight if vector_weight is not None else _get_float(
            "DENSE_VECTOR_WEIGHT",
            0.85,
        )
        self.keyword_weight = keyword_weight if keyword_weight is not None else _get_float(
            "DENSE_KEYWORD_WEIGHT",
            0.15,
        )
        self.default_rerank = (
            default_rerank
            if default_rerank is not None
            else _get_bool("DENSE_ENABLE_RERANK", True)
        )
        self._normalize_weights()

    def search(
        self,
        *,
        query: str,
        records: Sequence[VectorDocument],
        embedding_client: SupportsEmbedding,
        top_k: int = 4,
        min_score: float | None = None,
        rerank: bool | None = None,
        **_: Any,
    ) -> list[SearchResult]:
        """Rank records by dense similarity and optional keyword overlap."""
        if not records:
            return []

        active_rerank = self.default_rerank if rerank is None else rerank
        query_vector = embedding_client.embed_text(query)
        query_tokens = tokenize_text(query)
        results: list[SearchResult] = []

        for record in records:
            vector_score = cosine_similarity(query_vector, record.vector)
            keyword_score = keyword_overlap_score(query_tokens, record.text)
            final_score = self.combine_scores(
                vector_score=vector_score,
                keyword_score=keyword_score,
                rerank=active_rerank,
            )

            if min_score is not None and final_score < min_score:
                continue

            results.append(
                SearchResult(
                    id=record.id,
                    text=record.text,
                    metadata=dict(record.metadata),
                    score=final_score,
                    vector_score=vector_score,
                    rerank_score=keyword_score if active_rerank else 0.0,
                    retrieval_method=self.name,
                    details={
                        "vector_weight": self.vector_weight,
                        "keyword_weight": self.keyword_weight if active_rerank else 0.0,
                        "keyword_score": keyword_score,
                        "rerank_enabled": active_rerank,
                    },
                )
            )

        results.sort(key=lambda item: (item.score, item.vector_score), reverse=True)
        return results[: max(top_k, 0)]

    def combine_scores(
        self,
        *,
        vector_score: float,
        keyword_score: float,
        rerank: bool,
    ) -> float:
        """Fuse dense and lexical scores into one sortable score."""
        if not rerank:
            return vector_score
        return (self.vector_weight * vector_score) + (self.keyword_weight * keyword_score)

    def _normalize_weights(self) -> None:
        total = self.vector_weight + self.keyword_weight
        if total <= 0:
            raise ValueError("Dense retrieval weights must sum to a positive value.")
        self.vector_weight /= total
        self.keyword_weight /= total


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


def tokenize_text(text: str) -> set[str]:
    """Tokenize English words, digits, underscores, and single Chinese characters."""
    return {token.lower() for token in TOKEN_PATTERN.findall(text or "")}


def keyword_overlap_score(query_tokens: set[str], text: str) -> float:
    """Compute a lightweight lexical overlap score for reranking."""
    if not query_tokens:
        return 0.0

    candidate_tokens = tokenize_text(text)
    if not candidate_tokens:
        return 0.0

    overlap = len(query_tokens & candidate_tokens)
    return overlap / len(query_tokens)


def _get_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value in (None, ""):
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _get_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value in (None, ""):
        return default
    return float(value)
