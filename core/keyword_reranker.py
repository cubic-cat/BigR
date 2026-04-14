"""Keyword-based reranker that operates on retrieval candidates."""

from __future__ import annotations

import os
import re
from typing import Any, Sequence

from .reranker import RerankStrategy
from .search_types import SearchResult

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")


class KeywordReranker(RerankStrategy):
    """Rerank candidates by combining retrieval score and token overlap."""

    name = "keyword"

    def __init__(
        self,
        *,
        retrieval_weight: float | None = None,
        keyword_weight: float | None = None,
    ) -> None:
        self.retrieval_weight = retrieval_weight if retrieval_weight is not None else _get_float(
            "KEYWORD_RERANK_RETRIEVAL_WEIGHT",
            0.85,
        )
        self.keyword_weight = keyword_weight if keyword_weight is not None else _get_float(
            "KEYWORD_RERANK_KEYWORD_WEIGHT",
            0.15,
        )
        self._normalize_weights()

    def rerank(
        self,
        *,
        query: str,
        results: Sequence[SearchResult],
        top_k: int = 4,
        min_score: float | None = None,
        **_: Any,
    ) -> list[SearchResult]:
        """Apply keyword overlap reranking to retrieval candidates."""
        query_tokens = tokenize_text(query)
        reranked: list[SearchResult] = []

        for result in results:
            retrieval_score = result.retrieval_score or result.score
            keyword_score = keyword_overlap_score(query_tokens, result.text)
            final_score = self.combine_scores(
                retrieval_score=retrieval_score,
                keyword_score=keyword_score,
            )

            if min_score is not None and final_score < min_score:
                continue

            details = dict(result.details)
            details.update(
                {
                    "keyword_score": keyword_score,
                    "retrieval_weight": self.retrieval_weight,
                    "keyword_weight": self.keyword_weight,
                    "rerank_enabled": True,
                }
            )
            reranked.append(
                SearchResult(
                    id=result.id,
                    text=result.text,
                    metadata=dict(result.metadata),
                    score=final_score,
                    retrieval_score=retrieval_score,
                    vector_score=result.vector_score,
                    rerank_score=keyword_score,
                    retrieval_method=result.retrieval_method,
                    rerank_method=self.name,
                    details=details,
                )
            )

        reranked.sort(
            key=lambda item: (item.score, item.retrieval_score, item.vector_score),
            reverse=True,
        )
        return reranked[: max(top_k, 0)]

    def combine_scores(
        self,
        *,
        retrieval_score: float,
        keyword_score: float,
    ) -> float:
        """Fuse retrieval score and keyword overlap into one final score."""
        return (
            self.retrieval_weight * retrieval_score
            + self.keyword_weight * keyword_score
        )

    def _normalize_weights(self) -> None:
        total = self.retrieval_weight + self.keyword_weight
        if total <= 0:
            raise ValueError("Keyword reranker weights must sum to a positive value.")
        self.retrieval_weight /= total
        self.keyword_weight /= total


def tokenize_text(text: str) -> set[str]:
    """Tokenize English words, digits, underscores, and single Chinese characters."""
    return {token.lower() for token in TOKEN_PATTERN.findall(text or "")}


def keyword_overlap_score(query_tokens: set[str], text: str) -> float:
    """Compute token overlap ratio against the query tokens."""
    if not query_tokens:
        return 0.0

    candidate_tokens = tokenize_text(text)
    if not candidate_tokens:
        return 0.0

    overlap = len(query_tokens & candidate_tokens)
    return overlap / len(query_tokens)


def _get_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value in (None, ""):
        return default
    return float(value)
