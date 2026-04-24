"""Rerank experiment adapters.

This module provides a small, experiment-focused abstraction over existing
rerank implementations so we can run controlled ablation studies:
- NoRerank (baseline)
- KeywordReranker (existing implementation)
- CrossEncoderReranker (existing implementation)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from .reranker import resolve_reranker
from .search_types import SearchResult


class RerankStrategy(Protocol):
    """Unified rerank interface used by experiment scripts."""

    name: str

    def rerank(
        self,
        query: str,
        docs: Sequence[SearchResult],
        top_k: int,
    ) -> list[SearchResult]:
        """Rerank *docs* and return top-k results."""


@dataclass(slots=True)
class NoRerank:
    """Baseline strategy that keeps retrieval order unchanged."""

    name: str = "none"

    def rerank(
        self,
        query: str,
        docs: Sequence[SearchResult],
        top_k: int,
    ) -> list[SearchResult]:
        _ = query
        return list(docs)[: max(top_k, 0)]


@dataclass(slots=True)
class KeywordRerankAdapter:
    """Adapter around the existing keyword reranker implementation."""

    name: str = "keyword"

    def rerank(
        self,
        query: str,
        docs: Sequence[SearchResult],
        top_k: int,
    ) -> list[SearchResult]:
        strategy = resolve_reranker("keyword")
        return strategy.rerank(query=query, results=docs, top_k=top_k)


@dataclass(slots=True)
class CrossEncoderRerankAdapter:
    """Adapter around the existing cross-encoder reranker implementation."""

    name: str = "cross_encoder"

    def rerank(
        self,
        query: str,
        docs: Sequence[SearchResult],
        top_k: int,
    ) -> list[SearchResult]:
        strategy = resolve_reranker("cross_encoder")
        return strategy.rerank(query=query, results=docs, top_k=top_k)


def create_rerank_strategy(name: str | None) -> RerankStrategy:
    """Factory for experiment rerank strategies."""
    normalized = (name or "none").strip().lower()
    if normalized in {"none", "no_rerank", "off"}:
        return NoRerank()
    if normalized == "keyword":
        return KeywordRerankAdapter()
    if normalized in {"cross_encoder", "cross-encoder"}:
        return CrossEncoderRerankAdapter()
    raise ValueError(
        "Unknown rerank strategy: "
        f"{normalized}. Available: none, keyword, cross_encoder."
    )

