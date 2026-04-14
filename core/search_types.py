"""Shared search result types used by retrieval and reranking layers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class VectorDocument:
    """A text chunk plus its dense vector representation."""

    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    vector: list[float] = field(default_factory=list)


@dataclass(slots=True)
class SearchResult:
    """A retrieved chunk with retrieval and reranking metadata."""

    id: str
    text: str
    metadata: dict[str, Any]
    score: float
    retrieval_score: float = 0.0
    vector_score: float = 0.0
    rerank_score: float = 0.0
    retrieval_method: str = ""
    rerank_method: str = ""
    details: dict[str, Any] = field(default_factory=dict)
