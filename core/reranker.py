"""Pluggable reranker registry and configuration."""

from __future__ import annotations

import importlib
import os
from typing import Any, Callable, Protocol, Sequence

from .search_types import SearchResult


def _get_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value in (None, ""):
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _get_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value in (None, ""):
        return default
    return int(value)


DEFAULT_RERANK_ENABLED = _get_bool("RERANK_ENABLED", True)
DEFAULT_RERANK_METHOD = os.getenv("RERANK_METHOD", "keyword").strip().lower() or "keyword"
DEFAULT_RERANK_CANDIDATE_TOP_K = _get_int("RERANK_CANDIDATE_TOP_K", 10)

RerankerFactory = Callable[[], "RerankStrategy"]
_RERANKER_FACTORIES: dict[str, RerankerFactory] = {}
_LAZY_RERANKER_FACTORIES: dict[str, tuple[str, str]] = {}


class RerankStrategy(Protocol):
    """Protocol implemented by each reranker module."""

    name: str

    def rerank(
        self,
        *,
        query: str,
        results: Sequence[SearchResult],
        top_k: int = 4,
        min_score: float | None = None,
        **kwargs: Any,
    ) -> list[SearchResult]:
        """Return reranked results."""


def register_reranker(name: str, factory: RerankerFactory) -> None:
    """Register a ready-to-use reranker factory."""
    normalized_name = _normalize_rerank_method(name)
    _RERANKER_FACTORIES[normalized_name] = factory


def register_lazy_reranker(
    name: str,
    *,
    module_path: str,
    class_name: str,
) -> None:
    """Register a reranker that will be imported on demand."""
    normalized_name = _normalize_rerank_method(name)
    _LAZY_RERANKER_FACTORIES[normalized_name] = (module_path, class_name)


def available_rerank_methods() -> list[str]:
    """Return all registered rerank method names."""
    return sorted(set(_RERANKER_FACTORIES) | set(_LAZY_RERANKER_FACTORIES))


def get_default_rerank_method() -> str:
    """Return the current module-level default rerank method."""
    return DEFAULT_RERANK_METHOD


def set_default_rerank_method(name: str) -> str:
    """Update the module-level default rerank method."""
    global DEFAULT_RERANK_METHOD
    DEFAULT_RERANK_METHOD = _normalize_rerank_method(name)
    return DEFAULT_RERANK_METHOD


def get_default_rerank_enabled() -> bool:
    """Return whether reranking is enabled by default."""
    return DEFAULT_RERANK_ENABLED


def set_default_rerank_enabled(enabled: bool) -> bool:
    """Update the module-level default rerank enabled flag."""
    global DEFAULT_RERANK_ENABLED
    DEFAULT_RERANK_ENABLED = bool(enabled)
    return DEFAULT_RERANK_ENABLED


def get_default_rerank_candidate_top_k() -> int:
    """Return the default candidate count used before reranking."""
    return max(DEFAULT_RERANK_CANDIDATE_TOP_K, 1)


def resolve_reranker(name: str) -> RerankStrategy:
    """Instantiate the reranker matching the given name."""
    normalized_name = _normalize_rerank_method(name)

    if normalized_name in _RERANKER_FACTORIES:
        return _RERANKER_FACTORIES[normalized_name]()

    if normalized_name not in _LAZY_RERANKER_FACTORIES:
        available = ", ".join(available_rerank_methods()) or "none"
        raise ValueError(
            f"Unknown rerank method: {normalized_name}. "
            f"Available methods: {available}."
        )

    module_path, class_name = _LAZY_RERANKER_FACTORIES[normalized_name]
    module = importlib.import_module(module_path)
    reranker_cls = getattr(module, class_name)
    register_reranker(normalized_name, reranker_cls)
    return resolve_reranker(normalized_name)


def _normalize_rerank_method(name: str) -> str:
    normalized = (name or "").strip().lower()
    if not normalized:
        raise ValueError("Rerank method cannot be empty.")
    return normalized


register_lazy_reranker(
    "keyword",
    module_path=f"{__package__}.keyword_reranker",
    class_name="KeywordReranker",
)
