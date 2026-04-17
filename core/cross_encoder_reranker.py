"""Cross-Encoder reranker backed by sentence-transformers."""

from __future__ import annotations

import os
from typing import Any, Sequence

from .reranker import RerankStrategy
from .search_types import SearchResult

try:
    from sentence_transformers import CrossEncoder
except ImportError:  # pragma: no cover
    CrossEncoder = None


def _get_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value in (None, ""):
        return default
    return float(value)


def _get_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value in (None, ""):
        return default
    return int(value)


class CrossEncoderReranker(RerankStrategy):
    """Rerank candidates using a cross-encoder model."""

    name = "cross_encoder"

    def __init__(
        self,
        *,
        model_name: str | None = None,
        device: str | None = None,
        batch_size: int | None = None,
        max_length: int | None = None,
    ) -> None:
        if CrossEncoder is None:
            raise ImportError(
                "sentence-transformers is required for CrossEncoderReranker. "
                "Install it with `pip install sentence-transformers`."
            )

        self.model_name = (
            model_name
            if model_name is not None
            else os.getenv("CROSS_ENCODER_MODEL", "BAAI/bge-reranker-base")
        )
        self.device = (
            device
            if device is not None
            else os.getenv("CROSS_ENCODER_DEVICE", None)
        )
        self.batch_size = batch_size if batch_size is not None else _get_int(
            "CROSS_ENCODER_BATCH_SIZE",
            8,
        )
        self.max_length = max_length if max_length is not None else _get_int(
            "CROSS_ENCODER_MAX_LENGTH",
            512,
        )
        self._model: CrossEncoder | None = None

    @property
    def model(self) -> CrossEncoder:
        """Lazy-load the cross-encoder model."""
        if self._model is None:
            kwargs: dict[str, Any] = {"max_length": self.max_length}
            if self.device:
                kwargs["device"] = self.device
            self._model = CrossEncoder(self.model_name, **kwargs)
        return self._model

    def rerank(
        self,
        *,
        query: str,
        results: Sequence[SearchResult],
        top_k: int = 4,
        min_score: float | None = None,
        **_: Any,
    ) -> list[SearchResult]:
        """Rerank candidates by cross-encoder relevance scores."""
        if not results:
            return []

        pairs = [(query, result.text) for result in results]
        scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        reranked: list[SearchResult] = []
        for result, score in zip(results, scores, strict=True):
            score = float(score)
            if min_score is not None and score < min_score:
                continue

            details = dict(result.details)
            details.update(
                {
                    "cross_encoder_score": score,
                    "cross_encoder_model": self.model_name,
                    "rerank_enabled": True,
                }
            )
            reranked.append(
                SearchResult(
                    id=result.id,
                    text=result.text,
                    metadata=dict(result.metadata),
                    score=score,
                    retrieval_score=result.retrieval_score,
                    vector_score=result.vector_score,
                    rerank_score=score,
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
