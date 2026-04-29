"""End-to-end RAG chain: retrieve context, then generate an answer.

Backend selection
-----------------
The retriever backend is chosen by the VECTOR_DB_PROVIDER environment variable:
  - "local"  (default) : JSON file store, no external dependencies
  - "qdrant"           : Qdrant vector database (requires qdrant-client)

You can also pass a retriever instance directly to RAGChain(..., retriever=...).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from document.loader import DEFAULT_PROCESSED_DIR, load_processed_documents

from .generator import ChatMessage, GenerationResult, LLMGenerator
from .query_enhancer import create_query_enhancer
from .reranker import (
    get_default_rerank_enabled,
    get_default_rerank_method,
    resolve_reranker,
)
from .retriever import LocalVectorRetriever, SearchResult, VectorDocument


def _build_default_retriever():
    """Instantiate the retriever selected by VECTOR_DB_PROVIDER."""
    provider = os.getenv("VECTOR_DB_PROVIDER", "local").strip().lower()
    if provider == "qdrant":
        from .qdrant_retriever import QdrantRetriever
        return QdrantRetriever()
    return LocalVectorRetriever()


@dataclass(slots=True)
class RAGResult:
    """Output of a full RAG query."""

    query: str
    answer: str
    retrieved_documents: list[SearchResult] = field(default_factory=list)
    context: str = ""
    generation: GenerationResult | None = None


class RAGChain:
    """Compose retrieval and generation into a single callable chain."""

    def __init__(
        self,
        retriever=None,
        generator: LLMGenerator | None = None,
        knowledge_base_dir: str | Path = DEFAULT_PROCESSED_DIR,
    ) -> None:
        self.retriever = retriever if retriever is not None else _build_default_retriever()
        self.generator = generator or LLMGenerator()
        self.knowledge_base_dir = Path(knowledge_base_dir)

    def load_processed_documents(
        self,
        source_dir: str | Path | None = None,
    ) -> list[dict[str, Any]]:
        """Load processed files from disk and convert them to retriever records."""
        documents = load_processed_documents(source_dir or self.knowledge_base_dir)
        return [document.to_record() for document in documents]

    def build_knowledge_base(
        self,
        *,
        source_dir: str | Path | None = None,
        replace: bool = True,
        persist: bool = True,
    ) -> list[VectorDocument]:
        """Build or refresh the local vector store from `data/processed`."""
        records = self.load_processed_documents(source_dir)
        if not records:
            raise ValueError(
                f"No processed documents found in {Path(source_dir or self.knowledge_base_dir)}."
            )
        return self.add_documents(records, replace=replace, persist=persist)

    def add_documents(
        self,
        documents: Sequence[str | Mapping[str, Any] | Any],
        *,
        replace: bool = False,
        persist: bool = True,
    ) -> list[VectorDocument]:
        """Add documents into the local vector store."""
        if replace:
            return self.retriever.replace_documents(documents, persist=persist)
        return self.retriever.add_documents(documents, persist=persist)

    def search(
        self,
        query: str,
        *,
        top_k: int = 4,
        min_score: float | None = None,
        rerank: bool | None = None,
        retrieval_method: str | None = None,
        rerank_method: str | None = None,
        refresh_from_processed: bool = False,
        enable_query_enhancer: bool = False,
        query_enhancer_type: str = "identity",
        query_merge_method: str = "rrf",
    ) -> list[SearchResult]:
        """Expose retrieval without generation."""
        self.ensure_knowledge_base(refresh_from_processed=refresh_from_processed)
        if not enable_query_enhancer:
            return self.retriever.similarity_search(
                query,
                top_k=top_k,
                min_score=min_score,
                rerank=rerank,
                retrieval_method=retrieval_method,
                rerank_method=rerank_method,
            )

        enhancer = create_query_enhancer(query_enhancer_type)
        enhanced_queries = enhancer.enhance(query)
        if not enhanced_queries:
            enhanced_queries = [query]

        merged_results = self._multi_query_search(
            enhanced_queries,
            top_k=top_k,
            min_score=min_score,
            rerank=rerank,
            retrieval_method=retrieval_method,
            rerank_method=rerank_method,
            merge_method=query_merge_method,
            original_query=query,
        )
        return merged_results

    def _multi_query_search(
        self,
        enhanced_queries: Sequence[str],
        *,
        top_k: int,
        min_score: float | None,
        rerank: bool | None,
        retrieval_method: str | None,
        rerank_method: str | None,
        merge_method: str,
        original_query: str,
    ) -> list[SearchResult]:
        """Run retrieval for multiple queries then merge results."""
        per_query_results: list[list[SearchResult]] = []
        for enhanced_query in enhanced_queries:
            results = self.retriever.similarity_search(
                enhanced_query,
                top_k=max(top_k * 2, top_k),
                min_score=None,
                rerank=False,
                retrieval_method=retrieval_method,
                rerank_method=rerank_method,
            )
            per_query_results.append(results)

        merged = self._merge_multi_query_results(
            per_query_results,
            top_k=max(top_k * 2, top_k),
            merge_method=merge_method,
        )
        if min_score is not None:
            merged = [item for item in merged if item.score >= min_score]

        active_rerank = get_default_rerank_enabled() if rerank is None else rerank
        if not active_rerank:
            return merged[: max(top_k, 0)]

        method = (rerank_method or get_default_rerank_method()).strip().lower()
        strategy = resolve_reranker(method)
        return strategy.rerank(
            query=original_query,
            results=merged,
            top_k=top_k,
            min_score=min_score,
        )

    @staticmethod
    def _merge_multi_query_results(
        result_groups: Sequence[Sequence[SearchResult]],
        *,
        top_k: int,
        merge_method: str = "rrf",
        rrf_k: float = 20.0,
    ) -> list[SearchResult]:
        """Merge multi-query retrieval results via RRF or weighted score."""
        method = (merge_method or "rrf").strip().lower()
        if method not in {"rrf", "weighted"}:
            raise ValueError("query merge method must be `rrf` or `weighted`.")

        scores: dict[str, float] = {}
        best_result_by_id: dict[str, SearchResult] = {}

        for group in result_groups:
            for rank, item in enumerate(group, start=1):
                if method == "rrf":
                    score = 1.0 / (rrf_k + rank)
                else:
                    score = item.score
                scores[item.id] = scores.get(item.id, 0.0) + score
                current = best_result_by_id.get(item.id)
                if current is None or item.score > current.score:
                    best_result_by_id[item.id] = item

        merged: list[SearchResult] = []
        for doc_id, aggregate_score in scores.items():
            base = best_result_by_id[doc_id]
            merged.append(
                SearchResult(
                    id=base.id,
                    text=base.text,
                    metadata=dict(base.metadata),
                    score=float(aggregate_score),
                    retrieval_score=float(aggregate_score),
                    vector_score=base.vector_score,
                    rerank_score=base.rerank_score,
                    retrieval_method=base.retrieval_method,
                    rerank_method=base.rerank_method,
                    details={
                        **base.details,
                        "multi_query_merged": True,
                        "query_merge_method": method,
                    },
                )
            )

        merged.sort(
            key=lambda item: (item.score, item.retrieval_score, item.vector_score),
            reverse=True,
        )
        return merged[: max(top_k, 0)]

    def ask(
        self,
        query: str,
        *,
        top_k: int = 4,
        min_score: float | None = None,
        rerank: bool | None = None,
        retrieval_method: str | None = None,
        rerank_method: str | None = None,
        max_context_chars: int = 4000,
        system_prompt: str | None = None,
        history: Sequence[ChatMessage | Mapping[str, str]] | None = None,
        refresh_from_processed: bool = False,
        enable_query_enhancer: bool = False,
        query_enhancer_type: str = "identity",
        query_merge_method: str = "rrf",
    ) -> RAGResult:
        """Run retrieval and pass the retrieved context to the LLM."""
        retrieved_documents = self.search(
            query,
            top_k=top_k,
            min_score=min_score,
            rerank=rerank,
            retrieval_method=retrieval_method,
            rerank_method=rerank_method,
            refresh_from_processed=refresh_from_processed,
            enable_query_enhancer=enable_query_enhancer,
            query_enhancer_type=query_enhancer_type,
            query_merge_method=query_merge_method,
        )
        context = self.retriever.build_context(
            retrieved_documents,
            max_chars=max_context_chars,
        )
        generation = self.generator.generate(
            query,
            context,
            system_prompt=system_prompt,
            history=history,
        )
        return RAGResult(
            query=query,
            answer=generation.answer,
            retrieved_documents=retrieved_documents,
            context=context,
            generation=generation,
        )

    def __call__(self, query: str, **kwargs: Any) -> RAGResult:
        """Call `ask` directly for convenience."""
        return self.ask(query, **kwargs)

    def ensure_knowledge_base(self, *, refresh_from_processed: bool = False) -> None:
        """Ensure the local vector store exists, defaulting to `data/processed`."""
        existing_documents = self.retriever.list_documents()
        if existing_documents and not refresh_from_processed:
            return
        self.build_knowledge_base(replace=True, persist=True)
