"""End-to-end RAG chain: retrieve context, then generate an answer."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from document.loader import DEFAULT_PROCESSED_DIR, load_processed_documents

from .generator import ChatMessage, GenerationResult, LLMGenerator
from .retriever import LocalVectorRetriever, SearchResult, VectorDocument


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
        retriever: LocalVectorRetriever | None = None,
        generator: LLMGenerator | None = None,
        knowledge_base_dir: str | Path = DEFAULT_PROCESSED_DIR,
    ) -> None:
        self.retriever = retriever or LocalVectorRetriever()
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
        rerank: bool = True,
        refresh_from_processed: bool = False,
    ) -> list[SearchResult]:
        """Expose retrieval without generation."""
        self.ensure_knowledge_base(refresh_from_processed=refresh_from_processed)
        return self.retriever.similarity_search(
            query,
            top_k=top_k,
            min_score=min_score,
            rerank=rerank,
        )

    def ask(
        self,
        query: str,
        *,
        top_k: int = 4,
        min_score: float | None = None,
        rerank: bool = True,
        max_context_chars: int = 4000,
        system_prompt: str | None = None,
        history: Sequence[ChatMessage | Mapping[str, str]] | None = None,
        refresh_from_processed: bool = False,
    ) -> RAGResult:
        """Run retrieval and pass the retrieved context to the LLM."""
        retrieved_documents = self.search(
            query,
            top_k=top_k,
            min_score=min_score,
            rerank=rerank,
            refresh_from_processed=refresh_from_processed,
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
