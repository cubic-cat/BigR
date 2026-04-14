"""Core RAG pipeline exports."""

from .dense_retrieval import DenseRetrievalStrategy
from .embedding import EmbeddingClient, SupportsEmbedding
from .generator import ChatMessage, GenerationResult, LLMGenerator
from .rag_chain import RAGChain, RAGResult
from .retriever import (
    LocalVectorRetriever,
    SearchResult,
    VectorDocument,
    available_retrieval_methods,
    get_default_retrieval_method,
    set_default_retrieval_method,
)

__all__ = [
    "ChatMessage",
    "DenseRetrievalStrategy",
    "EmbeddingClient",
    "GenerationResult",
    "LLMGenerator",
    "LocalVectorRetriever",
    "RAGChain",
    "RAGResult",
    "SearchResult",
    "SupportsEmbedding",
    "VectorDocument",
    "available_retrieval_methods",
    "get_default_retrieval_method",
    "set_default_retrieval_method",
]
