"""Core RAG pipeline exports."""

from .dense_retrieval import DenseRetrievalStrategy
from .embedding import EmbeddingClient, SupportsEmbedding
from .generator import ChatMessage, GenerationResult, LLMGenerator
from .keyword_reranker import KeywordReranker
from .rag_chain import RAGChain, RAGResult
from .reranker import (
    available_rerank_methods,
    get_default_rerank_enabled,
    get_default_rerank_method,
    set_default_rerank_enabled,
    set_default_rerank_method,
)
from .retriever import (
    LocalVectorRetriever,
    available_retrieval_methods,
    get_default_retrieval_method,
    set_default_retrieval_method,
)
from .search_types import SearchResult, VectorDocument

__all__ = [
    "ChatMessage",
    "DenseRetrievalStrategy",
    "EmbeddingClient",
    "GenerationResult",
    "KeywordReranker",
    "LLMGenerator",
    "LocalVectorRetriever",
    "RAGChain",
    "RAGResult",
    "SearchResult",
    "SupportsEmbedding",
    "VectorDocument",
    "available_retrieval_methods",
    "available_rerank_methods",
    "get_default_rerank_enabled",
    "get_default_rerank_method",
    "get_default_retrieval_method",
    "set_default_rerank_enabled",
    "set_default_rerank_method",
    "set_default_retrieval_method",
]
