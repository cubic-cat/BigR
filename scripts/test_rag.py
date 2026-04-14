"""Test retrieval quality against the local vector store."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core import LocalVectorRetriever, RAGChain


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run retrieval-only or full-chain RAG against the local vector store.",
    )
    parser.add_argument(
        "-q",
        "--query",
        help="Query text used for retrieval. If omitted, the script will prompt for input.",
    )
    parser.add_argument(
        "-k",
        "--top-k",
        type=int,
        default=4,
        help="Number of retrieved chunks to return.",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=None,
        help="Filter out results below this final score.",
    )
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Disable the lightweight keyword reranking step.",
    )
    parser.add_argument(
        "--full-text",
        action="store_true",
        help="Print the full retrieved text instead of a preview.",
    )
    parser.add_argument(
        "--preview-chars",
        type=int,
        default=300,
        help="Preview length when --full-text is not set.",
    )
    parser.add_argument(
        "--retrieval-method",
        default=None,
        help="Retrieval method to use, for example: dense.",
    )
    parser.add_argument(
        "--list-methods",
        action="store_true",
        help="List available retrieval methods and exit.",
    )
    parser.add_argument(
        "--full-chain",
        action="store_true",
        help="Run the complete RAG chain: retrieval plus LLM answer generation.",
    )
    parser.add_argument(
        "--show-context",
        action="store_true",
        help="Print the assembled retrieval context block.",
    )
    parser.add_argument(
        "--max-context-chars",
        type=int,
        default=4000,
        help="Maximum context length passed into the full chain.",
    )
    parser.add_argument(
        "--refresh-from-processed",
        action="store_true",
        help="Rebuild the local vector store from data/processed before search.",
    )
    return parser


def ensure_vector_store_ready(retriever: LocalVectorRetriever) -> None:
    stats = retriever.stats()
    store_path = Path(stats["store_path"])
    if not store_path.exists() or stats["document_count"] == 0:
        raise FileNotFoundError(
            "Local vector store is empty. Run `python scripts/build_kb.py` first "
            "to embed documents from `data/processed`."
        )


def get_query_text(initial_query: str | None) -> str:
    if initial_query and initial_query.strip():
        return initial_query.strip()

    query = input("Enter retrieval query: ").strip()
    if not query:
        raise ValueError("Query cannot be empty.")
    return query


def render_text(text: str, *, full_text: bool, preview_chars: int) -> str:
    normalized = " ".join(text.split())
    if full_text or len(normalized) <= preview_chars:
        return normalized
    return f"{normalized[:preview_chars].rstrip()}..."


def print_available_methods() -> None:
    methods = LocalVectorRetriever.available_retrieval_methods()
    print("=== Available Retrieval Methods ===")
    for method in methods:
        print(method)


def run_retrieval_only(args: argparse.Namespace, query: str) -> None:
    retriever = LocalVectorRetriever(retrieval_method=args.retrieval_method)

    if args.refresh_from_processed:
        chain = RAGChain(retriever=retriever)
        if not retriever.embedding_client.is_configured():
            raise ValueError(
                "Embedding API key is missing. Set EMBEDDING_API_KEY or the "
                "provider-specific key in `.env` before rebuilding the knowledge base."
            )
        chain.build_knowledge_base(replace=True, persist=True)

    ensure_vector_store_ready(retriever)

    if not retriever.embedding_client.is_configured():
        raise ValueError(
            "Embedding API key is missing. Set EMBEDDING_API_KEY or the "
            "provider-specific key in `.env` before running retrieval tests."
        )

    results = retriever.similarity_search(
        query,
        top_k=args.top_k,
        min_score=args.min_score,
        rerank=False if args.no_rerank else None,
        retrieval_method=args.retrieval_method,
    )

    stats = retriever.stats()
    print("=== Retrieval Request ===")
    print(f"query: {query}")
    print(f"collection: {stats['collection_name']}")
    print(f"store_path: {stats['store_path']}")
    print(f"document_count: {stats['document_count']}")
    print(f"vector_dimensions: {stats['dimensions']}")
    print(f"retrieval_method: {stats['retrieval_method']}")
    print(f"available_methods: {', '.join(stats['available_retrieval_methods'])}")
    print(f"top_k: {args.top_k}")
    print(f"rerank: {not args.no_rerank}")
    print()

    if not results:
        print("No retrieval results matched the current filters.")
        return

    print(f"=== Retrieval Results ({len(results)}) ===")
    for rank, result in enumerate(results, start=1):
        source = (
            result.metadata.get("source")
            or result.metadata.get("file_name")
            or result.metadata.get("title")
            or result.id
        )
        print(f"[Rank {rank}]")
        print(f"id: {result.id}")
        print(f"source: {source}")
        print(f"retrieval_method: {result.retrieval_method}")
        print(f"score: {result.score:.6f}")
        print(f"vector_score: {result.vector_score:.6f}")
        print(f"rerank_score: {result.rerank_score:.6f}")
        print(
            f"text: {render_text(result.text, full_text=args.full_text, preview_chars=args.preview_chars)}"
        )
        print()


def run_full_chain(args: argparse.Namespace, query: str) -> None:
    retriever = LocalVectorRetriever(retrieval_method=args.retrieval_method)
    chain = RAGChain(retriever=retriever)

    if not retriever.embedding_client.is_configured():
        raise ValueError(
            "Embedding API key is missing. Set EMBEDDING_API_KEY or the "
            "provider-specific key in `.env` before running the full chain."
        )
    if not chain.generator.is_configured():
        raise ValueError(
            "LLM API key is missing. Set LLM_API_KEY or the provider-specific "
            "key in `.env` before running the full chain."
        )

    result = chain.ask(
        query,
        top_k=args.top_k,
        min_score=args.min_score,
        rerank=False if args.no_rerank else None,
        max_context_chars=args.max_context_chars,
        refresh_from_processed=args.refresh_from_processed,
    )

    stats = retriever.stats()
    print("=== Full Chain Request ===")
    print(f"query: {query}")
    print(f"retrieval_method: {stats['retrieval_method']}")
    print(f"collection: {stats['collection_name']}")
    print(f"document_count: {stats['document_count']}")
    print(f"top_k: {args.top_k}")
    print(f"rerank: {not args.no_rerank}")
    print()

    print("=== Answer ===")
    print(result.answer)
    print()

    if args.show_context:
        print("=== Context ===")
        print(result.context or "(empty)")
        print()

    print(f"=== Retrieved Documents ({len(result.retrieved_documents)}) ===")
    for rank, item in enumerate(result.retrieved_documents, start=1):
        source = (
            item.metadata.get("source")
            or item.metadata.get("file_name")
            or item.metadata.get("title")
            or item.id
        )
        print(f"[Rank {rank}]")
        print(f"id: {item.id}")
        print(f"source: {source}")
        print(f"retrieval_method: {item.retrieval_method}")
        print(f"score: {item.score:.6f}")
        print(f"vector_score: {item.vector_score:.6f}")
        print(f"rerank_score: {item.rerank_score:.6f}")
        print(
            f"text: {render_text(item.text, full_text=args.full_text, preview_chars=args.preview_chars)}"
        )
        print()


def main() -> None:
    args = build_parser().parse_args()
    if args.list_methods:
        print_available_methods()
        return

    query = get_query_text(args.query)
    if args.full_chain:
        run_full_chain(args, query)
        return
    run_retrieval_only(args, query)


if __name__ == "__main__":
    main()
