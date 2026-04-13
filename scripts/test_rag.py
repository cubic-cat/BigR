"""Test retrieval quality against the local vector store."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core import LocalVectorRetriever


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run similarity search against the local vector store.",
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

    query = input("请输入检索问题: ").strip()
    if not query:
        raise ValueError("Query cannot be empty.")
    return query


def render_text(text: str, *, full_text: bool, preview_chars: int) -> str:
    normalized = " ".join(text.split())
    if full_text or len(normalized) <= preview_chars:
        return normalized
    return f"{normalized[:preview_chars].rstrip()}..."


def main() -> None:
    args = build_parser().parse_args()
    retriever = LocalVectorRetriever()

    ensure_vector_store_ready(retriever)

    if not retriever.embedding_client.is_configured():
        raise ValueError(
            "Embedding API key is missing. Set EMBEDDING_API_KEY or the "
            "provider-specific key in `.env` before running retrieval tests."
        )

    query = get_query_text(args.query)
    results = retriever.similarity_search(
        query,
        top_k=args.top_k,
        min_score=args.min_score,
        rerank=not args.no_rerank,
    )

    stats = retriever.stats()
    print("=== Retrieval Request ===")
    print(f"query: {query}")
    print(f"collection: {stats['collection_name']}")
    print(f"store_path: {stats['store_path']}")
    print(f"document_count: {stats['document_count']}")
    print(f"vector_dimensions: {stats['dimensions']}")
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
        print(f"score: {result.score:.6f}")
        print(f"vector_score: {result.vector_score:.6f}")
        print(f"rerank_score: {result.rerank_score:.6f}")
        print(f"text: {render_text(result.text, full_text=args.full_text, preview_chars=args.preview_chars)}")
        print()


if __name__ == "__main__":
    main()
