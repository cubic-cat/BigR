#!/usr/bin/env python3
"""Bulk ingestion pipeline: Wikipedia JSONL → Qdrant vector store.

Usage examples
--------------
# Ingest up to 1000 articles from a JSONL file (quick smoke test):
    python scripts/ingest_wikipedia.py --file data/raw/en_wiki.jsonl --max-articles 1000

# Ingest a full chunk file (~2.15 GB, ~500 k articles):
    python scripts/ingest_wikipedia.py --file data/raw/en_wiki.jsonl

# Resume an interrupted run (skip already-processed articles):
    python scripts/ingest_wikipedia.py --file data/raw/en_wiki.jsonl --skip 250000

# Dry run (parse + chunk, no embedding/upload):
    python scripts/ingest_wikipedia.py --file data/raw/en_wiki.jsonl --dry-run --max-articles 50

Environment variables
---------------------
Set in your .env file (see .env.example):
  VECTOR_DB_PROVIDER=qdrant
  VECTOR_DB_COLLECTION_NAME=wikipedia_en
  QDRANT_HOST=localhost
  QDRANT_PORT=6333
  QDRANT_VECTOR_SIZE=1024        # must match your embedding model's output dim
  EMBEDDING_PROVIDER=qwen        # or openai
  EMBEDDING_MODEL_NAME=text-embedding-v4
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Ensure project root is on the path when run as a script
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(_PROJECT_ROOT / ".env")

from document.wiki_loader import iter_articles_batch, WikiArticle
from document.splitter import chunk_article, TextChunk

try:
    from tqdm import tqdm as _tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


def _progress(iterable, total: int | None = None, desc: str = ""):
    if HAS_TQDM:
        return _tqdm(iterable, total=total, desc=desc, unit="article")
    return iterable


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Ingest Wikipedia JSONL into Qdrant",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--file", required=True, help="Path to Wikipedia JSONL file")
    p.add_argument(
        "--collection",
        default=os.getenv("VECTOR_DB_COLLECTION_NAME", "wikipedia_en"),
        help="Qdrant collection name",
    )
    p.add_argument(
        "--max-articles",
        type=int,
        default=None,
        help="Stop after N articles (useful for smoke tests)",
    )
    p.add_argument(
        "--skip",
        type=int,
        default=0,
        help="Skip the first N lines in the JSONL file (resume support)",
    )
    p.add_argument(
        "--chunk-max-tokens",
        type=int,
        default=512,
        help="Maximum tokens per chunk (approx: 1 token ≈ 4 characters)",
    )
    p.add_argument(
        "--chunk-overlap-tokens",
        type=int,
        default=50,
        help="Overlap tokens between sliding-window sub-chunks",
    )
    p.add_argument(
        "--embed-batch-size",
        type=int,
        default=10,
        help="Number of chunks per embedding API call (text-embedding-v4 max: 10)",
    )
    p.add_argument(
        "--upsert-batch-size",
        type=int,
        default=64,
        help="Number of points per Qdrant upsert call",
    )
    p.add_argument(
        "--article-batch-size",
        type=int,
        default=50,
        help="Number of articles to parse before flushing to Qdrant",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and chunk without embedding or uploading",
    )
    p.add_argument(
        "--no-abstract",
        action="store_true",
        help="Skip article abstracts (only index section text)",
    )
    return p.parse_args()


def embed_chunks(
    chunks: list[TextChunk],
    embedding_client,
    batch_size: int,
) -> list[TextChunk]:
    """Embed chunks in batches; attach vectors in-place and return chunks."""
    texts = [c.text for c in chunks]
    all_vectors: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        vecs = embedding_client.embed_texts(batch_texts)
        all_vectors.extend(vecs)
    for chunk, vec in zip(chunks, all_vectors):
        chunk.metadata["_vector"] = vec  # temporary storage; extracted below
    return chunks


def upsert_chunks(
    chunks: list[TextChunk],
    qdrant_retriever,
    upsert_batch_size: int,
) -> None:
    """Upsert pre-embedded chunks into Qdrant."""
    from core.search_types import VectorDocument

    docs: list[VectorDocument] = []
    for chunk in chunks:
        vector = chunk.metadata.pop("_vector", [])
        docs.append(
            VectorDocument(
                id=chunk.chunk_id,
                text=chunk.text,
                metadata=chunk.metadata,
                vector=vector,
            )
        )

    # Bypass re-embedding by calling the Qdrant client directly
    from qdrant_client.http import models as qmodels

    qdrant_retriever._ensure_collection(len(docs[0].vector) if docs else None)
    for start in range(0, len(docs), upsert_batch_size):
        batch = docs[start : start + upsert_batch_size]
        from core.qdrant_retriever import _doc_id_to_qdrant
        points = [
            qmodels.PointStruct(
                id=_doc_id_to_qdrant(doc.id),
                vector=doc.vector,
                payload={
                    "text": doc.text,
                    "metadata": doc.metadata,
                    "original_id": doc.id,
                },
            )
            for doc in batch
            if doc.vector  # skip any empty vectors
        ]
        if points:
            qdrant_retriever.client.upsert(
                collection_name=qdrant_retriever.collection_name,
                points=points,
            )


def main() -> None:
    args = parse_args()
    jsonl_path = Path(args.file)
    if not jsonl_path.exists():
        print(f"[ERROR] File not found: {jsonl_path}", file=sys.stderr)
        sys.exit(1)

    print(f"[ingest] Source : {jsonl_path}")
    print(f"[ingest] Collection : {args.collection}")
    print(f"[ingest] Skip lines : {args.skip}")
    print(f"[ingest] Max articles : {args.max_articles or 'unlimited'}")
    print(f"[ingest] Dry run : {args.dry_run}")
    print()

    if args.dry_run:
        print("[ingest] DRY RUN — no embeddings or uploads will be made.")

    # Build retriever (initialises Qdrant connection lazily)
    qdrant_retriever = None
    embedding_client = None
    if not args.dry_run:
        os.environ.setdefault("VECTOR_DB_PROVIDER", "qdrant")
        os.environ.setdefault("VECTOR_DB_COLLECTION_NAME", args.collection)

        from core.qdrant_retriever import QdrantRetriever
        from core.embedding import EmbeddingClient

        embedding_client = EmbeddingClient()
        if not embedding_client.is_configured():
            print("[ERROR] Embedding API key missing. Set EMBEDDING_API_KEY in .env", file=sys.stderr)
            sys.exit(1)

        qdrant_retriever = QdrantRetriever(embedding_client=embedding_client)
        cfg = qdrant_retriever.config
        from configs.vector_db_config import VectorDBConfig
        qdrant_retriever.config = VectorDBConfig(
            provider=cfg.provider,
            collection_name=args.collection,
            distance_metric=cfg.distance_metric,
            persist_directory=cfg.persist_directory,
        )

    # Counters
    total_articles = 0
    total_chunks = 0
    total_errors = 0
    t_start = time.time()

    pending_chunks: list[TextChunk] = []

    def flush(chunks: list[TextChunk]) -> None:
        nonlocal total_chunks
        if not chunks or args.dry_run:
            total_chunks += len(chunks)
            return
        try:
            embedded = embed_chunks(chunks, embedding_client, args.embed_batch_size)
            upsert_chunks(embedded, qdrant_retriever, args.upsert_batch_size)
            total_chunks += len(chunks)
        except Exception as exc:  # noqa: BLE001
            print(f"\n[WARN] flush error (skipping batch): {exc}", file=sys.stderr)

    print("[ingest] Streaming articles...")
    for article_batch in iter_articles_batch(
        jsonl_path,
        batch_size=args.article_batch_size,
        skip=args.skip,
        max_articles=args.max_articles,
    ):
        for article in article_batch:
            try:
                chunks = chunk_article(
                    article,
                    max_tokens=args.chunk_max_tokens,
                    overlap_tokens=args.chunk_overlap_tokens,
                    include_abstract=not args.no_abstract,
                )
                pending_chunks.extend(chunks)
                total_articles += 1
            except Exception as exc:  # noqa: BLE001
                total_errors += 1
                if total_errors <= 5:
                    print(f"\n[WARN] Parse error on article '{article.title}': {exc}", file=sys.stderr)

        # Flush after each article batch
        if pending_chunks:
            flush(pending_chunks)
            pending_chunks = []

        elapsed = time.time() - t_start
        rate = total_articles / max(elapsed, 1)
        print(
            f"\r[ingest] articles={total_articles:,}  chunks={total_chunks:,}  "
            f"errors={total_errors}  {rate:.1f} art/s",
            end="",
            flush=True,
        )

    # Flush any remaining
    if pending_chunks:
        flush(pending_chunks)

    elapsed = time.time() - t_start
    print(f"\n\n[ingest] Done in {elapsed:.1f}s")
    print(f"[ingest] Articles processed : {total_articles:,}")
    print(f"[ingest] Chunks ingested    : {total_chunks:,}")
    print(f"[ingest] Parse errors       : {total_errors}")

    if qdrant_retriever is not None:
        try:
            stats = qdrant_retriever.stats()
            print(f"[ingest] Qdrant points      : {stats.get('document_count', '?'):,}")
        except Exception:  # noqa: BLE001
            pass


if __name__ == "__main__":
    main()
