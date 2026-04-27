#!/usr/bin/env python3
"""Ingest Wikipedia JSONL into Qdrant with both dense (bge-m3) and sparse (BM25 TF) vectors."""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv
load_dotenv(_ROOT / ".env")

from document.wiki_loader import iter_articles_batch
from document.splitter import chunk_article, TextChunk
from core.embedding import build_embedding_client
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

COLLECTION = "wiki_dual"
DENSE_DIM = 1024
EMBED_BATCH = 64
UPSERT_BATCH = 64
ARTICLE_BATCH = 50
SPARSE_VOCAB_SIZE = 100_000  # hash space for token IDs
SPARSE_TOP_K = 128           # max non-zero entries per sparse vector


def _sparse_vector(text: str) -> tuple[list[int], list[float]]:
    """TF-based sparse vector: token hash → normalized term frequency."""
    tokens = text.lower().split()
    if not tokens:
        return [0], [0.0]
    tf = Counter(tokens)
    total = sum(tf.values())
    # Take top-K by frequency, hash to fixed vocab space
    top = tf.most_common(SPARSE_TOP_K)
    seen: dict[int, float] = {}
    for tok, count in top:
        idx = hash(tok) % SPARSE_VOCAB_SIZE
        seen[idx] = seen.get(idx, 0.0) + count / total
    indices = list(seen.keys())
    values  = list(seen.values())
    return indices, values


def _chunk_id_to_qdrant(chunk_id: str) -> int:
    return abs(hash(chunk_id)) % (2 ** 63)


def ensure_collection(client: QdrantClient) -> None:
    existing = {c.name for c in client.get_collections().collections}
    if COLLECTION in existing:
        print(f"Collection '{COLLECTION}' already exists, upserting into it.")
        return
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config={
            "dense": qm.VectorParams(size=DENSE_DIM, distance=qm.Distance.COSINE),
        },
        sparse_vectors_config={
            "sparse": qm.SparseVectorParams(
                index=qm.SparseIndexParams(on_disk=False)
            )
        },
    )
    print(f"Created collection '{COLLECTION}'")


def ingest(jsonl_path: Path, skip: int = 0, max_articles: int | None = None) -> None:
    client = QdrantClient(host="localhost", port=6333)
    ensure_collection(client)

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    embedding_client = build_embedding_client(model="BAAI/bge-m3", batch_size=EMBED_BATCH)

    total_articles = 0
    total_chunks = 0
    t0 = time.time()
    pending: list[TextChunk] = []

    def flush() -> None:
        nonlocal total_chunks
        if not pending:
            return

        texts = [c.text for c in pending]
        dense_vecs = embedding_client.embed_texts(texts)

        points = []
        for chunk, dvec in zip(pending, dense_vecs):
            sparse_idx, sparse_val = _sparse_vector(chunk.text)
            points.append(qm.PointStruct(
                id=_chunk_id_to_qdrant(chunk.chunk_id),
                vector={
                    "dense":  dvec,
                    "sparse": qm.SparseVector(indices=sparse_idx, values=sparse_val),
                },
                payload={
                    "text":        chunk.text,
                    "metadata":    chunk.metadata,
                    "original_id": chunk.chunk_id,
                },
            ))

        for start in range(0, len(points), UPSERT_BATCH):
            client.upsert(collection_name=COLLECTION, points=points[start:start + UPSERT_BATCH])

        total_chunks += len(pending)
        pending.clear()

    print(f"[dual] Source     : {jsonl_path}")
    print(f"[dual] Collection : {COLLECTION}")
    print(f"[dual] Skip       : {skip}")
    print()

    for batch in iter_articles_batch(
        jsonl_path,
        batch_size=ARTICLE_BATCH,
        skip=skip,
        max_articles=max_articles,
    ):
        for article in batch:
            chunks = chunk_article(article, max_tokens=512, overlap_tokens=50)
            pending.extend(chunks)
            total_articles += 1

        if pending:
            flush()

        elapsed = time.time() - t0
        rate = total_articles / max(elapsed, 1)
        print(
            f"\r[dual] articles={total_articles:,}  chunks={total_chunks:,}  "
            f"{rate:.1f} art/s",
            end="", flush=True,
        )

    flush()
    elapsed = time.time() - t0
    print(f"\n\n[dual] Done in {elapsed:.1f}s")
    print(f"[dual] Articles : {total_articles:,}")
    print(f"[dual] Chunks   : {total_chunks:,}")

    info = client.get_collection(COLLECTION)
    print(f"[dual] Qdrant points : {info.points_count:,}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Ingest Wikipedia JSONL into Qdrant with dense + sparse vectors",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--file", required=True, help="Path to Wikipedia JSONL file")
    p.add_argument("--skip", type=int, default=0, help="Skip first N lines (resume)")
    p.add_argument("--max-articles", type=int, default=None, help="Stop after N articles")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ingest(Path(args.file), skip=args.skip, max_articles=args.max_articles)
