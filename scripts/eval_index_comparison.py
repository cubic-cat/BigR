#!/usr/bin/env python3
"""Offline evaluation: compare wiki_single (dense) vs wiki_dual (dense+sparse hybrid).

Metrics: Recall@5, MRR, Precision@5

Usage:
    python scripts/eval_index_comparison.py \
      --file C:/learning/enwiki_namespace_0/enwiki_namespace_0_0.jsonl \
      --max-articles 1000 \
      --top-k 5 \
      --output results/index_comparison.json
"""

from __future__ import annotations

import argparse
import json
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
from document.splitter import chunk_article
from core.embedding import build_embedding_client
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

SPARSE_VOCAB_SIZE = 100_000
SPARSE_TOP_K = 128


# ---------------------------------------------------------------------------
# Sparse vector (must match ingest_wikipedia_dual.py exactly)
# ---------------------------------------------------------------------------

def _sparse_vector(text: str) -> tuple[list[int], list[float]]:
    tokens = text.lower().split()
    if not tokens:
        return [0], [0.0]
    tf = Counter(tokens)
    total = sum(tf.values())
    top = tf.most_common(SPARSE_TOP_K)
    seen: dict[int, float] = {}
    for tok, count in top:
        idx = hash(tok) % SPARSE_VOCAB_SIZE
        seen[idx] = seen.get(idx, 0.0) + count / total
    return list(seen.keys()), list(seen.values())


# ---------------------------------------------------------------------------
# Retrieval helpers
# ---------------------------------------------------------------------------

def search_single(client: QdrantClient, query_vec: list[float], top_k: int) -> list[str]:
    """Dense search against wiki_single."""
    response = client.query_points(
        collection_name="wiki_single",
        query=query_vec,
        limit=top_k,
        with_payload=True,
    )
    hits = response.points if hasattr(response, "points") else response
    return [str(h.payload.get("original_id", h.id)) for h in hits]


def search_dual(client: QdrantClient, query_vec: list[float], query_text: str, top_k: int) -> list[str]:
    """Hybrid RRF search against wiki_dual (dense + sparse)."""
    sparse_idx, sparse_val = _sparse_vector(query_text)

    # Dense leg
    dense_resp = client.query_points(
        collection_name="wiki_dual",
        query=query_vec,
        using="dense",
        limit=top_k * 2,
        with_payload=True,
    )
    dense_hits = dense_resp.points if hasattr(dense_resp, "points") else dense_resp

    # Sparse leg
    sparse_resp = client.query_points(
        collection_name="wiki_dual",
        query=qm.SparseVector(indices=sparse_idx, values=sparse_val),
        using="sparse",
        limit=top_k * 2,
        with_payload=True,
    )
    sparse_hits = sparse_resp.points if hasattr(sparse_resp, "points") else sparse_resp

    # RRF fusion
    k_rrf = 20.0
    scores: dict[str, float] = {}
    by_id: dict[str, str] = {}

    for rank, h in enumerate(dense_hits, start=1):
        oid = str(h.payload.get("original_id", h.id))
        scores[oid] = scores.get(oid, 0.0) + 1.0 / (k_rrf + rank)
        by_id[oid] = oid

    for rank, h in enumerate(sparse_hits, start=1):
        oid = str(h.payload.get("original_id", h.id))
        scores[oid] = scores.get(oid, 0.0) + 1.0 / (k_rrf + rank)
        by_id[oid] = oid

    ranked = sorted(scores, key=lambda x: scores[x], reverse=True)[:top_k]
    return ranked


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def recall_at_k(retrieved: list[str], relevant: set[str]) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for r in retrieved if r in relevant)
    return hits / len(relevant)


def precision_at_k(retrieved: list[str], relevant: set[str]) -> float:
    if not retrieved:
        return 0.0
    hits = sum(1 for r in retrieved if r in relevant)
    return hits / len(retrieved)


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    for rank, r in enumerate(retrieved, start=1):
        if r in relevant:
            return 1.0 / rank
    return 0.0


# ---------------------------------------------------------------------------
# Test-set generation
# ---------------------------------------------------------------------------

def build_test_set(
    jsonl_path: Path,
    max_articles: int,
    skip: int = 0,
) -> list[dict]:
    """Generate (query, relevant_chunk_ids) pairs from article abstracts.

    For each article, we use the abstract text as the "query" and the
    chunk IDs generated from that article as the ground-truth relevant set.
    This is a straightforward proxy: a query derived from the abstract
    should retrieve chunks from the same article.
    """
    test_cases = []
    for batch in iter_articles_batch(jsonl_path, batch_size=50, skip=skip, max_articles=max_articles):
        for article in batch:
            abstract = (article.get("abstract") or "").strip()
            if not abstract or abstract.upper().startswith("REDIRECT"):
                continue
            chunks = chunk_article(article, max_tokens=512, overlap_tokens=50)
            if not chunks:
                continue
            relevant_ids = {c.chunk_id for c in chunks}
            test_cases.append({
                "query": abstract[:500],
                "article_title": article.get("name", ""),
                "relevant_ids": list(relevant_ids),
            })
        if len(test_cases) >= max_articles:
            break
    return test_cases


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def evaluate(
    jsonl_path: Path,
    max_articles: int,
    top_k: int,
    output_path: Path | None,
    skip: int = 0,
) -> None:
    client = QdrantClient(host="localhost", port=6333)

    # Verify both collections exist
    existing = {c.name for c in client.get_collections().collections}
    missing = {"wiki_single", "wiki_dual"} - existing
    if missing:
        print(f"[ERROR] Missing collections: {missing}", file=sys.stderr)
        print("Run ingestion scripts first.", file=sys.stderr)
        sys.exit(1)

    single_info = client.get_collection("wiki_single")
    dual_info = client.get_collection("wiki_dual")
    print(f"wiki_single points : {single_info.points_count:,}")
    print(f"wiki_dual   points : {dual_info.points_count:,}")
    print()

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    emb = build_embedding_client(model="BAAI/bge-m3", batch_size=64)

    print(f"Building test set from first {max_articles} articles...")
    t0 = time.time()
    test_cases = build_test_set(jsonl_path, max_articles=max_articles, skip=skip)
    print(f"Test cases: {len(test_cases)}  ({time.time() - t0:.1f}s)")
    print()

    # Embed all queries in one pass
    queries = [tc["query"] for tc in test_cases]
    print(f"Embedding {len(queries)} queries...")
    t1 = time.time()
    query_vecs = emb.embed_texts(queries)
    print(f"Embedded in {time.time() - t1:.1f}s")
    print()

    single_metrics = {"recall": [], "precision": [], "mrr": []}
    dual_metrics   = {"recall": [], "precision": [], "mrr": []}

    for i, (tc, qvec) in enumerate(zip(test_cases, query_vecs)):
        relevant = set(tc["relevant_ids"])

        s_ids = search_single(client, list(qvec), top_k)
        d_ids = search_dual(client, list(qvec), tc["query"], top_k)

        single_metrics["recall"].append(recall_at_k(s_ids, relevant))
        single_metrics["precision"].append(precision_at_k(s_ids, relevant))
        single_metrics["mrr"].append(reciprocal_rank(s_ids, relevant))

        dual_metrics["recall"].append(recall_at_k(d_ids, relevant))
        dual_metrics["precision"].append(precision_at_k(d_ids, relevant))
        dual_metrics["mrr"].append(reciprocal_rank(d_ids, relevant))

        if (i + 1) % 50 == 0:
            print(
                f"\r[eval] {i+1}/{len(test_cases)}  "
                f"single R@{top_k}={sum(single_metrics['recall'])/(i+1):.3f}  "
                f"dual R@{top_k}={sum(dual_metrics['recall'])/(i+1):.3f}",
                end="", flush=True,
            )

    print()

    def avg(lst: list[float]) -> float:
        return sum(lst) / len(lst) if lst else 0.0

    n = len(test_cases)
    results = {
        "n_test_cases": n,
        "top_k": top_k,
        "wiki_single": {
            f"Recall@{top_k}":    round(avg(single_metrics["recall"]), 4),
            "MRR":                round(avg(single_metrics["mrr"]), 4),
            f"Precision@{top_k}": round(avg(single_metrics["precision"]), 4),
        },
        "wiki_dual": {
            f"Recall@{top_k}":    round(avg(dual_metrics["recall"]), 4),
            "MRR":                round(avg(dual_metrics["mrr"]), 4),
            f"Precision@{top_k}": round(avg(dual_metrics["precision"]), 4),
        },
    }

    print()
    print("=" * 55)
    print(f"{'Metric':<20} {'wiki_single':>15} {'wiki_dual':>15}")
    print("-" * 55)
    for metric in [f"Recall@{top_k}", "MRR", f"Precision@{top_k}"]:
        sv = results["wiki_single"][metric]
        dv = results["wiki_dual"][metric]
        winner = " ◄" if dv > sv else ("" if dv == sv else "")
        print(f"{metric:<20} {sv:>15.4f} {dv:>15.4f}{winner}")
    print("=" * 55)
    print(f"Test cases: {n}")

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\nSaved: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare wiki_single vs wiki_dual: Recall@K, MRR, Precision@K",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--file", required=True, help="Path to Wikipedia JSONL file")
    p.add_argument("--max-articles", type=int, default=500,
                   help="Number of articles to use for test set generation")
    p.add_argument("--top-k", type=int, default=5, help="Retrieval depth K")
    p.add_argument("--skip", type=int, default=0, help="Skip first N articles")
    p.add_argument("--output", default=None, help="Save results to this JSON file")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(
        jsonl_path=Path(args.file),
        max_articles=args.max_articles,
        top_k=args.top_k,
        output_path=Path(args.output) if args.output else None,
        skip=args.skip,
    )
