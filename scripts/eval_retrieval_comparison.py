#!/usr/bin/env python3
"""Compare retrieval methods on an existing Qdrant collection.

Methods compared:
  - dense:      HNSW approximate nearest neighbor (Qdrant native)
  - bm25:       Client-side BM25 keyword matching
  - hybrid_rrf: Dense + BM25 fused via Reciprocal Rank Fusion (k=20)

Metrics: Recall@K, Precision@K, MRR, Latency (ms/query)

Usage:
    python scripts/eval_retrieval_comparison.py \
      --file C:/learning/enwiki_namespace_0/enwiki_namespace_0_0.jsonl \
      --collection documents \
      --max-articles 200 \
      --top-k 5
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

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm


# ---------------------------------------------------------------------------
# BM25 helpers
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    """Simple whitespace + lowercase tokenizer."""
    return text.lower().split()


def _bm25_scores(
    query_tokens: list[str],
    corpus: list[dict],
    k1: float = 1.5,
    b: float = 0.75,
) -> list[tuple[str, float]]:
    """Compute BM25 scores for query against corpus.

    corpus: list of {"id": str, "text": str, "tokens": list[str], "doc_len": int}
    Returns sorted (id, score) pairs descending.
    """
    n_docs = len(corpus)
    if n_docs == 0:
        return []

    avg_dl = sum(d["doc_len"] for d in corpus) / n_docs

    # document frequency for query terms
    df: dict[str, int] = {}
    for tok in set(query_tokens):
        df[tok] = sum(1 for d in corpus if tok in d["token_set"])

    scores: list[tuple[str, float]] = []
    for doc in corpus:
        score = 0.0
        dl = doc["doc_len"]
        tf_map = doc["tf"]
        for tok in query_tokens:
            if tok not in tf_map:
                continue
            tf = tf_map[tok]
            n_q = df.get(tok, 0)
            idf = max(0.0, ((n_docs - n_q + 0.5) / (n_q + 0.5)))
            import math
            idf = math.log(1 + idf)
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * dl / avg_dl)
            score += idf * numerator / denominator
        scores.append((doc["id"], score))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


# ---------------------------------------------------------------------------
# Retrieval functions
# ---------------------------------------------------------------------------

def search_dense(
    client: QdrantClient,
    collection: str,
    query_vec: list[float],
    top_k: int,
) -> list[str]:
    """Dense HNSW search via Qdrant."""
    response = client.query_points(
        collection_name=collection,
        query=query_vec,
        limit=top_k,
        with_payload=["original_id"],
    )
    hits = response.points if hasattr(response, "points") else response
    return [str(h.payload.get("original_id", h.id)) for h in hits]


def search_bm25(
    query_text: str,
    bm25_pool: list[dict],
    top_k: int,
) -> list[str]:
    """Client-side BM25 on pre-built pool."""
    query_tokens = _tokenize(query_text)
    if not query_tokens:
        return []
    scored = _bm25_scores(query_tokens, bm25_pool)
    return [doc_id for doc_id, _ in scored[:top_k]]


def search_hybrid_rrf(
    client: QdrantClient,
    collection: str,
    query_vec: list[float],
    query_text: str,
    bm25_pool: list[dict],
    top_k: int,
    k_rrf: float = 20.0,
) -> list[str]:
    """Hybrid: Dense + BM25 fused with Reciprocal Rank Fusion."""
    # Dense leg: get 2*top_k candidates
    dense_ids = search_dense(client, collection, query_vec, top_k * 2)

    # BM25 leg: get 2*top_k candidates
    bm25_ids = search_bm25(query_text, bm25_pool, top_k * 2)

    # RRF fusion
    scores: dict[str, float] = {}
    for rank, doc_id in enumerate(dense_ids, start=1):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k_rrf + rank)
    for rank, doc_id in enumerate(bm25_ids, start=1):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k_rrf + rank)

    ranked = sorted(scores, key=lambda x: scores[x], reverse=True)
    return ranked[:top_k]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def recall_at_k(retrieved: list[str], relevant: set[str]) -> float:
    if not relevant:
        return 0.0
    return sum(1 for r in retrieved if r in relevant) / len(relevant)


def precision_at_k(retrieved: list[str], relevant: set[str]) -> float:
    if not retrieved:
        return 0.0
    return sum(1 for r in retrieved if r in relevant) / len(retrieved)


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    for rank, r in enumerate(retrieved, start=1):
        if r in relevant:
            return 1.0 / rank
    return 0.0


# ---------------------------------------------------------------------------
# Test-set generation from Qdrant collection
# ---------------------------------------------------------------------------

def build_test_set_from_collection(
    client: QdrantClient,
    collection: str,
    source_file: str | None,
    max_articles: int,
    skip: int = 0,
    query_mode: str = "title",
) -> list[dict]:
    """Build test set by sampling articles from the collection.

    query_mode:
      - "title": use article title as query (simulates real user questions)
      - "abstract": use first sentence of chunk_index=0 as query
      - "full": use full chunk_index=0 text as query (easy mode, for debugging)
    """
    print(f"Scanning collection for test articles (max={max_articles}, skip={skip}, query_mode={query_mode})...")

    # Build filter for specific source file if provided
    scroll_filter = None
    if source_file:
        file_name = Path(source_file).name
        scroll_filter = qm.Filter(must=[
            qm.FieldCondition(
                key="metadata.file_name",
                match=qm.MatchValue(value=file_name),
            )
        ])

    # Scroll through collection to find articles
    articles: dict[str, list[dict]] = {}  # title -> list of {id, chunk_index, text}
    offset = None
    batch_size = 200

    while len(articles) < max_articles + skip:
        pts, offset = client.scroll(
            collection_name=collection,
            scroll_filter=scroll_filter,
            limit=batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        if not pts:
            break

        for p in pts:
            meta = p.payload.get("metadata", {})
            title = meta.get("title", "")
            if not title:
                continue
            chunk_index = meta.get("chunk_index", 0)
            oid = str(p.payload.get("original_id", p.id))
            text = p.payload.get("text", "")

            if title not in articles:
                articles[title] = []
            articles[title].append({
                "id": oid,
                "chunk_index": chunk_index,
                "text": text,
            })

        if offset is None:
            break

    # Build test cases
    test_cases = []
    article_titles = list(articles.keys())

    for title in article_titles[skip:skip + max_articles]:
        chunks = articles[title]
        query_chunks = [c for c in chunks if c["chunk_index"] == 0]
        if not query_chunks:
            continue

        first_text = query_chunks[0]["text"]
        # Strip "# Title " prefix (format: "# Title Content..." with space, not newline)
        body = first_text
        if body.startswith("# "):
            # Remove "# Title " by finding where title ends
            prefix = "# " + title + " "
            if body.startswith(prefix):
                body = body[len(prefix):]
            else:
                # fallback: skip first "# " and the title word
                body = body[2 + len(title):].lstrip()

        if query_mode == "title":
            query_text = title
        elif query_mode == "abstract":
            # Use first sentence (up to first period)
            period_idx = body.find(". ")
            if period_idx > 0:
                query_text = body[:period_idx + 1]
            else:
                query_text = body[:200]
        else:  # "full"
            query_text = body[:300]

        if len(query_text) < 3:
            continue

        relevant_ids = {c["id"] for c in chunks}
        test_cases.append({
            "query": query_text,
            "title": title,
            "relevant_ids": relevant_ids,
            "n_chunks": len(chunks),
        })

    return test_cases


# ---------------------------------------------------------------------------
# BM25 pool builder
# ---------------------------------------------------------------------------

def build_bm25_pool(
    client: QdrantClient,
    collection: str,
    titles: set[str],
    max_pool: int = 10000,
) -> list[dict]:
    """Build BM25 candidate pool from collection, filtered to relevant articles."""
    print(f"Building BM25 pool (target titles: {len(titles)}, max_pool: {max_pool})...")

    pool = []
    offset = None
    seen_titles = set()

    while len(pool) < max_pool:
        pts, offset = client.scroll(
            collection_name=collection,
            limit=200,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        if not pts:
            break

        for p in pts:
            meta = p.payload.get("metadata", {})
            title = meta.get("title", "")
            # Include docs from test articles + some extra for BM25 noise
            text = p.payload.get("text", "")
            oid = str(p.payload.get("original_id", p.id))
            tokens = _tokenize(text)
            tf = Counter(tokens)
            pool.append({
                "id": oid,
                "text": text,
                "tokens": tokens,
                "token_set": set(tokens),
                "tf": tf,
                "doc_len": len(tokens),
                "title": title,
            })
            if title in titles:
                seen_titles.add(title)

        if offset is None:
            break

        if len(pool) % 2000 == 0:
            print(f"  pool size: {len(pool)}, covered {len(seen_titles)}/{len(titles)} test titles")

    print(f"  Final pool: {len(pool)} docs, covered {len(seen_titles)}/{len(titles)} test titles")
    return pool


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def evaluate(args: argparse.Namespace) -> None:
    client = QdrantClient(host="localhost", port=6333)
    collection = args.collection

    # Verify collection exists
    existing = {c.name for c in client.get_collections().collections}
    if collection not in existing:
        print(f"[ERROR] Collection '{collection}' not found. Available: {existing}")
        sys.exit(1)

    info = client.get_collection(collection)
    print(f"Collection: {collection}")
    print(f"  Points: {info.points_count:,}")
    print(f"  Vectors: {info.config.params.vectors}")
    print()

    # Methods to evaluate
    methods = [m.strip() for m in args.methods.split(",")]
    valid = {"dense", "bm25", "hybrid_rrf"}
    invalid = set(methods) - valid
    if invalid:
        print(f"[ERROR] Unknown methods: {invalid}. Valid: {valid}")
        sys.exit(1)

    # Build test set
    t0 = time.time()
    test_cases = build_test_set_from_collection(
        client, collection,
        source_file=args.file,
        max_articles=args.max_articles,
        skip=args.skip,
        query_mode=args.query_mode,
    )
    print(f"Test cases: {len(test_cases)} ({time.time() - t0:.1f}s)")
    print()

    if not test_cases:
        print("[ERROR] No test cases generated. Check --file or --max-articles.")
        sys.exit(1)

    # Embed all queries
    need_dense = "dense" in methods or "hybrid_rrf" in methods
    query_vecs = None
    if need_dense:
        print("Loading embedding model...")
        from sentence_transformers import SentenceTransformer
        # Determine model from vector dimension
        vec_dim = info.config.params.vectors.size
        if vec_dim == 384:
            model_name = args.embedding or "sentence-transformers/all-MiniLM-L6-v2"
        elif vec_dim == 1024:
            model_name = args.embedding or "BAAI/bge-m3"
        else:
            model_name = args.embedding or "sentence-transformers/all-MiniLM-L6-v2"
        print(f"  Model: {model_name} (dim={vec_dim})")

        emb_model = SentenceTransformer(model_name)
        queries = [tc["query"] for tc in test_cases]
        print(f"  Embedding {len(queries)} queries...")
        t1 = time.time()
        query_vecs = emb_model.encode(queries, normalize_embeddings=True, show_progress_bar=True)
        print(f"  Done in {time.time() - t1:.1f}s")
        print()

    # Build BM25 pool if needed
    bm25_pool = None
    need_bm25 = "bm25" in methods or "hybrid_rrf" in methods
    if need_bm25:
        test_titles = {tc["title"] for tc in test_cases}
        bm25_pool = build_bm25_pool(
            client, collection, test_titles, max_pool=args.bm25_pool
        )
        print()

    # Parse K values for Recall@K curve
    k_values = sorted(set([args.top_k] + [int(x) for x in args.also_k.split(",") if x.strip()]))
    max_k = max(k_values)

    # Run evaluation: retrieve at max_k, compute metrics at each K by slicing
    metrics: dict[str, dict[str, list[float]]] = {}
    for m in methods:
        metrics[m] = {"latency_ms": []}
        for k in k_values:
            metrics[m][f"recall@{k}"] = []
            metrics[m][f"precision@{k}"] = []
        metrics[m]["mrr"] = []

    n = len(test_cases)

    for i, tc in enumerate(test_cases):
        relevant = tc["relevant_ids"]
        query_text = tc["query"]
        qvec = list(query_vecs[i]) if query_vecs is not None else None

        for method in methods:
            t_start = time.perf_counter()

            if method == "dense":
                retrieved = search_dense(client, collection, qvec, max_k)
            elif method == "bm25":
                retrieved = search_bm25(query_text, bm25_pool, max_k)
            elif method == "hybrid_rrf":
                retrieved = search_hybrid_rrf(
                    client, collection, qvec, query_text, bm25_pool, max_k
                )
            else:
                retrieved = []

            elapsed_ms = (time.perf_counter() - t_start) * 1000
            metrics[method]["latency_ms"].append(elapsed_ms)
            metrics[method]["mrr"].append(reciprocal_rank(retrieved, relevant))

            for k in k_values:
                sliced = retrieved[:k]
                metrics[method][f"recall@{k}"].append(recall_at_k(sliced, relevant))
                metrics[method][f"precision@{k}"].append(precision_at_k(sliced, relevant))

        if (i + 1) % 50 == 0 or i == n - 1:
            parts = []
            for m in methods:
                r = sum(metrics[m][f"recall@{args.top_k}"]) / (i + 1)
                parts.append(f"{m} R@{args.top_k}={r:.3f}")
            print(f"  [{i+1}/{n}] {' | '.join(parts)}")

    # Print results table
    def avg(lst):
        return sum(lst) / len(lst) if lst else 0.0

    print()
    # Header
    recall_headers = [f"R@{k}" for k in k_values]
    header = f"{'Method':<14} " + " ".join(f"{h:>7}" for h in recall_headers) + f" {'MRR':>7} {'Latency':>9}"
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    results = {}
    for m in methods:
        recalls = [avg(metrics[m][f"recall@{k}"]) for k in k_values]
        mrr = avg(metrics[m]["mrr"])
        lat = avg(metrics[m]["latency_ms"])
        row = f"{m:<14} " + " ".join(f"{r:>7.4f}" for r in recalls) + f" {mrr:>7.4f} {lat:>7.1f}ms"
        print(row)
        results[m] = {}
        for k, r in zip(k_values, recalls):
            results[m][f"Recall@{k}"] = round(r, 4)
        results[m]["MRR"] = round(mrr, 4)
        results[m]["Latency_ms"] = round(lat, 2)

    print("=" * len(header))
    print(f"Test cases: {n}  K values: {k_values}  collection: {collection}")
    if need_bm25:
        print(f"Note: BM25 uses client-side scoring (pool={len(bm25_pool)} docs)")
    print()

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_data = {
            "n_test_cases": n,
            "k_values": k_values,
            "collection": collection,
            "query_mode": args.query_mode,
            "bm25_pool_size": len(bm25_pool) if bm25_pool else 0,
            "results": results,
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"Saved: {output_path}")

    if args.save_queries:
        queries_path = Path(args.save_queries)
        queries_path.parent.mkdir(parents=True, exist_ok=True)
        queries_data = [
            {
                "query": tc["query"],
                "title": tc["title"],
                "n_chunks": tc["n_chunks"],
                "relevant_ids": list(tc["relevant_ids"]),
            }
            for tc in test_cases
        ]
        with open(queries_path, "w", encoding="utf-8") as f:
            json.dump({"query_mode": args.query_mode, "queries": queries_data}, f, ensure_ascii=False, indent=2)
        print(f"Saved queries: {queries_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare retrieval methods: Dense (HNSW) vs BM25 vs Hybrid-RRF",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--file", default=None,
                   help="Filter test set to articles from this JSONL file")
    p.add_argument("--collection", default="documents",
                   help="Qdrant collection name")
    p.add_argument("--max-articles", type=int, default=200,
                   help="Number of articles for test set")
    p.add_argument("--top-k", type=int, default=5,
                   help="Primary retrieval depth K")
    p.add_argument("--also-k", default="10,20,50",
                   help="Additional K values for Recall@K curve (comma-separated)")
    p.add_argument("--skip", type=int, default=0,
                   help="Skip first N articles in scroll")
    p.add_argument("--methods", default="dense,bm25,hybrid_rrf",
                   help="Comma-separated methods to compare")
    p.add_argument("--bm25-pool", type=int, default=5000,
                   help="Max docs in BM25 candidate pool")
    p.add_argument("--query-mode", default="title",
                   choices=["title", "abstract", "full"],
                   help="How to generate queries: title (realistic), abstract (first sentence), full (chunk text)")
    p.add_argument("--embedding", default=None,
                   help="Embedding model (auto-detected from vector dim if omitted)")
    p.add_argument("--output", default=None,
                   help="Save results to JSON file")
    p.add_argument("--save-queries", default=None,
                   help="Save generated query set to JSON file")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
