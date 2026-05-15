#!/usr/bin/env python3
"""Supplemental evaluation: cross-article semantic relevance.

This evaluates whether retrieval methods can find "semantically related but
not from the same article" content — measuring generalization ability.

Approach:
  1. Sample articles from the collection
  2. For each article, generate a paraphrased/abstracted query that does NOT
     copy the original text verbatim
  3. Retrieve Top20/Top50 from the full collection
  4. Judge relevance: same-article hits (trivial) vs cross-article hits (valuable)
  5. Compare Dense vs BM25 in discovering cross-article relevant content

Query generation strategies (no LLM needed):
  - "category_query": Use article categories + title to form a topical query
  - "paraphrase": Rearrange first-sentence words into a question form
  - "entity_query": Extract named entities and form "what is X" queries

Usage:
    python scripts/eval_cross_article.py \
      --collection documents \
      --max-queries 100 \
      --top-k 20 \
      --embedding BAAI/bge-small-en-v1.5
"""

from __future__ import annotations

import argparse
import json
import os
import re
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
# BM25 helpers (same as eval_retrieval_comparison.py)
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    return text.lower().split()


def _bm25_scores(query_tokens, corpus, k1=1.5, b=0.75):
    import math
    n_docs = len(corpus)
    if n_docs == 0:
        return []
    avg_dl = sum(d["doc_len"] for d in corpus) / n_docs
    df = {}
    for tok in set(query_tokens):
        df[tok] = sum(1 for d in corpus if tok in d["token_set"])
    scores = []
    for doc in corpus:
        score = 0.0
        dl = doc["doc_len"]
        tf_map = doc["tf"]
        for tok in query_tokens:
            if tok not in tf_map:
                continue
            tf = tf_map[tok]
            n_q = df.get(tok, 0)
            idf = math.log(1 + max(0.0, (n_docs - n_q + 0.5) / (n_q + 0.5)))
            score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avg_dl))
        scores.append((doc["id"], doc["title"], score))
    scores.sort(key=lambda x: x[2], reverse=True)
    return scores


# ---------------------------------------------------------------------------
# Query generation (no LLM, heuristic-based)
# ---------------------------------------------------------------------------

def generate_semantic_queries(
    client: QdrantClient,
    collection: str,
    max_queries: int,
) -> list[dict]:
    """Generate queries designed to test cross-article semantic retrieval.

    Strategy: For each sampled article, create a query that captures the
    *topic/concept* rather than quoting text verbatim. This tests whether
    retrieval can find related articles beyond exact text matching.
    """
    print(f"Generating semantic queries (max={max_queries})...")

    queries = []
    offset = None
    seen_titles = set()

    while len(queries) < max_queries:
        pts, offset = client.scroll(
            collection_name=collection,
            scroll_filter=qm.Filter(must=[
                qm.FieldCondition(
                    key="metadata.chunk_index",
                    match=qm.MatchValue(value=0),
                )
            ]),
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        if not pts:
            break

        for p in pts:
            if len(queries) >= max_queries:
                break
            meta = p.payload.get("metadata", {})
            title = meta.get("title", "")
            if not title or title in seen_titles:
                continue
            seen_titles.add(title)

            text = p.payload.get("text", "")
            # Strip "# Title " prefix
            prefix = f"# {title} "
            body = text[len(prefix):] if text.startswith(prefix) else text

            if len(body) < 50:
                continue

            # Generate different query types
            query_text, query_type = _make_conceptual_query(title, body)
            if not query_text or len(query_text) < 10:
                continue

            queries.append({
                "query": query_text,
                "source_title": title,
                "source_id": str(p.payload.get("original_id", p.id)),
                "query_type": query_type,
                "body_preview": body[:100],
            })

        if offset is None:
            break

    print(f"  Generated {len(queries)} queries")
    return queries


def _make_conceptual_query(title: str, body: str) -> tuple[str, str]:
    """Create a conceptual query that does NOT mention the article title.

    This tests true semantic retrieval: can the system find the article
    from a description of its content, without keyword matching on the title?
    """
    first_sent_end = body.find(". ")
    if first_sent_end < 0 or first_sent_end > 200:
        first_sent_end = min(200, len(body))
    first_sent = body[:first_sent_end]

    # Remove the title words from the query to prevent trivial keyword matching
    title_words = set(title.lower().split())
    # Keep only content words from first sentence, removing title mentions
    words = first_sent.split()
    filtered = [w for w in words if w.lower().strip(".,;:!?()\"'") not in title_words]
    content_no_title = " ".join(filtered)

    if len(content_no_title) < 20:
        # fallback: use second sentence if available
        second_start = first_sent_end + 2
        second_end = body.find(". ", second_start)
        if second_end > second_start:
            content_no_title = body[second_start:second_end]
        else:
            return "", "skip"

    # Determine query type
    patterns = [
        (r"\bis a\b|\bwas a\b", "definition"),
        (r"\blocated in\b|\bin the\b.*\bregion\b|\bcity\b|\bcountry\b", "location"),
        (r"\bborn\b|\bdied\b|\bcareer\b", "biography"),
        (r"\breleased\b|\bpublished\b|\balbum\b|\bfilm\b|\bsong\b", "media"),
    ]

    query_type = "concept"
    for pattern, qtype in patterns:
        if re.search(pattern, first_sent, re.IGNORECASE):
            query_type = qtype
            break

    # Truncate to ~60 words max to keep it query-like
    query_words = content_no_title.split()[:60]
    query = " ".join(query_words)

    return query, query_type


# ---------------------------------------------------------------------------
# Retrieval functions
# ---------------------------------------------------------------------------

def search_dense_with_details(client, collection, query_vec, top_k):
    """Dense search returning (id, title, score) tuples."""
    response = client.query_points(
        collection_name=collection,
        query=query_vec,
        limit=top_k,
        with_payload=["original_id", "metadata", "text"],
    )
    hits = response.points if hasattr(response, "points") else response
    results = []
    for h in hits:
        meta = h.payload.get("metadata", {})
        results.append({
            "id": str(h.payload.get("original_id", h.id)),
            "title": meta.get("title", ""),
            "score": h.score,
            "text_preview": str(h.payload.get("text", ""))[:80],
        })
    return results


def search_bm25_with_details(query_text, bm25_pool, top_k):
    """BM25 search returning (id, title, score) tuples."""
    query_tokens = _tokenize(query_text)
    if not query_tokens:
        return []
    scored = _bm25_scores(query_tokens, bm25_pool)
    results = []
    for doc_id, title, score in scored[:top_k]:
        results.append({
            "id": doc_id,
            "title": title,
            "score": score,
        })
    return results


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(args: argparse.Namespace) -> None:
    client = QdrantClient(host="localhost", port=6333)
    collection = args.collection

    existing = {c.name for c in client.get_collections().collections}
    if collection not in existing:
        print(f"[ERROR] Collection '{collection}' not found.")
        sys.exit(1)

    info = client.get_collection(collection)
    print(f"Collection: {collection} ({info.points_count:,} points)")
    print()

    # Generate queries
    queries = generate_semantic_queries(client, collection, args.max_queries)
    if not queries:
        print("[ERROR] No queries generated.")
        sys.exit(1)

    # Embed queries for dense search
    print("Loading embedding model...")
    from sentence_transformers import SentenceTransformer
    model_name = args.embedding
    emb_model = SentenceTransformer(model_name)
    query_texts = [q["query"] for q in queries]
    print(f"  Embedding {len(query_texts)} queries...")
    query_vecs = emb_model.encode(query_texts, normalize_embeddings=True, show_progress_bar=True)
    print()

    # Build BM25 pool
    print(f"Building BM25 pool (max={args.bm25_pool})...")
    pool = []
    offset = None
    while len(pool) < args.bm25_pool:
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
            text = p.payload.get("text", "")
            tokens = _tokenize(text)
            pool.append({
                "id": str(p.payload.get("original_id", p.id)),
                "title": meta.get("title", ""),
                "tokens": tokens,
                "token_set": set(tokens),
                "tf": Counter(tokens),
                "doc_len": len(tokens),
            })
        if offset is None:
            break
    print(f"  Pool size: {len(pool)}")
    print()

    # Evaluate
    top_k = args.top_k
    dense_stats = {"same_article": 0, "cross_article": 0, "total_hits": 0}
    bm25_stats = {"same_article": 0, "cross_article": 0, "total_hits": 0}
    dense_unique_titles = []
    bm25_unique_titles = []
    dense_has_relevant = 0
    bm25_has_relevant = 0

    print(f"Evaluating {len(queries)} queries at Top-{top_k}...")
    print()

    for i, q in enumerate(queries):
        source_title = q["source_title"]
        qvec = list(query_vecs[i])

        # Dense search
        dense_hits = search_dense_with_details(client, collection, qvec, top_k)
        dense_titles_in_result = set()
        dense_has_same = False
        for h in dense_hits:
            if h["title"] == source_title:
                dense_stats["same_article"] += 1
                dense_has_same = True
            else:
                dense_stats["cross_article"] += 1
                dense_titles_in_result.add(h["title"])
            dense_stats["total_hits"] += 1
        dense_unique_titles.append(len(dense_titles_in_result))
        if dense_has_same or dense_titles_in_result:
            dense_has_relevant += 1

        # BM25 search
        bm25_hits = search_bm25_with_details(q["query"], pool, top_k)
        bm25_titles_in_result = set()
        bm25_has_same = False
        for h in bm25_hits:
            if h["title"] == source_title:
                bm25_stats["same_article"] += 1
                bm25_has_same = True
            else:
                bm25_stats["cross_article"] += 1
                bm25_titles_in_result.add(h["title"])
            bm25_stats["total_hits"] += 1
        bm25_unique_titles.append(len(bm25_titles_in_result))
        if bm25_has_same or bm25_titles_in_result:
            bm25_has_relevant += 1

        if (i + 1) % 25 == 0:
            print(f"  [{i+1}/{len(queries)}] dense cross-article={dense_stats['cross_article']} | bm25 cross-article={bm25_stats['cross_article']}")

    # Results
    n = len(queries)
    print()
    print("=" * 70)
    print("CROSS-ARTICLE RETRIEVAL ANALYSIS")
    print("=" * 70)
    print()
    print(f"Queries: {n}  |  Top-K: {top_k}  |  BM25 pool: {len(pool)}")
    print()

    print(f"{'Metric':<40} {'Dense':>10} {'BM25':>10}")
    print("-" * 62)

    # Same-article hit rate (finding source article)
    dense_same_rate = dense_stats["same_article"] / (n * top_k)
    bm25_same_rate = bm25_stats["same_article"] / (n * top_k)
    print(f"{'Same-article hits (% of all slots)':<40} {dense_same_rate:>9.1%} {bm25_same_rate:>9.1%}")

    # Cross-article hit rate
    dense_cross_rate = dense_stats["cross_article"] / (n * top_k)
    bm25_cross_rate = bm25_stats["cross_article"] / (n * top_k)
    print(f"{'Cross-article hits (% of all slots)':<40} {dense_cross_rate:>9.1%} {bm25_cross_rate:>9.1%}")

    # Diversity: avg unique titles per query
    dense_diversity = sum(dense_unique_titles) / n
    bm25_diversity = sum(bm25_unique_titles) / n
    print(f"{'Avg unique cross-article titles per query':<40} {dense_diversity:>10.1f} {bm25_diversity:>10.1f}")

    # At least 1 relevant hit rate
    print(f"{'Queries with source article in Top-K':<40} {dense_has_relevant/n:>9.1%} {bm25_has_relevant/n:>9.1%}")

    print()
    print("-" * 62)
    print("INTERPRETATION:")
    print()
    print("  - 'Same-article hits': Finding chunks from the query's source article")
    print("    (expected to be high — this is the trivial retrieval task)")
    print()
    print("  - 'Cross-article hits': Finding chunks from OTHER articles")
    print("    (this measures semantic generalization — can the method discover")
    print("    related content beyond exact text matching?)")
    print()
    print("  - 'Avg unique titles': Diversity of cross-article results")
    print("    (higher = method explores more diverse related content)")
    print()

    if dense_diversity > bm25_diversity * 1.2:
        print("  >>> Dense retrieves MORE diverse cross-article content than BM25.")
        print("      This suggests Dense has stronger semantic generalization.")
    elif bm25_diversity > dense_diversity * 1.2:
        print("  >>> BM25 retrieves MORE diverse cross-article content than Dense.")
        print("      This may be due to keyword overlap across articles.")
    else:
        print("  >>> Dense and BM25 have similar cross-article diversity.")

    print()
    print("=" * 70)

    # Save detailed results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_data = {
            "n_queries": n,
            "top_k": top_k,
            "bm25_pool": len(pool),
            "dense": {
                "same_article_hits": dense_stats["same_article"],
                "cross_article_hits": dense_stats["cross_article"],
                "avg_unique_cross_titles": round(dense_diversity, 2),
                "source_found_rate": round(dense_has_relevant / n, 4),
            },
            "bm25": {
                "same_article_hits": bm25_stats["same_article"],
                "cross_article_hits": bm25_stats["cross_article"],
                "avg_unique_cross_titles": round(bm25_diversity, 2),
                "source_found_rate": round(bm25_has_relevant / n, 4),
            },
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"Saved: {output_path}")

    if args.save_queries:
        queries_path = Path(args.save_queries)
        queries_path.parent.mkdir(parents=True, exist_ok=True)
        save_data = [
            {
                "query": q["query"],
                "source_title": q["source_title"],
                "source_id": q["source_id"],
                "query_type": q["query_type"],
                "body_preview": q["body_preview"],
            }
            for q in queries
        ]
        with open(queries_path, "w", encoding="utf-8") as f:
            json.dump({"n_queries": len(save_data), "queries": save_data}, f, ensure_ascii=False, indent=2)
        print(f"Saved queries: {queries_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Cross-article semantic relevance evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--collection", default="documents")
    p.add_argument("--max-queries", type=int, default=100)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--bm25-pool", type=int, default=50000)
    p.add_argument("--embedding", default="BAAI/bge-small-en-v1.5")
    p.add_argument("--output", default=None)
    p.add_argument("--save-queries", default=None,
                   help="Save generated query set to JSON file")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
