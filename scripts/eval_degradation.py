#!/usr/bin/env python3
"""Evaluate source-article recall under query degradation.

Tests: When queries progressively deviate from original text,
which retrieval method maintains better recall?

Degradation levels:
  L0: Full first-sentence (baseline, ~easy)
  L1: First sentence with title words removed
  L2: Only keep nouns/entities from first sentence (harder)
  L3: Paraphrased concept (hardest — uses shuffled key terms)

This directly answers: "Is Dense's high Recall due to text overlap,
or does it genuinely understand semantics?"
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
# BM25
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
# Query degradation levels
# ---------------------------------------------------------------------------

STOPWORDS = set("a an the is was were are be been being have has had do does did "
                "will would shall should may might can could this that these those "
                "i me my we our you your he him his she her it its they them their "
                "what which who whom where when how why all each every both few "
                "more most other some such no nor not only own same so than too "
                "very just don doesn didn won wouldn from into with by for on at "
                "to in of and but or if then else also as".split())


def degrade_query(title: str, body: str, level: int) -> str:
    """Generate query at specified degradation level."""
    first_sent_end = body.find(". ")
    if first_sent_end < 0 or first_sent_end > 300:
        first_sent_end = min(300, len(body))
    first_sent = body[:first_sent_end]

    title_words = set(w.lower() for w in re.findall(r'\w+', title))

    if level == 0:
        # L0: Full first sentence
        return first_sent[:200]

    elif level == 1:
        # L1: First sentence with title words removed
        words = first_sent.split()
        filtered = [w for w in words
                    if w.lower().strip(".,;:!?()\"'") not in title_words]
        result = " ".join(filtered)
        return result[:200] if len(result) > 20 else ""

    elif level == 2:
        # L2: Only content words (no stopwords, no title words)
        words = re.findall(r'\b[a-zA-Z]{3,}\b', first_sent)
        content = [w for w in words
                   if w.lower() not in STOPWORDS
                   and w.lower() not in title_words]
        # Keep max 15 content words
        return " ".join(content[:15])

    elif level == 3:
        # L3: Shuffled key terms from first 2 sentences (destroys word order)
        second_end = body.find(". ", first_sent_end + 2)
        if second_end < 0:
            second_end = min(400, len(body))
        text_span = body[:second_end]
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text_span)
        content = [w for w in words
                   if w.lower() not in STOPWORDS
                   and w.lower() not in title_words]
        # Deduplicate, take top 10, shuffle by reversing
        seen = set()
        unique = []
        for w in content:
            wl = w.lower()
            if wl not in seen:
                seen.add(wl)
                unique.append(w)
        unique = unique[:10]
        unique.reverse()
        return " ".join(unique)

    return ""


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(args):
    client = QdrantClient(host="localhost", port=6333)
    collection = args.collection

    existing = {c.name for c in client.get_collections().collections}
    if collection not in existing:
        print(f"[ERROR] Collection '{collection}' not found.")
        sys.exit(1)

    info = client.get_collection(collection)
    print(f"Collection: {collection} ({info.points_count:,} points)")
    print()

    # Gather articles
    print(f"Sampling articles for degradation test (max={args.max_queries})...")
    articles = []
    offset = None
    while len(articles) < args.max_queries:
        pts, offset = client.scroll(
            collection_name=collection,
            scroll_filter=qm.Filter(must=[
                qm.FieldCondition(key="metadata.chunk_index", match=qm.MatchValue(value=0))
            ]),
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        if not pts:
            break
        for p in pts:
            meta = p.payload.get("metadata", {})
            title = meta.get("title", "")
            text = p.payload.get("text", "")
            prefix = f"# {title} "
            body = text[len(prefix):] if text.startswith(prefix) else text
            if len(body) < 80 or not title:
                continue
            articles.append({
                "title": title,
                "body": body,
                "source_id": str(p.payload.get("original_id", p.id)),
            })
            if len(articles) >= args.max_queries:
                break
        if offset is None:
            break

    print(f"  Sampled {len(articles)} articles")
    print()

    # Generate queries at each level
    levels = [0, 1, 2, 3]
    level_names = {0: "L0:full_sent", 1: "L1:no_title", 2: "L2:content_words", 3: "L3:shuffled_keys"}

    all_queries = {}  # level -> list of {"query", "title", "source_id"}
    for lvl in levels:
        queries = []
        for art in articles:
            q = degrade_query(art["title"], art["body"], lvl)
            if q and len(q) >= 10:
                queries.append({
                    "query": q,
                    "title": art["title"],
                    "source_id": art["source_id"],
                })
        all_queries[lvl] = queries
        print(f"  {level_names[lvl]}: {len(queries)} queries (sample: '{queries[0]['query'][:60]}...')")

    print()

    # Load embedding model
    print("Loading embedding model...")
    from sentence_transformers import SentenceTransformer
    emb_model = SentenceTransformer(args.embedding)
    print()

    # Build BM25 pool
    print(f"Building BM25 pool (max={args.bm25_pool})...")
    pool = []
    offset = None
    while len(pool) < args.bm25_pool:
        pts, offset = client.scroll(
            collection_name=collection, limit=200, offset=offset,
            with_payload=True, with_vectors=False,
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
    print(f"  Pool: {len(pool)} docs")
    print()

    # Evaluate each level
    top_k = args.top_k
    results = {}

    for lvl in levels:
        queries = all_queries[lvl]
        if not queries:
            continue

        # Embed
        qtexts = [q["query"] for q in queries]
        qvecs = emb_model.encode(qtexts, normalize_embeddings=True, show_progress_bar=False)

        dense_found = 0
        bm25_found = 0
        dense_rank_sum = 0
        bm25_rank_sum = 0
        n = len(queries)

        for i, q in enumerate(queries):
            source_title = q["title"]
            qvec = list(qvecs[i])

            # Dense
            resp = client.query_points(
                collection_name=collection, query=qvec, limit=top_k,
                with_payload=["metadata"],
            )
            hits = resp.points if hasattr(resp, "points") else resp
            for rank, h in enumerate(hits, 1):
                if h.payload.get("metadata", {}).get("title") == source_title:
                    dense_found += 1
                    dense_rank_sum += rank
                    break

            # BM25
            query_tokens = _tokenize(q["query"])
            if query_tokens:
                scored = _bm25_scores(query_tokens, pool)
                for rank, (_, title, _) in enumerate(scored[:top_k], 1):
                    if title == source_title:
                        bm25_found += 1
                        bm25_rank_sum += rank
                        break

        dense_recall = dense_found / n
        bm25_recall = bm25_found / n
        dense_avg_rank = dense_rank_sum / dense_found if dense_found else 0
        bm25_avg_rank = bm25_rank_sum / bm25_found if bm25_found else 0

        results[lvl] = {
            "name": level_names[lvl],
            "n_queries": n,
            "dense_recall": round(dense_recall, 4),
            "bm25_recall": round(bm25_recall, 4),
            "dense_avg_rank": round(dense_avg_rank, 1),
            "bm25_avg_rank": round(bm25_avg_rank, 1),
        }

        print(f"  {level_names[lvl]:20s}  Dense R@{top_k}={dense_recall:.3f} (avg_rank={dense_avg_rank:.1f})  |  BM25 R@{top_k}={bm25_recall:.3f} (avg_rank={bm25_avg_rank:.1f})")

    # Summary table
    print()
    print("=" * 78)
    print(f"{'Level':<22} {'Dense R@'+str(top_k):>12} {'Dense AvgRk':>12} {'BM25 R@'+str(top_k):>12} {'BM25 AvgRk':>12}")
    print("-" * 78)
    for lvl in levels:
        if lvl not in results:
            continue
        r = results[lvl]
        print(f"{r['name']:<22} {r['dense_recall']:>12.4f} {r['dense_avg_rank']:>12.1f} {r['bm25_recall']:>12.4f} {r['bm25_avg_rank']:>12.1f}")
    print("=" * 78)
    print(f"Top-K: {top_k}  |  BM25 pool: {len(pool)}  |  Collection: {collection}")
    print()
    print("INTERPRETATION:")
    print("  L0 = easy (full sentence from article)")
    print("  L1 = medium (title words removed)")
    print("  L2 = hard (only content words, no title, no stopwords)")
    print("  L3 = hardest (shuffled key terms, word order destroyed)")
    print()
    print("  If Dense degrades slower than BM25 across levels:")
    print("    → Dense has genuine semantic understanding")
    print("    → Dense + Rerank is a viable strategy")
    print()
    print("  If BM25 degrades slower:")
    print("    → Current scenario favors keyword matching")
    print("    → BM25-based pipeline is sufficient")
    print()

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({"top_k": top_k, "bm25_pool": len(pool), "levels": results}, f, indent=2)
        print(f"Saved: {output_path}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Query degradation test: Dense vs BM25 robustness",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--collection", default="documents")
    p.add_argument("--max-queries", type=int, default=150)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--bm25-pool", type=int, default=50000)
    p.add_argument("--embedding", default="BAAI/bge-small-en-v1.5")
    p.add_argument("--output", default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
