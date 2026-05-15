#!/usr/bin/env python3
"""Generate test set from Qdrant collection for retrieval/rerank evaluation.

This mirrors the test set generation approach used in eval_retrieval_comparison.py:
- Scroll through collection to find articles
- Generate queries based on title/abstract/full mode
- Output JSONL format for reuse across experiments

Usage:
    python scripts/generate_test_set.py \
      --collection documents \
      --max-articles 100 \
      --query-mode title \
      --output test_set.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv
load_dotenv(_ROOT / ".env")

from qdrant_client import QdrantClient


def build_test_set_from_collection(
    client: QdrantClient,
    collection: str,
    max_articles: int,
    query_mode: str = "title",
) -> list[dict]:
    """Build test set by sampling articles from the collection."""
    print(f"Scanning collection '{collection}' for test articles...")
    
    articles: dict[str, list[dict]] = {}  # title -> list of {id, chunk_index, text}
    offset = None
    batch_size = 200
    processed = 0
    
    while len(articles) < max_articles:
        pts, offset = client.scroll(
            collection_name=collection,
            scroll_filter=None,
            limit=batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        
        if not pts:
            break
        
        for p in pts:
            processed += 1
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
        
        if processed % 1000 == 0:
            print(f"  Processed {processed} points, found {len(articles)} unique articles...")
        
        if offset is None:
            break
    
    print(f"Found {len(articles)} unique articles from {processed} points")
    
    test_cases = []
    titles = list(articles.keys())[:max_articles]
    
    for i, title in enumerate(titles, 1):
        chunks = articles[title]
        query_chunks = [c for c in chunks if c["chunk_index"] == 0]
        if not query_chunks:
            continue
        
        first_text = query_chunks[0]["text"]
        body = first_text
        
        if body.startswith("# "):
            prefix = "# " + title + " "
            if body.startswith(prefix):
                body = body[len(prefix):]
            else:
                body = body[2 + len(title):].lstrip()
        
        if query_mode == "title":
            query_text = title
        elif query_mode == "abstract":
            period_idx = body.find(". ")
            if period_idx > 0:
                query_text = body[:period_idx + 1]
            else:
                query_text = body[:200]
        else:
            query_text = body[:300]
        
        if len(query_text) < 3:
            continue
        
        reference_chunk = chunks[0]["text"] if chunks else ""
        test_cases.append({
            "question": query_text,
            "reference": reference_chunk,
            "title": title,
            "relevant_ids": list({c["id"] for c in chunks}),
            "n_chunks": len(chunks),
        })
    
    return test_cases


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate test set from Qdrant collection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--collection", default="documents", help="Qdrant collection name")
    parser.add_argument("--max-articles", type=int, default=100, help="Number of articles to sample")
    parser.add_argument("--query-mode", default="title", choices=["title", "abstract", "full"],
                        help="Query generation mode")
    parser.add_argument("--output", required=True, help="Output JSONL file path")
    args = parser.parse_args()
    
    client = QdrantClient(host="localhost", port=6333)
    
    existing = {c.name for c in client.get_collections().collections}
    if args.collection not in existing:
        print(f"[ERROR] Collection '{args.collection}' not found. Available: {existing}")
        sys.exit(1)
    
    info = client.get_collection(args.collection)
    print(f"Collection: {args.collection}")
    print(f"  Points: {info.points_count:,}")
    print()
    
    test_cases = build_test_set_from_collection(
        client, args.collection, args.max_articles, args.query_mode
    )
    
    print(f"\nGenerated {len(test_cases)} test cases")
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        for case in test_cases:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")
    
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
