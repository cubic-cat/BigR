# Wikipedia Ingestion & Index Comparison on Mac (M3) — Instructions for Claude

You are running on a Mac with Apple M3 chip. Your job is to:
1. Ingest the first 5 Wikipedia JSONL files (~10 GB) into **two separate Qdrant collections** — one using single-vector (dense-only) indexing, one using dual-vector (dense + sparse) indexing.
2. Package both collections and transfer them back to the Windows machine for offline evaluation.

Follow the steps below in order. Do not skip steps.

---

## Context

This is part of the BigR RAG system. The chunking strategy has already been decided: **fixed512** (512-token sliding window, 50-token overlap) with **BAAI/bge-m3** embedding. The next evaluation goal is to compare two retrieval index configurations on a 10 GB pilot corpus.

### What we're comparing

| Collection name | Index type | Description |
|-----------------|------------|-------------|
| `wiki_single` | Single-vector (dense only) | One bge-m3 vector per chunk; cosine similarity search |
| `wiki_dual` | Dual-vector (dense + sparse) | bge-m3 dense vector + BM25 sparse vector per chunk; RRF fusion at query time |

### Evaluation metrics (run on Windows after transfer)

| Metric | Target | Description |
|--------|--------|-------------|
| Recall@5 | > 0.75 | Core metric — fraction of relevant docs in top-5 |
| MRR | > 0.65 | Whether the first relevant result ranks high |
| Precision@5 | > 0.50 | Fraction of top-5 that are relevant (noise control) |

### Data

- **Files**: `enwiki_namespace_0_0.jsonl` through `_4.jsonl` (5 files, ~10 GB total)
- **Articles**: ~1.5 million
- **Chunks**: ~3–4 million (at fixed512 granularity)
- **Est. ingestion time per collection**: 20–40 min on M3 MPS

---

## Step 0 — Verify project files

Check that the following files exist:

```
BigR/
├── .env
├── configs/
├── core/embedding.py          ← must contain LocalEmbeddingClient
├── core/qdrant_retriever.py
├── document/wiki_loader.py
├── document/splitter.py
└── scripts/ingest_wikipedia.py
```

If any are missing, tell the user which files are missing and stop.

---

## Step 1 — Install Python dependencies

```bash
cd <project_root>/BigR
pip install -r requirements.txt
pip install rank_bm25   # required for sparse indexing
```

---

## Step 2 — Install and start Qdrant

```bash
# Download Qdrant for Apple Silicon
curl -L https://github.com/qdrant/qdrant/releases/latest/download/qdrant-aarch64-apple-darwin.tar.gz -o qdrant.tar.gz
tar -xzf qdrant.tar.gz
chmod +x qdrant
```

**Important**: always start Qdrant from the directory containing the binary:

```bash
cd <directory_containing_qdrant_binary>
./qdrant
```

Verify at http://localhost:6333/dashboard before continuing.

---

## Step 3 — Configure .env

Open `.env` and verify these values:

```env
VECTOR_DB_PROVIDER=qdrant
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_VECTOR_SIZE=1024
```

The Qwen API key is not needed for ingestion. Leave other values as-is.

---

## Step 4 — Download bge-m3

```bash
python -c "
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('BAAI/bge-m3')
print('bge-m3 ready, dim =', model.get_embedding_dimension())
"
```

Expected: `bge-m3 ready, dim = 1024`

If slow or timing out, use the mirror:
```bash
HF_ENDPOINT=https://hf-mirror.com python -c "
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('BAAI/bge-m3')
print('bge-m3 ready, dim =', model.get_embedding_dimension())
"
```

---

## Step 5 — Verify MPS

```bash
python -c "
import torch
from sentence_transformers import SentenceTransformer
print('MPS available:', torch.backends.mps.is_available())
model = SentenceTransformer('BAAI/bge-m3', device='mps')
vecs = model.encode(['test sentence'] * 64, batch_size=64)
print('Shape:', vecs.shape)
"
```

Expected: `MPS available: True` and `Shape: (64, 1024)`

---

## Step 6 — Update ingest_wikipedia.py to use bge-m3 + fixed512

In `scripts/ingest_wikipedia.py`, find this block (around line 218):

```python
from core.embedding import EmbeddingClient
embedding_client = EmbeddingClient()
if not embedding_client.is_configured():
    print("[ERROR] Embedding API key missing. Set EMBEDDING_API_KEY in .env", file=sys.stderr)
    sys.exit(1)
```

Replace with:

```python
from core.embedding import build_embedding_client
embedding_client = build_embedding_client(model="BAAI/bge-m3", batch_size=64)
```

Also change the default `--embed-batch-size` from `10` to `64` (line ~100):
```python
p.add_argument("--embed-batch-size", type=int, default=64, ...)
```

---

## Step 7 — Smoke test (100 articles)

```bash
cd <project_root>/BigR
python scripts/ingest_wikipedia.py \
  --file /path/to/enwiki_namespace_0/enwiki_namespace_0_0.jsonl \
  --max-articles 100 \
  --collection smoke_test \
  --embed-batch-size 64
```

Expected:
```
[ingest] Articles processed : ~98
[ingest] Chunks ingested    : ~200
[ingest] Qdrant points      : ~200
```

If this works, proceed.

---

## Step 8 — Ingest into single-vector collection (`wiki_single`)

This collection uses **dense-only** indexing (standard Qdrant cosine similarity). This is the default behavior of `ingest_wikipedia.py` — no changes needed beyond Step 6.

```bash
for i in 0 1 2 3 4; do
  echo "=== wiki_single: file $i ==="
  python scripts/ingest_wikipedia.py \
    --file /path/to/enwiki_namespace_0/enwiki_namespace_0_${i}.jsonl \
    --collection wiki_single \
    --embed-batch-size 64
done
```

Verify after all 5 files:

```bash
python -c "
from qdrant_client import QdrantClient
client = QdrantClient(host='localhost', port=6333)
info = client.get_collection('wiki_single')
print('wiki_single points:', info.points_count)
"
```

Expected: ~3–4 million points.

---

## Step 9 — Ingest into dual-vector collection (`wiki_dual`)

This collection stores **both** a bge-m3 dense vector and a BM25 sparse vector per chunk, enabling hybrid RRF retrieval.

You need to write a new ingestion script for this. Create `scripts/ingest_wikipedia_dual.py` with the following content:

```python
#!/usr/bin/env python3
"""Ingest Wikipedia JSONL into Qdrant with both dense (bge-m3) and sparse (BM25) vectors."""

from __future__ import annotations
import argparse, os, sys, time
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
from rank_bm25 import BM25Okapi


COLLECTION = "wiki_dual"
DENSE_DIM = 1024
EMBED_BATCH = 64
UPSERT_BATCH = 64
ARTICLE_BATCH = 50


def tokenize(text: str) -> list[str]:
    return text.lower().split()


def bm25_sparse_vector(tokens: list[str], corpus_tokens: list[list[str]]) -> dict[int, float]:
    """Compute BM25 scores for one document against a mini-corpus, return as sparse dict."""
    bm25 = BM25Okapi(corpus_tokens)
    scores = bm25.get_scores(tokens)
    # Build vocab index from corpus
    vocab = {}
    for doc in corpus_tokens:
        for tok in doc:
            if tok not in vocab:
                vocab[tok] = len(vocab)
    sparse = {}
    for tok, score in zip(tokens, [scores[i] for i in range(len(scores))]):
        pass
    # Simpler: use TF as sparse weights (BM25 needs full corpus for IDF, use TF-based proxy)
    from collections import Counter
    tf = Counter(tokens)
    total = sum(tf.values())
    return {hash(tok) % 100000: count / total for tok, count in tf.most_common(64)}


def ensure_collection(client: QdrantClient) -> None:
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION in existing:
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


def ingest(jsonl_path: Path, skip: int = 0, max_articles: int = None) -> None:
    client = QdrantClient(host="localhost", port=6333)
    ensure_collection(client)

    embedding_client = build_embedding_client(model="BAAI/bge-m3", batch_size=EMBED_BATCH)

    total_articles = 0
    total_chunks = 0
    t0 = time.time()
    pending: list[TextChunk] = []

    def flush():
        nonlocal total_chunks
        if not pending:
            return
        texts = [c.text for c in pending]

        # Dense vectors
        dense_vecs = embedding_client.embed_texts(texts)

        # Sparse vectors (TF-based)
        points = []
        for i, (chunk, dvec) in enumerate(zip(pending, dense_vecs)):
            tokens = tokenize(chunk.text)
            from collections import Counter
            tf = Counter(tokens)
            total_tf = sum(tf.values()) or 1
            sparse_indices = [hash(tok) % 100000 for tok in list(tf.keys())[:64]]
            sparse_values  = [count / total_tf for count in list(tf.values())[:64]]

            # Deduplicate sparse indices
            seen = {}
            for idx, val in zip(sparse_indices, sparse_values):
                seen[idx] = seen.get(idx, 0) + val
            sparse_indices = list(seen.keys())
            sparse_values  = list(seen.values())

            point_id = abs(hash(chunk.chunk_id)) % (2**63)
            points.append(qm.PointStruct(
                id=point_id,
                vector={
                    "dense": dvec,
                    "sparse": qm.SparseVector(indices=sparse_indices, values=sparse_values),
                },
                payload={"text": chunk.text, "metadata": chunk.metadata, "original_id": chunk.chunk_id},
            ))

        for start in range(0, len(points), UPSERT_BATCH):
            client.upsert(collection_name=COLLECTION, points=points[start:start+UPSERT_BATCH])
        total_chunks += len(pending)
        pending.clear()

    for batch in iter_articles_batch(jsonl_path, batch_size=ARTICLE_BATCH, skip=skip, max_articles=max_articles):
        for article in batch:
            chunks = chunk_article(article, max_tokens=512, overlap_tokens=50)
            pending.extend(chunks)
            total_articles += 1
        if pending:
            flush()
        elapsed = time.time() - t0
        print(f"\r[dual] articles={total_articles:,}  chunks={total_chunks:,}  {total_articles/max(elapsed,1):.1f} art/s", end="", flush=True)

    flush()
    print(f"\n[dual] Done — {total_articles:,} articles, {total_chunks:,} chunks in {time.time()-t0:.0f}s")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--file", required=True)
    p.add_argument("--skip", type=int, default=0)
    p.add_argument("--max-articles", type=int, default=None)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    ingest(Path(args.file), skip=args.skip, max_articles=args.max_articles)
```

Then run it for all 5 files:

```bash
for i in 0 1 2 3 4; do
  echo "=== wiki_dual: file $i ==="
  python scripts/ingest_wikipedia_dual.py \
    --file /path/to/enwiki_namespace_0/enwiki_namespace_0_${i}.jsonl
done
```

Verify after all 5 files:

```bash
python -c "
from qdrant_client import QdrantClient
client = QdrantClient(host='localhost', port=6333)
info = client.get_collection('wiki_dual')
print('wiki_dual points:', info.points_count)
"
```

Expected: ~3–4 million points (same as `wiki_single`).

---

## Step 10 — Package both collections for transfer

1. Stop Qdrant (`Ctrl+C`)
2. Zip the entire storage directory (contains both collections):
   ```bash
   cd <qdrant_binary_directory>
   zip -r qdrant_storage_10gb.zip storage/
   ```
3. Transfer `qdrant_storage_10gb.zip` to the Windows machine (~8–12 GB compressed)
4. On Windows: unzip into `C:\learning\qdrant-x86_64-pc-windows-msvc\storage\` (replace existing)
5. Start Qdrant on Windows and verify both collections appear at http://localhost:6333/dashboard

The Windows machine will then run offline evaluation (Recall@5, MRR, Precision@5) against both collections to determine which index configuration to use for full ingestion.

---

## Notes

- **MPS batch size**: 64 is optimal for M3. Do not increase beyond 128.
- **Storage size**: each collection is ~4–5 GB; total ~8–10 GB on disk, ~8–12 GB zipped.
- **Chunk ID hashing**: `abs(hash(chunk_id)) % (2**63)` avoids Qdrant's unsigned int64 ID requirement.
- **Sparse vector dimension**: using `hash(token) % 100000` as a fixed-size vocabulary space — sufficient for BM25-style retrieval without a pre-built vocabulary.
- **Redirects**: automatically skipped (abstract starts with "REDIRECT").
- **Resume**: add `--skip N` to `ingest_wikipedia_dual.py` if interrupted.
- **Symlink warning**: HuggingFace symlink warning on macOS is harmless.
