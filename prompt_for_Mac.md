# Wikipedia Ingestion on Mac (M3) — Instructions for Claude

You are running on a Mac with Apple M3 chip. Your job is to ingest Wikipedia JSONL files into a local Qdrant vector database using the bge-m3 embedding model (runs on MPS for fast inference). Follow the steps below in order. Do not skip steps.

---

## Context

This is part of a RAG (Retrieval-Augmented Generation) system called BigR. The goal is to embed Wikipedia articles as vector chunks and store them in Qdrant so the system can do semantic search.

- **Embedding model**: `BAAI/bge-m3` (local, sentence-transformers, MPS-accelerated on M3)
- **Vector DB**: Qdrant (local binary)
- **Chunking strategy**: section-first + sliding window fallback (already implemented)
- **Data files**: 5 Wikipedia JSONL files for the pilot run (`enwiki_namespace_0_0.jsonl` … `_4.jsonl`, total ~10 GB), transferred from the Windows machine
- **Qdrant storage**: will be saved locally, then transferred back to the Windows machine for testing

### Two-phase plan

| Phase | Files | Articles | Est. time (M3 MPS) | Purpose |
|-------|-------|----------|--------------------|---------|
| **Pilot** (do this first) | `_0` – `_4` (5 files, ~10 GB) | ~1.5M | ~20–40 min | Verify quality, test RAG system |
| **Full** (only if pilot looks good) | `_0` – `_37` (38 files, ~75 GB) | ~11.76M | ~2–4 hours | Production knowledge base |

**Start with the pilot. Do not run the full ingestion until the user confirms the pilot results are acceptable.**

---

## Step 0 — Verify the project files are present

Check that the following files exist in the project directory:

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
```

If `sentence-transformers` is not in requirements.txt, also run:
```bash
pip install sentence-transformers
```

---

## Step 2 — Install Qdrant

Download and run Qdrant as a local binary (no Docker needed):

```bash
# Download Qdrant for Apple Silicon
curl -L https://github.com/qdrant/qdrant/releases/latest/download/qdrant-aarch64-apple-darwin.tar.gz -o qdrant.tar.gz
tar -xzf qdrant.tar.gz
chmod +x qdrant
```

**Important**: Always start Qdrant from the directory where the binary lives, so `./storage` is created there:

```bash
# Start Qdrant in a separate terminal (keep it running during ingestion)
cd <directory_containing_qdrant_binary>
./qdrant
```

Verify it's running: open http://localhost:6333/dashboard in a browser.

---

## Step 3 — Configure .env

The `.env` file is already in the project. Open it and verify/update these lines:

```env
VECTOR_DB_PROVIDER=qdrant
VECTOR_DB_COLLECTION_NAME=wikipedia_en
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_VECTOR_SIZE=1024
```

The Qwen API key in `.env` is only needed for RAGAS evaluation (not needed for ingestion with bge-m3). Leave other values as-is.

---

## Step 4 — Download bge-m3 (first run only)

The model will auto-download on first use. To pre-download and verify:

```bash
cd <project_root>/BigR
python -c "
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('BAAI/bge-m3')
print('bge-m3 ready, dim =', model.get_embedding_dimension())
"
```

Expected output: `bge-m3 ready, dim = 1024`

The model is ~2.27 GB and downloads once, cached at `~/.cache/huggingface/hub/`.

If the download is slow or times out, use the China mirror:
```bash
HF_ENDPOINT=https://hf-mirror.com python -c "
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('BAAI/bge-m3')
print('bge-m3 ready, dim =', model.get_embedding_dimension())
"
```

---

## Step 5 — Verify bge-m3 runs on MPS

```bash
python -c "
import torch
from sentence_transformers import SentenceTransformer
print('MPS available:', torch.backends.mps.is_available())
model = SentenceTransformer('BAAI/bge-m3', device='mps')
vecs = model.encode(['test sentence'] * 64, batch_size=64)
print('Shape:', vecs.shape)
print('MPS inference OK')
"
```

Expected: `MPS available: True` and `Shape: (64, 1024)`

---

## Step 6 — Update ingest_wikipedia.py to use bge-m3

The current `ingest_wikipedia.py` uses `EmbeddingClient` (Qwen API). You need to switch it to `LocalEmbeddingClient` (bge-m3). Edit the file:

In `scripts/ingest_wikipedia.py`, find this block (around line 218):

```python
from core.embedding import EmbeddingClient
embedding_client = EmbeddingClient()
if not embedding_client.is_configured():
    print("[ERROR] Embedding API key missing. Set EMBEDDING_API_KEY in .env", file=sys.stderr)
    sys.exit(1)
```

Replace it with:

```python
from core.embedding import build_embedding_client
embedding_client = build_embedding_client(model="BAAI/bge-m3", batch_size=64)
```

Also change the default `--embed-batch-size` argument from `10` to `64` (line ~100):

```python
p.add_argument("--embed-batch-size", type=int, default=64, ...)
```

---

## Step 7 — Smoke test (100 articles)

Before running the pilot, verify everything works end-to-end:

```bash
cd <project_root>/BigR
python scripts/ingest_wikipedia.py \
  --file /path/to/enwiki_namespace_0/enwiki_namespace_0_0.jsonl \
  --max-articles 100 \
  --collection wikipedia_en_test \
  --embed-batch-size 64
```

Expected output after ~1 minute:
```
[ingest] Done in ...s
[ingest] Articles processed : ~98  (some are redirects, skipped)
[ingest] Chunks ingested    : ~200
[ingest] Qdrant points      : ~200
```

If this works, proceed to the pilot.

---

## Step 8 — Pilot ingestion (files 0–4, ~10 GB)

Ingest the first 5 files into collection `wikipedia_en_pilot`. This covers ~1.5 million articles and takes about 20–40 minutes on M3 MPS.

```bash
for i in 0 1 2 3 4; do
  echo "=== Ingesting file $i ==="
  python scripts/ingest_wikipedia.py \
    --file /path/to/enwiki_namespace_0/enwiki_namespace_0_${i}.jsonl \
    --collection wikipedia_en_pilot \
    --embed-batch-size 64
done
```

After all 5 files finish, verify the collection:

```bash
python -c "
from qdrant_client import QdrantClient
client = QdrantClient(host='localhost', port=6333)
info = client.get_collection('wikipedia_en_pilot')
print('Points:', info.points_count)
"
```

Expected: ~3–4 million points.

**Stop here.** Package the storage and transfer it back to the Windows machine so the user can test the RAG system quality before committing to the full run.

---

## Step 9 — Package pilot storage for transfer

1. Stop Qdrant (`Ctrl+C` in its terminal)
2. Zip the storage directory:
   ```bash
   cd <qdrant_binary_directory>
   zip -r qdrant_storage_pilot.zip storage/
   ```
3. Transfer `qdrant_storage_pilot.zip` to the Windows machine (~4–6 GB compressed)
4. On Windows: unzip into `C:\learning\qdrant-x86_64-pc-windows-msvc\storage\` (replacing existing storage)
5. Start Qdrant on Windows and verify via http://localhost:6333/dashboard

The user will then test the RAG system and decide whether to proceed with full ingestion.

---

## Step 10 — Full ingestion (only if user confirms pilot is good)

Only run this after the user has tested the pilot and confirmed the quality is acceptable.

Run all 38 files sequentially into collection `wikipedia_en`. Each file takes ~3–7 minutes on M3 MPS; total ~2–4 hours.

```bash
for i in $(seq 0 37); do
  echo "=== Ingesting file $i ==="
  python scripts/ingest_wikipedia.py \
    --file /path/to/enwiki_namespace_0/enwiki_namespace_0_${i}.jsonl \
    --collection wikipedia_en \
    --embed-batch-size 64
done
```

**If a run is interrupted**, resume with `--skip N` where N is the number of lines already processed (shown in the progress output):

```bash
python scripts/ingest_wikipedia.py \
  --file /path/to/enwiki_namespace_0/enwiki_namespace_0_5.jsonl \
  --collection wikipedia_en \
  --embed-batch-size 64 \
  --skip 150000
```

After all 38 files, verify:

```bash
python -c "
from qdrant_client import QdrantClient
client = QdrantClient(host='localhost', port=6333)
info = client.get_collection('wikipedia_en')
print('Points:', info.points_count)
"
```

Expected: ~29 million points.

---

## Notes

- **MPS batch size**: 64 is optimal for M3. Larger batches (128+) may not be faster due to memory bandwidth.
- **Pilot storage size**: ~4–5 GB (5 files). Full Wikipedia storage: ~168 GB. Make sure the Mac has enough disk space before starting the full run.
- **Collection name**: use the same collection name consistently across all files — they all upsert into the same collection.
- **Redirects**: articles whose abstract starts with "REDIRECT" are automatically skipped by the ingestion script.
- **Symlink warning**: HuggingFace may warn about symlinks on macOS — this is harmless, the model will still work.
