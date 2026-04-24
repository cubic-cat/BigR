# BigR

A lightweight RAG (Retrieval-Augmented Generation) system extended to support large-scale Wikipedia knowledge bases. Supports local vector stores for small-scale use and Qdrant for Wikipedia-scale ingestion.

## Features

- Flexible retrieval: `dense` | `sparse` | `hybrid` (RRF)
- Pluggable reranking: `keyword` | `cross_encoder`
- Wikipedia JSONL ingestion pipeline (Wikimedia Enterprise format)
- Pluggable embedding: local `BAAI/bge-m3` (sentence-transformers) or Qwen/OpenAI API
- Qdrant vector database backend for million-scale knowledge bases
- RAGAS-based chunking strategy evaluation framework

## Chunking Strategy Evaluation Results

Evaluated 4 configurations (100 Wikipedia articles, 8 questions, RAGAS 0.4.3):

| Strategy | Embedding | context_precision | faithfulness | answer_relevancy | **Overall avg** |
|----------|-----------|:-----------------:|:------------:|:----------------:|:---------------:|
| **fixed512** | **bge-m3** | **0.917** | **1.000** | 0.962 | **0.970** ★ |
| fixed256 | bge-m3 | 0.771 | 1.000 | 0.960 | 0.933 |
| section | qwen | 0.760 | 0.958 | 0.963 | 0.920 |
| section | bge-m3 | 0.765 | 0.875 | **0.971** | 0.903 |

context_recall = 1.000 for all strategies at top-k=5.

**Recommended configuration: `fixed512` chunking + `BAAI/bge-m3` local embedding**

See `chunking_strategy_evaluation_report.docx` for full analysis.

## Roadmap

### Next: Retrieval Index Comparison (on 10 GB pilot corpus)

Using fixed512 + bge-m3 as the confirmed baseline, compare two indexing approaches on the first 5 JSONL files (~10 GB, ~1.5M articles):

| Index type | Description |
|------------|-------------|
| **Single-vector** (dense only) | One bge-m3 vector per chunk; cosine similarity |
| **Dual-vector** (dense + sparse) | bge-m3 dense + BM25 sparse per chunk; RRF fusion |

Offline evaluation metrics (after building a ground-truth test set from the corpus):

| Metric | Target | Description |
|--------|--------|-------------|
| Recall@5 | > 0.75 | Core metric — fraction of relevant docs in top-5 |
| MRR | > 0.65 | Whether the first relevant result ranks high |
| Precision@5 | > 0.50 | Fraction of top-5 that are relevant (noise control) |

### Later

- Scale-up RAGAS evaluation (500–5000 articles) to validate chunking strategy findings
- Full ingestion of all 38 files (~75 GB) once index configuration is confirmed
- Expand evaluation question set with multi-hop, summarization, and comparison questions


## Project Structure

```
BigR/
├── configs/                     Configuration (embedding, LLM, vector DB)
├── core/
│   ├── embedding.py             EmbeddingClient (API) + LocalEmbeddingClient (bge-m3)
│   ├── retriever.py             Local JSON vector store (small-scale)
│   ├── qdrant_retriever.py      Qdrant backend (Wikipedia-scale)
│   ├── dense_retrieval.py       Dense retrieval
│   ├── sparse_retrieval.py      BM25 sparse retrieval
│   ├── hybrid_retrieval.py      RRF hybrid retrieval
│   ├── reranker.py              Reranking framework
│   ├── keyword_reranker.py      Keyword-based reranker
│   ├── cross_encoder_reranker.py  Cross-encoder reranker
│   ├── generator.py             LLM generation
│   └── rag_chain.py             End-to-end RAG chain
├── document/
│   ├── loader.py                Text file loader
│   ├── wiki_loader.py           Wikipedia JSONL parser
│   └── splitter.py              Chunking strategies (section-first, fixed-window)
├── scripts/
│   ├── build_kb.py              Small-scale knowledge base builder
│   ├── ingest_wikipedia.py      Wikipedia bulk ingestion pipeline
│   ├── eval_strategies.py       RAGAS chunking strategy evaluation
│   └── test_rag.py              Retrieval and RAG chain testing
├── chunking_strategy_evaluation_report.docx   Evaluation report
├── EVAL_RESULTS.md              Previous evaluation results (section vs fixed512, Qwen)
└── PROGRESS.md                  Project progress log
```

## Setup

### Install dependencies

```bash
pip install -r requirements.txt
```

### Configure environment

```bash
cp .env.example .env
# Edit .env and set your API keys
```

Minimum required for Wikipedia ingestion with bge-m3 (no API key needed):
```env
VECTOR_DB_PROVIDER=qdrant
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_VECTOR_SIZE=1024
```

For RAG generation and RAGAS evaluation, also set:
```env
QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
QWEN_API_KEY=your_key_here
```

### Start Qdrant

Download the Qdrant binary for your platform from https://github.com/qdrant/qdrant/releases, then:

```bash
# Must run from the directory containing the binary
cd /path/to/qdrant
./qdrant          # Linux/Mac
.\qdrant.exe      # Windows
```

Dashboard: http://localhost:6333/dashboard

## Wikipedia Ingestion

### Ingest with bge-m3 (recommended — free, local)

Edit `scripts/ingest_wikipedia.py` to use `LocalEmbeddingClient`:

```python
from core.embedding import build_embedding_client
embedding_client = build_embedding_client(model="BAAI/bge-m3", batch_size=64)
```

Then run:

```bash
python scripts/ingest_wikipedia.py \
  --file /path/to/enwiki_namespace_0_0.jsonl \
  --collection wikipedia_en \
  --embed-batch-size 64
```

### Resume an interrupted run

```bash
python scripts/ingest_wikipedia.py \
  --file /path/to/enwiki_namespace_0_5.jsonl \
  --collection wikipedia_en \
  --skip 150000
```

### Dry run (no embedding or upload)

```bash
python scripts/ingest_wikipedia.py \
  --file /path/to/enwiki_namespace_0_0.jsonl \
  --dry-run --max-articles 100
```

## Chunking Strategy Evaluation

Compare chunking strategies using RAGAS metrics:

```bash
# Set HF_HUB_OFFLINE=1 if bge-m3 is already downloaded
$env:HF_HUB_OFFLINE = "1"   # PowerShell

python scripts/eval_strategies.py \
  --max-articles 100 \
  --strategies section,fixed512,fixed256 \
  --embedding BAAI/bge-m3 \
  --also-qwen-section \
  --output results/eval_4way.json
```

Re-run evaluation on existing collections (skip re-ingestion):

```bash
python scripts/eval_strategies.py \
  --strategies section,fixed512,fixed256 \
  --embedding BAAI/bge-m3 \
  --skip-ingest \
  --output results/eval_4way.json
```

## Testing Retrieval

```bash
# List available retrieval and reranking methods
python scripts/test_rag.py --list-methods

# Retrieval-only test
python scripts/test_rag.py -q "Who discovered asteroid 1214 Richilde?"

# Full RAG chain (retrieval + LLM generation)
python scripts/test_rag.py -q "Who discovered asteroid 1214 Richilde?" --full-chain

# Specify retrieval and reranking methods
python scripts/test_rag.py -q "your question" \
  --retrieval-method hybrid \
  --rerank-method cross_encoder \
  --full-chain
```

## Key .env Settings

```env
# Retrieval
RETRIEVAL_METHOD=hybrid          # dense | sparse | hybrid
RERANK_ENABLED=true
RERANK_METHOD=keyword            # keyword | cross_encoder
RERANK_CANDIDATE_TOP_K=10

# Embedding (API mode)
EMBEDDING_MODEL_NAME=text-embedding-v4
EMBEDDING_BATCH_SIZE=10          # max 10 for text-embedding-v4

# Vector DB
VECTOR_DB_PROVIDER=qdrant
VECTOR_DB_COLLECTION_NAME=wikipedia_en
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_VECTOR_SIZE=1024
```
