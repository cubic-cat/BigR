# BigR

轻量级 RAG（检索增强生成）系统，已扩展支持大规模 Wikipedia 知识库。小规模场景使用本地向量库，Wikipedia 级别入库使用 Qdrant。

## 主要功能

- 灵活检索方式：`dense` | `sparse` | `hybrid`（RRF 融合）
- 可插拔重排：`keyword` | `cross_encoder`
- Wikipedia JSONL 批量入库流水线（Wikimedia Enterprise 格式）
- 可插拔 Embedding：本地 `BAAI/bge-m3`（sentence-transformers）或 Qwen/OpenAI API
- Qdrant 向量库后端，支持百万级知识库
- 基于 RAGAS 的分块策略评估框架

## 分块策略评测结果

4 种配置对比（100 篇 Wikipedia 文章，8 个问题，RAGAS 0.4.3）：

| 分块策略 | Embedding | context_precision | faithfulness | answer_relevancy | **综合均值** |
|----------|-----------|:-----------------:|:------------:|:----------------:|:------------:|
| **fixed512** | **bge-m3** | **0.917** | **1.000** | 0.962 | **0.970** ★ |
| fixed256 | bge-m3 | 0.771 | 1.000 | 0.960 | 0.933 |
| section | qwen | 0.760 | 0.958 | 0.963 | 0.920 |
| section | bge-m3 | 0.765 | 0.875 | **0.971** | 0.903 |

所有策略在 top-k=5 下 context_recall = 1.000。

**推荐配置：`fixed512` 分块 + `BAAI/bge-m3` 本地 Embedding**

完整分析见 `chunking_strategy_evaluation_report.docx`。

## 后续计划

### 下一步：检索索引方式对比（基于 10 GB 测试语料）

以 fixed512 + bge-m3 为确定基线，在前 5 个 JSONL 文件（约 10 GB，约 150 万篇文章）上对比两种索引方案：

| 索引方式 | 说明 |
|----------|------|
| **单向量**（仅 dense） | 每个 chunk 一个 bge-m3 向量，余弦相似度检索 |
| **双重向量**（dense + sparse） | 每个 chunk 同时建 bge-m3 稠密向量和 BM25 稀疏向量，RRF 融合 |

离线评估指标（构建测试集后跑）：

| 指标 | 目标值 | 说明 |
|------|--------|------|
| Recall@5 | > 0.75 | 核心指标——相关文档出现在 top-5 中的比例 |
| MRR | > 0.65 | 第一个相关结果是否排在前面 |
| Precision@5 | > 0.50 | top-5 中相关文档的比例（噪声是否可控） |

### 后续

- 扩大 RAGAS 评估规模（500–5000 篇），验证分块策略结论的稳定性
- 确定最优索引配置后，全量入库所有 38 个文件（约 75 GB）
- 补充多跳问题、摘要类问题和比较类问题，扩充评测问题集

## 项目结构

```
BigR/
├── configs/                     配置层（embedding、LLM、向量库）
├── core/
│   ├── embedding.py             EmbeddingClient (API) + LocalEmbeddingClient (bge-m3)
│   ├── retriever.py             本地 JSON 向量库（小规模）
│   ├── qdrant_retriever.py      Qdrant 后端（Wikipedia 规模）
│   ├── dense_retrieval.py       稠密检索
│   ├── sparse_retrieval.py      BM25 稀疏检索
│   ├── hybrid_retrieval.py      RRF 混合检索
│   ├── reranker.py              重排框架
│   ├── keyword_reranker.py      关键词重排
│   ├── cross_encoder_reranker.py  交叉编码器重排
│   ├── generator.py             LLM 生成
│   └── rag_chain.py             端到端 RAG 链
├── document/
│   ├── loader.py                文本文件加载器
│   ├── wiki_loader.py           Wikipedia JSONL 解析器
│   └── splitter.py              分块策略（section-first、固定窗口）
├── scripts/
│   ├── build_kb.py              小规模知识库构建脚本
│   ├── ingest_wikipedia.py      Wikipedia 批量入库流水线
│   ├── eval_strategies.py       RAGAS 分块策略评估脚本
│   └── test_rag.py              检索与 RAG 链测试脚本
├── chunking_strategy_evaluation_report.docx   评测报告
├── EVAL_RESULTS.md              早期评测结果（section vs fixed512，Qwen）
└── PROGRESS.md                  项目进度记录
```

## 环境配置

### 安装依赖

```bash
pip install -r requirements.txt
```

### 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填入 API Key
```

使用 bge-m3 本地入库时最少需要配置（不需要 API Key）：
```env
VECTOR_DB_PROVIDER=qdrant
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_VECTOR_SIZE=1024
```

RAG 生成和 RAGAS 评估还需要配置：
```env
QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
QWEN_API_KEY=your_key_here
```

### 启动 Qdrant

从 https://github.com/qdrant/qdrant/releases 下载对应平台的二进制文件，然后：

```bash
# 必须在二进制文件所在目录下启动
cd /path/to/qdrant
./qdrant          # Linux/Mac
.\qdrant.exe      # Windows
```

管理界面：http://localhost:6333/dashboard

## Wikipedia 入库

### 使用 bge-m3 入库（推荐——免费、本地运行）

编辑 `scripts/ingest_wikipedia.py`，切换为 `LocalEmbeddingClient`：

```python
from core.embedding import build_embedding_client
embedding_client = build_embedding_client(model="BAAI/bge-m3", batch_size=64)
```

然后运行：

```bash
python scripts/ingest_wikipedia.py \
  --file /path/to/enwiki_namespace_0_0.jsonl \
  --collection wikipedia_en \
  --embed-batch-size 64
```

### 断点续传

```bash
python scripts/ingest_wikipedia.py \
  --file /path/to/enwiki_namespace_0_5.jsonl \
  --collection wikipedia_en \
  --skip 150000
```

### 空跑模式（只解析不入库）

```bash
python scripts/ingest_wikipedia.py \
  --file /path/to/enwiki_namespace_0_0.jsonl \
  --dry-run --max-articles 100
```

## 分块策略评估

使用 RAGAS 对比多种分块策略：

```bash
# 如果 bge-m3 已下载，设置离线模式
$env:HF_HUB_OFFLINE = "1"   # PowerShell

python scripts/eval_strategies.py \
  --max-articles 100 \
  --strategies section,fixed512,fixed256 \
  --embedding BAAI/bge-m3 \
  --also-qwen-section \
  --output results/eval_4way.json
```

复用已有 collection，跳过重新入库：

```bash
python scripts/eval_strategies.py \
  --strategies section,fixed512,fixed256 \
  --embedding BAAI/bge-m3 \
  --skip-ingest \
  --output results/eval_4way.json
```

## 检索测试

```bash
# 列出可用的检索和重排方式
python scripts/test_rag.py --list-methods

# 仅检索测试
python scripts/test_rag.py -q "Who discovered asteroid 1214 Richilde?"

# 全链路测试（检索 + LLM 生成）
python scripts/test_rag.py -q "Who discovered asteroid 1214 Richilde?" --full-chain

# 指定检索和重排方式
python scripts/test_rag.py -q "your question" \
  --retrieval-method hybrid \
  --rerank-method cross_encoder \
  --full-chain
```

## 主要 .env 配置项

```env
# 检索
RETRIEVAL_METHOD=hybrid          # dense | sparse | hybrid
RERANK_ENABLED=true
RERANK_METHOD=keyword            # keyword | cross_encoder
RERANK_CANDIDATE_TOP_K=10

# Embedding（API 模式）
EMBEDDING_MODEL_NAME=text-embedding-v4
EMBEDDING_BATCH_SIZE=10          # text-embedding-v4 最大 batch=10

# 向量库
VECTOR_DB_PROVIDER=qdrant
VECTOR_DB_COLLECTION_NAME=wikipedia_en
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_VECTOR_SIZE=1024
```
