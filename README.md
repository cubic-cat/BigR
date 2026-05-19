# BigR

轻量级 RAG（检索增强生成）系统，已扩展支持大规模 Wikipedia 知识库。小规模场景使用本地向量库，Wikipedia 级别入库使用 Qdrant。

## 主要功能

- 灵活检索方式：`dense` | `sparse` | `hybrid`（RRF 融合）
- 可插拔重排：`keyword` | `cross_encoder`
- Wikipedia JSONL 批量入库流水线（Wikimedia Enterprise 格式）
- 可插拔 Embedding：本地 `BAAI/bge-m3`（sentence-transformers）或 Qwen/OpenAI API
- Qdrant 向量库后端，支持百万级知识库
- 基于 RAGAS 的分块策略评估框架
- 多维度检索方案对比评测体系
- 语义QA基准测试与端到端 RAGAS 评估

---

## 数据集与索引信息

### Embedding 模型

| 属性 | 值 |
|------|-----|
| 模型名称 | `BAAI/bge-small-en-v1.5` |
| 向量维度 | 384 |
| 距离度量 | Cosine |
| 来源 | HuggingFace sentence-transformers |
| 归一化 | L2 normalize（encode 时 `normalize_embeddings=True`） |

### 数据源文件

使用 Wikimedia Enterprise JSONL 格式的英文 Wikipedia 数据，共 5 个文件：

| # | 文件名 | 大小（约） |
|---|--------|-----------|
| 1 | `enwiki_namespace_0_0.jsonl` | ~2 GB |
| 2 | `enwiki_namespace_0_3.jsonl` | ~2 GB |
| 3 | `enwiki_namespace_0_6.jsonl` | ~2 GB |
| 4 | `enwiki_namespace_0_9.jsonl` | ~2 GB |
| 5 | `enwiki_namespace_0_12.jsonl` | ~2 GB |

**共计约 10 GB 原始语料。**

### Qdrant 集合信息

| 属性 | 值 |
|------|-----|
| 集合名称 | `documents` |
| 总向量数（chunks） | **1,963,836** |
| 索引类型 | HNSW |
| HNSW 参数 | m=16, ef_construct=100 |
| 分段数 | 7 |
| 分块策略 | fixed512（512 token 固定窗口） |

---

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

---

## 检索方案对比评测

### 评测概述

在已入库的 196 万 chunks 集合上，对比三种检索策略在多种评测维度下的表现。

**被评测的检索方案：**

| 方案 | 实现方式 | 说明 |
|------|---------|------|
| **Dense (HNSW)** | Qdrant `query_points()` + HNSW 近似最近邻 | 384 维 bge-small-en-v1.5 向量检索 |
| **BM25** | 客户端 BM25 评分（k1=1.5, b=0.75） | 对文档池全量遍历打分 |
| **Hybrid-RRF** | Dense(2K) + BM25(2K) → Reciprocal Rank Fusion (k=20) | 双路召回 + RRF 融合排序 |

**评测维度（共 4 项）：**

1. Recall@K 曲线（不同 query 难度）
2. Recall@K 随 K 增大的变化趋势
3. 跨文章语义关联发现能力
4. Query 退化鲁棒性

---

### 评测一：基础 Recall@K（不同 Query 难度）

**实验设计**：从集合中采样文章，分别用"文章标题"、"首句摘要"、"全文片段"作为 query，衡量能否检索到源文章的 chunks。

#### 结果：Query = 文章标题（最难，模拟真实用户提问）

| 方案 | Recall@5 | Precision@5 | MRR | Latency |
|------|:--------:|:-----------:|:---:|:-------:|
| Dense | 0.8150 | 0.163 | 0.7904 | 21.7 ms |
| **BM25** | **1.0000** | **0.200** | **0.9967** | 12.4 ms |
| Hybrid-RRF | 1.0000 | 0.200 | 0.9097 | 32.8 ms |

> 测试集: 227 篇文章, BM25 pool: 10,000 docs

#### 结果：Query = 首句摘要（中等难度）

| 方案 | Recall@5 | Precision@5 | MRR | Latency |
|------|:--------:|:-----------:|:---:|:-------:|
| Dense | 0.9690 | — | 0.9639 | 32.4 ms |
| **BM25** | **0.9912** | — | **0.9841** | 356.2 ms |
| Hybrid-RRF | 0.9867 | — | 0.9745 | 391.7 ms |

> 测试集: 226 篇文章, BM25 pool: 50,000 docs

#### 结果：Query = 全文片段（简单，调试用）

| 方案 | Recall@5 | Precision@5 | MRR | Latency |
|------|:--------:|:-----------:|:---:|:-------:|
| Dense | 1.0000 | 0.200 | 0.9967 | 27.6 ms |
| BM25 | 1.0000 | 0.200 | 1.0000 | 168.5 ms |
| Hybrid-RRF | 1.0000 | 0.200 | 1.0000 | 192.6 ms |

> 测试集: 151 篇文章, BM25 pool: 10,000 docs（所有方案表现一致）

---

### 评测二：Recall@K 曲线（K=5, 10, 20, 50）

**实验设计**：固定 query 难度，增大 K 值，观察 Recall 是否随 K 增长——即"找不到"是排名靠后还是根本不在索引覆盖范围内。

#### Query = 文章标题（BM25 pool = 50,000）

| 方案 | R@5 | R@10 | R@20 | R@50 | MRR |
|------|:---:|:----:|:----:|:----:|:---:|
| Dense | 0.8150 | 0.8194 | 0.8194 | 0.8194 | 0.7909 |
| **BM25** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **0.9978** |
| Hybrid-RRF | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9046 |

> 测试集: 227 篇文章

#### Query = 首句摘要（BM25 pool = 50,000）

| 方案 | R@5 | R@10 | R@20 | R@50 | MRR |
|------|:---:|:----:|:----:|:----:|:---:|
| Dense | 0.9690 | 0.9690 | 0.9690 | 0.9690 | 0.9639 |
| BM25 | 0.9912 | 0.9912 | 0.9956 | 0.9956 | 0.9841 |
| Hybrid-RRF | 0.9867 | 0.9912 | 0.9912 | 0.9956 | 0.9745 |

> 测试集: 226 篇文章

**关键发现**：
- Dense 的 Recall 曲线**完全扁平**（R@5 = R@10 = R@20 = R@50），说明有 ~18% 的文章在任何 K 值下都检索不到，是**硬性召回盲区**，而非排序问题
- BM25 在 K=5 即达到几乎完美召回，无需更大 K 值

---

### 评测三：跨文章语义关联发现能力

**实验设计**：去除 query 中的文章标题词，测试检索方法能否发现"语义相关但来自不同文章"的内容。衡量语义泛化能力。

| 指标 | Dense | BM25 |
|------|:-----:|:----:|
| 同源文章命中（占 Top-20 全部 slot） | 3.97% | 4.77% |
| **跨文章命中**（占 Top-20 全部 slot） | **96.03%** | **95.23%** |
| 平均每 query 发现的不同文章数 | 18.29 | 19.03 |
| 源文章出现在 Top-20 的比例 | 100% | 100% |

> 测试集: 150 queries, Top-K=20, BM25 pool=50,000

**结论**：Dense 和 BM25 在跨文章关联发现上**无显著差异**。两者均能发现大量来自其他文章的相关内容。

---

### 评测四：Query 退化鲁棒性测试

**实验设计**：逐步从 query 中剥离信息，测试哪种方法在 query 质量下降时仍能保持高召回。

**退化级别定义：**

| 级别 | 操作 | 难度 | 示例 |
|------|------|------|------|
| L0 | 原文首句（完整） | 简单 | "Bendemeer is a subzone within the planning area of Kallang..." |
| L1 | 首句去除标题词 | 中等 | "is a subzone within the planning area of ..." |
| L2 | 仅保留内容词（去停词、去标题词） | 困难 | "subzone within planning area Kallang defined..." |
| L3 | 关键词打乱顺序 | 极难 | "boundary Authority Redevelopment Urban defined..." |

#### 结果（Top-K = 20, BM25 pool = 50,000）

| 退化级别 | Dense R@20 | Dense AvgRank | BM25 R@20 | BM25 AvgRank |
|---------|:----------:|:-------------:|:---------:|:------------:|
| **L0: 完整首句** | 0.9932 | 1.0 | 1.0000 | 1.0 |
| **L1: 去标题词** | 0.6573 | 3.6 | 0.9650 | 1.4 |
| **L2: 仅内容词** | 0.5385 | 4.2 | 0.9371 | 1.8 |
| **L3: 打乱关键词** | 0.5533 | 4.3 | 0.9733 | 1.4 |

**关键发现**：
- Dense Recall 从 L0 的 0.993 **暴跌**至 L1 的 0.657（降幅 34%），进一步降至 L2/L3 的 ~0.54
- BM25 在所有退化级别均保持 **>0.93** 的 Recall
- L3（词序完全打乱）BM25 Recall 甚至回升至 0.973，证明 BM25 不依赖词序
- **结论：Dense 的高 Recall 主要依赖文本重叠（verbatim match），并非真正的语义理解**

---

### 综合结论

#### 各方案最终评价

| 维度 | Dense (HNSW) | BM25 | Hybrid-RRF |
|------|:------------:|:----:|:----------:|
| 基础 Recall（title query） | 0.815 | **1.000** | **1.000** |
| Recall@K 是否随 K 增长 | 否（扁平线） | 是（已满） | 是（已满） |
| 退化鲁棒性（L3） | 0.553 | **0.973** | — |
| 跨文章多样性 | 18.3 | **19.0** | — |
| 延迟 | **~22 ms** | ~12-356 ms* | ~33-392 ms |
| 召回天花板 | 有（~18% 盲区） | **无** | **无** |

> *BM25 延迟取决于 pool 大小：10K pool ≈ 12ms，50K pool ≈ 356ms

#### 核心结论

1. **在当前配置下（bge-small-en-v1.5 384d + Wikipedia 长文本），BM25 全面优于 Dense 检索**
2. Dense 存在 ~18% 的硬性召回盲区，无法通过增大 K 解决
3. Dense 的"语义理解"能力有限——退化测试证实其高 Recall 依赖文本重叠
4. Hybrid-RRF 相比纯 BM25 没有额外增益（BM25 已接近 1.0 天花板）
5. BM25 对 query 退化极其鲁棒（词序打乱后仍 0.97）

#### 推荐方案

| 场景 | 推荐 | 原因 |
|------|------|------|
| 当前数据集最优 | **BM25** | Recall 最高，鲁棒性最强 |
| 低延迟要求 | BM25（小 pool）或 Dense | Dense 延迟稳定 ~22ms |
| 加 Rerank | BM25 Top-20 → Cross-Encoder | 给 Reranker 提供高质量候选 |
| 提升 Dense | 换用大模型（bge-m3 1024d）+ **重建索引** | 可能解决语义理解不足 |

---

## 语义QA补充评估

### 概述

本补充评估流程用于测试当前启发式基准测试（title / abstract / degradation queries）是否引入了有利于BM25的**词法偏差**。通过生成LLM-based语义QA对（低词法重叠）并运行RAGAS端到端评估来验证。

**定位**：这是一个*补充*评估，**不是**替代现有检索基准测试。

### RAGAS评估结果

| 指标 | bm25 | bm25+cross_encoder | dense |
|------|------|-------------------|-------|
| 上下文精确率 | 0.4545 | **0.5581** | 0.4159 |
| 上下文召回率 | 0.3900 | **0.4100** | 0.3700 |
| 忠实度 | **0.9383** | 0.8835 | 0.9021 |
| 答案相关性 | 0.4077 | 0.4014 | **0.4481** |
| 平均检索延迟(ms) | 714.3 | 18361.0 | 12148.9 |

### 关键发现

1. **词法偏差分析**：BM25和Dense在语义QA条件下的context_recall表现相当（0.39 vs 0.37），启发式基准测试中的词法偏差可能略微夸大了BM25的优势，但差异并不显著。

2. **重排效果分析**：CrossEncoder重排在上下文精确率方面提供了明显改善（+22.7%），在语义QA条件下，重排有助于过滤不相关的上下文。

3. **QA基准测试词法重叠**：平均词法重叠为0.335（参考：启发式标题查询通常重叠 > 0.8），重叠越低表示词法偏差越小，查询越贴近真实用户。

### 语义QA评估结论

1. BM25即使在语义QA条件下仍保持对Dense检索的优势，表明其优势并非纯粹源于词法偏差。
2. CrossEncoder重排在语义QA条件下显示明显的上下文精确率改善。
3. 在语义QA测试集下，BM25与Dense差距显著缩小（从启发式测试的18.5%降至5.1%）。

### 快速开始

```bash
# 安装增量依赖
pip install -r scripts/requirements_semantic_qa.txt

# 生成QA基准测试（50个问题）
python scripts/generate_llm_qa_dataset.py --num_samples 50 --seed 42

# 运行RAGAS评估
python scripts/run_ragas_evaluation.py --retrieval_method bm25
python scripts/run_ragas_evaluation.py --retrieval_method dense
python scripts/run_ragas_evaluation.py --retrieval_method bm25+cross_encoder

# 生成评估报告
python scripts/generate_eval_report.py
```

---

## 后续计划

- 在 BM25 Top-20 基础上测试 Cross-Encoder Rerank（BAAI/bge-reranker-base）
- 评估更大 Embedding 模型（bge-m3 1024d）能否消除 Dense 的召回盲区
- 如延迟敏感：用 `ingest_wikipedia_dual.py` 重建双索引集合（Qdrant 原生 BM25）
- 扩大评估规模至 500+ 文章，验证结论稳定性

---

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
│   ├── rag_chain.py             端到端 RAG 链
│   └── query_enhancer.py        查询增强（rewrite/expansion）
├── document/
│   ├── loader.py                文本文件加载器
│   ├── wiki_loader.py           Wikipedia JSONL 解析器
│   └── splitter.py              分块策略（section-first、固定窗口）
├── scripts/
│   ├── build_kb.py              小规模知识库构建脚本
│   ├── ingest_wikipedia.py      Wikipedia 批量入库流水线
│   ├── ingest_wikipedia_dual.py 双索引入库（Qdrant原生BM25）
│   ├── eval_strategies.py       RAGAS 分块策略评估脚本
│   ├── eval_retrieval_comparison.py   检索方案对比评测（Dense/BM25/Hybrid）
│   ├── eval_retrieval_rerank.py  重排效果评估
│   ├── eval_degradation.py      Query 退化鲁棒性测试
│   ├── eval_cross_article.py    跨文章语义关联评测
│   ├── generate_test_set.py     生成测试集
│   ├── generate_llm_qa_dataset.py   生成LLM语义QA基准测试
│   ├── run_ragas_evaluation.py  运行RAGAS端到端评估
│   ├── generate_eval_report.py  自动生成Markdown评估报告
│   ├── generate_retrieval_report.py  生成检索报告
│   ├── generate_ragas_charts.py 生成RAGAS评估图表
│   ├── analyze_lexical_overlap.py   词法重叠分析
│   ├── run_all_ragas_eval.py    自动执行完整RAGAS评估流程
│   ├── README_semantic_qa.md    语义QA评估说明文档
│   ├── requirements_semantic_qa.txt  增量依赖
│   └── test_rag.py              检索与 RAG 链测试脚本
├── results/                     评测结果（JSON）
│   ├── recall_curve_title.json          Recall@K 曲线（title query）
│   ├── recall_curve_abstract.json       Recall@K 曲线（abstract query）
│   ├── degradation_test.json            Query 退化测试结果
│   ├── cross_article_eval_v2.json       跨文章关联评测结果
│   ├── retrieval_comparison_title.json  基础对比（title query）
│   └── retrieval_comparison_abstract.json  基础对比（abstract query）
├── outputs/
│   ├── qa_benchmark/            LLM生成的语义QA基准测试
│   ├── ragas_eval/              RAGAS评估结果
│   └── reports/                 评估报告（ragas_eval_report.md）
├── chunking_strategy_evaluation_report.docx   评测报告
├── EVAL_RESULTS.md              早期评测结果（section vs fixed512，Qwen）
└── PROGRESS.md                  项目进度记录
```

---

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

---

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

---

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

---

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

---

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

# Cross-Encoder重排
CROSS_ENCODER_MODEL=BAAI/bge-reranker-base
CROSS_ENCODER_DEVICE=
CROSS_ENCODER_BATCH_SIZE=8
CROSS_ENCODER_MAX_LENGTH=512
```
