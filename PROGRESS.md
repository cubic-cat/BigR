# 项目进度记录

> 最后更新：2026-04-23

---

## 一、项目目标

将 Wikimedia Enterprise 提供的英文 Wikipedia 预解析数据集（79 GB，38 个 JSONL 文件，约 1176 万篇文章）接入已有的 BigR RAG 系统，建立可检索的知识库，支持语义问答。

---

## 二、当前系统架构

```
BigR/
├── configs/          配置层（embedding、LLM、向量库）
├── core/             核心检索与生成
│   ├── embedding.py          Embedding 客户端（Qwen/OpenAI）
│   ├── retriever.py          本地 JSON 向量库（小规模）
│   ├── qdrant_retriever.py   Qdrant 向量库（★ 新增，大规模专用）
│   ├── dense_retrieval.py    稠密检索
│   ├── sparse_retrieval.py   BM25 稀疏检索
│   ├── hybrid_retrieval.py   RRF 混合检索
│   ├── reranker.py           重排序框架
│   ├── keyword_reranker.py   关键词重排序
│   ├── cross_encoder_reranker.py  交叉编码器重排序
│   ├── generator.py          LLM 生成
│   └── rag_chain.py          端到端 RAG 链（★ 已更新，支持 qdrant 后端）
├── document/         文档处理层
│   ├── loader.py             原有文本文件加载器
│   ├── wiki_loader.py        ★ 新增：Wikipedia JSONL 解析器
│   ├── splitter.py           ★ 重写：文本分块器（section-first + 滑动窗口）
│   └── preprocessor.py       占位（暂未实现）
└── scripts/
    ├── build_kb.py           原有小规模建库脚本
    ├── test_rag.py           原有检索测试脚本
    ├── ingest_wikipedia.py   ★ 新增：Wikipedia 批量入库流水线
    └── eval_strategies.py    ★ 新增：RAGAS 分块策略评估脚本
```

---

## 三、已完成工作

### 3.1 Wikipedia 解析器（`document/wiki_loader.py`）

- 解析 Wikimedia Enterprise 格式的 JSONL（每行一篇文章）
- 正确处理嵌套 section 结构（`has_parts` 中 paragraph / section 递归提取）
- 跳过 Abstract section（内容已在 `article.abstract` 字段中）
- 过滤 list、image、table 等非正文 part 类型
- 支持 `max_articles`、`skip`（断点续传）参数

### 3.2 文本分块器（`document/splitter.py`）

实现两层分块策略：
1. **主策略（section-first）**：以 Wikipedia section 为单位切割，保留语义完整性
2. **次策略（滑动窗口）**：对超过 max_tokens（默认 512）的 section 二次切割，50 token 重叠
3. 每个 chunk 携带完整元数据：`article_id`、`title`、`url`、`section`、`section_depth`、`categories`

同时提供 `split_text()` 用于普通纯文本的固定窗口切割（fixed512 / fixed256 策略均使用此函数）。

### 3.3 Qdrant 检索器（`core/qdrant_retriever.py`）

- 与 `LocalVectorRetriever` 接口完全兼容，可在 `RAGChain` 中直接替换
- 支持 dense / sparse（客户端 BM25）/ hybrid（RRF）三路检索
- 使用 `qdrant-client` 1.9+ 新 API（`query_points`，非废弃的 `search`）
- 支持 `QDRANT_HOST`、`QDRANT_PORT`、`QDRANT_URL`、`QDRANT_API_KEY` 环境变量配置

### 3.4 批量入库流水线（`scripts/ingest_wikipedia.py`）

- 流式读取 JSONL，按 article batch 分块→嵌入→upsert，内存占用低
- 支持 `--skip N` 断点续传
- 支持 `--dry-run` 仅统计不入库
- 进度实时打印（articles/s、chunks 数）
- `--embed-batch-size` 默认已修正为 10（`text-embedding-v4` 的 API 上限）

### 3.7 本地 Embedding 支持（`core/embedding.py`）

- 新增 `LocalEmbeddingClient`：基于 sentence-transformers，支持任意 HuggingFace 模型
- 自动设备检测：CUDA → MPS (Apple Silicon) → CPU
- 懒加载模型（首次调用时才下载/初始化）
- 新增 `build_embedding_client(model, batch_size)` 工厂函数：
  - 含 `/` 的名称（如 `BAAI/bge-m3`）→ LocalEmbeddingClient
  - 不含 `/`（如 `text-embedding-v4`）→ EmbeddingClient（Qwen API）

### 3.8 eval_strategies.py 支持多 Embedding 模型对比

新增参数：
- `--embedding BAAI/bge-m3`（默认）或 `--embedding text-embedding-v4`（Qwen API）
- `--embed-batch-size`：覆盖默认 batch size（本地默认 32，API 默认 10）
- `--also-qwen-section`：将已有 `eval_section`（Qwen 入库）作为对比基线加入评估

Collection 命名规则：`eval_{strategy}_{model_tag}`（如 `eval_section_bgem3`、`eval_fixed512_bgem3`）

4-way 对比测试命令：
```powershell
# 先跑 bge-m3 的三种分块策略
python scripts\eval_strategies.py `
  --max-articles 100 `
  --strategies section,fixed512,fixed256 `
  --embedding BAAI/bge-m3 `
  --also-qwen-section `
  --output results\eval_4way.json
```

- 对比多种分块策略：`section`、`fixed512`、`fixed256`
- 4 项 RAGAS 指标：`context_precision`、`context_recall`、`faithfulness`、`answer_relevancy`
- 使用 `AsyncOpenAI` 客户端（RAGAS 0.4.x 要求）
- LLM `max_tokens=4096`（避免 RAGAS JSON 推理链被截断导致失败）
- 支持自定义问题集（`--questions`）、保存完整结果（`--output`）
- 内置 8 个基于已入库文章的评估问题

### 3.6 配置与依赖更新

- `.env` 已创建，Qdrant + Qwen 配置就绪
- `requirements.txt` 新增：`qdrant-client>=1.9.0`、`tqdm>=4.66.0`
- `ragas`、`datasets` 已 pip 安装
- `.env.example` 补充了全部 Qdrant 参数说明

---

## 四、环境与配置状态

### Qdrant
- **版本**：1.17.1（本地可执行文件）
- **路径**：`C:\learning\qdrant-x86_64-pc-windows-msvc\qdrant.exe`
- **启动方式**：在该目录下运行 `.\qdrant.exe`（必须在此目录启动，否则 `./storage` 权限错误）
- **Web UI**：http://localhost:6333/dashboard
- **当前 collections**：
  - `rag_knowledge_base`（原有，空）
  - `eval_section`（100 篇测试，195 chunks）
  - `eval_fixed512`（100 篇测试，115 chunks）

### Embedding
- **API 模型**：`text-embedding-v4`（阿里云 Qwen DashScope）—— 维度 1024，batch 上限 10，¥0.0007/1K tokens
- **本地模型**：`BAAI/bge-m3`（sentence-transformers，CPU Intel i7）—— 维度 1024，batch=64 最优，约 442 chunks/s
  - 模型缓存路径：`C:\Users\I605229\.cache\torch\sentence_transformers\BAAI_bge-m3`
  - 全量入库预估：2922万 chunks ÷ 442 chunks/s ≈ **18.3 小时**（单进程 CPU）

### LLM（生成 + RAGAS 评估）
- **模型**：`qwen-plus`
- **用途**：RAG 生成答案 + RAGAS 评估打分

### 数据文件
- **位置**：`C:\learning\enwiki_namespace_0\`
- **文件数**：38 个 JSONL 文件
- **总大小**：75 GB
- **每文件**：约 30 万篇文章，2 GB
- **总文章数**：约 1176 万篇（其中 1.4% 是重定向页）

---

## 五、待完成工作

### 优先级高

- [ ] **运行 4-way 对比测试**（bge-m3 × 3策略 + Qwen section 基线）：
  ```powershell
  python scripts\eval_strategies.py --max-articles 100 --strategies section,fixed512,fixed256 --embedding BAAI/bge-m3 --also-qwen-section --output results\eval_4way.json
  ```
  前提：确认 bge-m3 模型已下载完成（`C:\Users\I605229\.cache\torch\sentence_transformers\BAAI_bge-m3`）
- [ ] **确定最优方案后扩大评估**：500–5000 篇，三策略对比，确定最终入库方案
- [ ] **实现多进程并行入库**（`scripts/ingest_parallel.py`）：6 进程并行处理不同 JSONL 文件

### 优先级中

- [ ] **全量入库**：选定策略后入库全部 38 个文件（约 ¥1984 / 7 天）
- [ ] **更新 `scripts/test_rag.py`**：增加 `--backend qdrant` 参数，切换检索后端

### 优先级低

- [ ] 实现 `document/preprocessor.py`（文本清洗、去重）
- [ ] 自定义评估问题集（覆盖更多 Wikipedia 主题域）
- [ ] 探索本地 embedding 模型（如 BGE-M3）降低全量入库成本和时间（✅ 已实现 LocalEmbeddingClient，待实际运行）

---

## 六、已知问题与注意事项

1. **Qdrant 启动目录**：必须在 `C:\learning\qdrant-x86_64-pc-windows-msvc\` 目录下启动，否则报 `拒绝访问` 错误
2. **embedding batch size**：`text-embedding-v4` 最大 batch=10，超过会报 400 错误，已在 `.env` 和脚本默认值中修正
3. **RAGAS LLM max_tokens**：必须设为 4096+，否则 Faithfulness/ContextRecall 的 JSON 推理链会被截断，导致全部评分失败
4. **RAGAS 客户端**：必须使用 `AsyncOpenAI`（非 `OpenAI`），RAGAS 0.4.x 的 `ascore()` 要求异步客户端
5. **中断续传**：入库中断后用 `--skip N` 参数跳过已处理行数恢复
