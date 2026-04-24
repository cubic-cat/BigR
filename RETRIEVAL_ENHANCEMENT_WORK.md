# BigR 检索增强扩展说明

## 本次完成内容
在不破坏原有代码结构的前提下，新增了“查询增强 + 检索/重排实验框架 + 自动报告”能力。

### 1) Query Enhancement（新增）
- 新增文件：`core/query_enhancer.py`
- 新增统一接口：`QueryEnhancer.enhance(query) -> List[str]`
- 已实现策略：
  - `IdentityEnhancer`（原样查询）
  - `SimpleRewriteEnhancer`（LLM 单条改写）
  - `QueryExpansionEnhancer`（LLM 多查询扩展，3~5条）
- 设计要点：
  - prompt 函数独立封装
  - 工厂函数 `create_query_enhancer()` 支持插件式切换
  - 保留 `enhance_batch()` 便于后续批处理扩展

### 2) RAG 链路可选增强（扩展）
- 扩展文件：`core/rag_chain.py`
- 在 `search/ask` 中新增（向后兼容）可选参数：
  - `enable_query_enhancer: bool = False`
  - `query_enhancer_type: str = "identity"`
  - `query_merge_method: str = "rrf"`（支持 `rrf` / `weighted`）
  - `retrieval_method: str | None = None`（可显式指定 dense/sparse/hybrid）
- 行为保证：
  - 不开启 enhancer 时与历史行为一致
  - 开启后执行 multi-query retrieval，并做结果融合
  - 可选复用已有 rerank 机制进行最终排序

### 3) Rerank 实验适配层（新增）
- 新增文件：`core/rerank_experiments.py`
- 新增统一接口：`RerankStrategy.rerank(query, docs, top_k)`
- 已实现策略：
  - `NoRerank`（baseline）
  - `KeywordRerankAdapter`（调用已有 keyword reranker）
  - `CrossEncoderRerankAdapter`（调用已有 cross-encoder reranker）

### 4) 检索增强实验脚本（新增）
- 新增文件：`scripts/eval_retrieval_rerank.py`
- 支持实验矩阵：
  - Retrieval: `dense` / `bm25` / `hybrid`（内部映射 bm25->sparse）
  - Rerank: `none` / `keyword` / `cross_encoder`
  - QueryEnhancer: `identity` / `rewrite` / `expansion`
- 输出：
  - 控制台表格
  - JSON（默认 `output/retrieval_eval_results.json`）
  - CSV（默认 `output/retrieval_eval_results.csv`）
- 指标：
  - `Recall@K`
  - `MRR@K`
  - `nDCG@K`
  - `latency`（avg/p95）

### 5) 自动 Markdown 报告脚本（新增）
- 新增文件：`scripts/generate_retrieval_report.py`
- 输入：`eval_retrieval_rerank.py` 生成的 JSON 结果
- 输出：`output/retrieval_experiment_report.md`
- 自动生成内容包含：
  - 实验目标
  - 系统结构图（mermaid）
  - Query Enhancement / Retrieval / Rerank 方法说明
  - 3D 实验矩阵说明
  - 指标定义
  - 实验结果表格（自动填充）
  - 基于结果的初步结论（自动生成）

## 兼容性说明
- 未删除任何原有模块
- 未破坏原有函数调用方式
- 新增参数均有默认值，保持 backward compatible
- 原系统在不启用新能力时应与历史行为一致

## 快速使用
```bash
# 1) 运行检索增强实验
python scripts/eval_retrieval_rerank.py

# 2) 生成 Markdown 实验报告
python scripts/generate_retrieval_report.py
```

