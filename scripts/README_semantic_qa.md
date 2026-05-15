# 语义QA基准测试 — 补充评估流程

## 概述

本补充评估流程用于测试当前启发式基准测试（title / abstract / degradation queries）是否引入了有利于BM25的**词法偏差**。通过生成LLM-based语义QA对（低词法重叠）并运行RAGAS端到端评估来验证。

**定位**：这是一个*补充*评估，**不是**替代现有检索基准测试。

## 新增文件

| 文件                                     | 用途                  |
| -------------------------------------- | ------------------- |
| `scripts/generate_llm_qa_dataset.py`   | 生成LLM-based语义QA基准测试 |
| `scripts/run_ragas_evaluation.py`      | 运行RAGAS端到端评估        |
| `scripts/generate_eval_report.py`      | 自动生成Markdown评估报告    |
| `scripts/requirements_semantic_qa.txt` | 增量依赖                |

## 输出文件

| 文件                                           | 生成器                           | 内容          |
| -------------------------------------------- | ----------------------------- | ----------- |
| `outputs/qa_benchmark/llm_generated_qa.json` | generate\_llm\_qa\_dataset.py | 带词法重叠指标的QA对 |
| `outputs/ragas_eval/results_<method>.json`   | run\_ragas\_evaluation.py     | 每题的RAGAS分数  |
| `outputs/ragas_eval/summary_<method>.json`   | run\_ragas\_evaluation.py     | 每种方法的聚合指标   |
| `outputs/reports/ragas_eval_report.md`       | generate\_eval\_report.py     | 完整分析报告      |

## 快速开始

### 步骤0：安装增量依赖

```bash
pip install -r scripts/requirements_semantic_qa.txt
```

### 步骤1：生成QA基准测试（50个问题）

```bash
python scripts/generate_llm_qa_dataset.py --num_samples 50 --seed 42
```

自定义选项：

```bash
python scripts/generate_llm_qa_dataset.py --num_samples 50 --seed 123 --output_path outputs/qa_benchmark/my_qa.json
```

### 步骤2：运行RAGAS评估

```bash
# 评估BM25
python scripts/run_ragas_evaluation.py --retrieval_method bm25

# 评估Dense检索
python scripts/run_ragas_evaluation.py --retrieval_method dense

# 评估BM25 + CrossEncoder重排
python scripts/run_ragas_evaluation.py --retrieval_method bm25+cross_encoder
```

自定义选项：

```bash
python scripts/run_ragas_evaluation.py --retrieval_method bm25 --top_k 10 --qa_path outputs/qa_benchmark/llm_generated_qa.json --output_dir outputs/ragas_eval
```

### 步骤3：生成评估报告

```bash
# 自动发现 outputs/ragas_eval/ 中的所有摘要
python scripts/generate_eval_report.py

# 或显式指定文件
python scripts/generate_eval_report.py --results outputs/ragas_eval/summary_bm25.json outputs/ragas_eval/summary_dense.json outputs/ragas_eval/summary_bm25_cross_encoder.json
```

## 前提条件

- **Qdrant** 必须运行，且已加载 `documents` 集合
- **LLM API密钥** 必须在 `.env` 中配置（QWEN\_API\_KEY 或 LLM\_API\_KEY）
- LLM用于QA生成和RAGAS评分

## 设计原则

1. **不修改现有代码** — 所有新功能都在新文件中
2. **可插拔** — 每个脚本都可以独立运行
3. **默认小规模** — 30-50个QA对，一个晚上可以完成
4. **可重现** — 种子随机采样，确定性输出路径
5. **进度可见** — 所有循环都有tqdm进度条
6. **结果持久化** — 所有输出都保存到文件

## 关键分析问题

运行完整流程后，报告将回答：

1. **BM25在语义QA下是否仍然占优？**
   - 如果是 → BM25的优势并非纯粹词法偏差
   - 如果否 → 启发式基准测试引入了词法偏差
2. **Dense检索是否相对提升？**
   - 语义查询应该有利于Dense的理解能力
   - 如果Dense赶上或超过BM25，确认存在词法偏差
3. **CrossEncoder重排是否提供边际收益？**
   - 在BM25检索饱和情况下，重排可能效果有限
   - 在语义QA下，重排可能有助于过滤不相关上下文

