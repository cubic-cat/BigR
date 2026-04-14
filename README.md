# BigR

一个使用本地 `vector_store/` 作为向量库的轻量 RAG 项目。当前重点能力：

- 从 `data/processed/` 读取文本并构建知识库
- 使用可切换的召回方式做第一阶段检索
- 使用可切换的重排方式做第二阶段排序
- 支持仅召回测试
- 支持完整 RAG 全链路测试：召回 + 重排 + LLM 生成

当前已实现：

- 召回方式：`dense`
- 重排方式：`keyword`

## 目录说明

```text
configs/         配置文件
core/            embedding / retriever / reranker / generator / rag_chain
data/raw/        原始文件
data/processed/  当前默认知识库来源
document/        文档加载模块
scripts/         命令行脚本
vector_store/    本地向量库存储目录
```

## 环境准备

### 安装依赖

```bash
pip install -r requirements.txt
```

### 初始化环境变量

如果还没有 `.env`，先复制模板：

```powershell
Copy-Item .env.example .env
```

至少要保证：

- 建库和召回时，embedding key 可用
- 跑全链路时，embedding key 和 LLM key 都可用

### 准备知识库文本

把待入库文本放到：

```text
data/processed/
```

当前默认读取的文件类型：

- `.txt`
- `.md`
- `.markdown`
- `.text`

## 常用命令

### 查看当前配置

```bash
python main.py
```

### 构建知识库

推荐命令：

```bash
python scripts/build_kb.py
```

作用：

- 读取 `data/processed/`
- 调用 embedding 模型生成向量
- 写入 `vector_store/rag_knowledge_base/store.json`

等价入口：

```bash
python -m core.embedding
```

## test_rag.py 的执行方式

[`scripts/test_rag.py`](scripts/test_rag.py) 现在支持两种模式：

- 仅召回测试
- 全链路测试

### 1. 列出可用召回方式和重排方式

```bash
python scripts/test_rag.py --list-methods
```

当前输出应至少包含：

```text
dense
keyword
```

### 2. 仅召回测试

这是最常用的方式，只做检索和重排，不调用 LLM。

#### 直接检索

```bash
python scripts/test_rag.py -q "RAG 是什么"
```

输出内容包括：

- query
- collection
- store_path
- document_count
- vector_dimensions
- retrieval_method
- rerank_method
- rank
- score
- retrieval_score
- vector_score
- rerank_score
- text

#### 交互式输入 query

```bash
python scripts/test_rag.py
```

#### 指定返回条数

```bash
python scripts/test_rag.py -q "RAG 是什么" -k 5
```

#### 设置最小分数阈值

```bash
python scripts/test_rag.py -q "RAG 是什么" --min-score 0.6
```

#### 输出完整召回文本

```bash
python scripts/test_rag.py -q "RAG 是什么" --full-text
```

#### 控制文本预览长度

```bash
python scripts/test_rag.py -q "RAG 是什么" --preview-chars 500
```

#### 关闭重排

```bash
python scripts/test_rag.py -q "RAG 是什么" --no-rerank
```

#### 指定重排前候选数

```bash
python scripts/test_rag.py -q "RAG 是什么" --candidate-top-k 10
```

说明：

- 检索阶段先取 `candidate_top_k` 个候选
- 然后重排器对候选重新排序
- 最终返回 `top_k` 个结果

### 3. 更换召回方式

有两种方式。

#### 方法 A：命令行临时切换

```bash
python scripts/test_rag.py -q "RAG 是什么" --retrieval-method dense
```

#### 方法 B：环境变量切换默认召回方式

在 `.env` 中设置：

```env
RETRIEVAL_METHOD=dense
```

然后直接运行：

```bash
python scripts/test_rag.py -q "RAG 是什么"
```

### 4. 更换重排方式

同样有两种方式。

#### 方法 A：命令行临时切换

```bash
python scripts/test_rag.py -q "RAG 是什么" --rerank-method keyword
```

#### 方法 B：环境变量切换默认重排方式

在 `.env` 中设置：

```env
RERANK_METHOD=keyword
```

如果要默认关闭重排：

```env
RERANK_ENABLED=false
```

### 5. 检索前强制重建知识库

如果你修改了 `data/processed/` 里的文本，想在检索前强制重建向量库，可以加：

```bash
python scripts/test_rag.py -q "RAG 是什么" --refresh-from-processed
```

## 运行全链路

全链路 = 召回 + 重排 + 组装上下文 + 调用 LLM 生成答案。

### 最基本的全链路命令

```bash
python scripts/test_rag.py -q "RAG 是什么" --full-chain
```

### 指定召回方式和重排方式跑全链路

```bash
python scripts/test_rag.py -q "RAG 是什么" --full-chain --retrieval-method dense --rerank-method keyword
```

### 全链路时显示上下文

```bash
python scripts/test_rag.py -q "RAG 是什么" --full-chain --show-context
```

### 全链路时控制上下文长度

```bash
python scripts/test_rag.py -q "RAG 是什么" --full-chain --max-context-chars 6000
```

### 全链路时同时刷新知识库

```bash
python scripts/test_rag.py -q "RAG 是什么" --full-chain --refresh-from-processed
```

## 推荐使用顺序

### 方式一：分步执行

```bash
python main.py
python scripts/build_kb.py
python scripts/test_rag.py -q "你的问题" --retrieval-method dense --rerank-method keyword
python scripts/test_rag.py -q "你的问题" --full-chain --retrieval-method dense --rerank-method keyword
```

### 方式二：直接检索前刷新

```bash
python scripts/test_rag.py -q "你的问题" --refresh-from-processed
```

### 方式三：直接跑全链路并刷新

```bash
python scripts/test_rag.py -q "你的问题" --full-chain --refresh-from-processed
```

## .env 中和召回/重排相关的字段

当前模板在 [`.env.example`](.env.example) 中，相关字段包括：

```env
RETRIEVAL_METHOD=dense
RERANK_ENABLED=true
RERANK_METHOD=keyword
RERANK_CANDIDATE_TOP_K=10
KEYWORD_RERANK_RETRIEVAL_WEIGHT=0.85
KEYWORD_RERANK_KEYWORD_WEIGHT=0.15
```

含义：

- `RETRIEVAL_METHOD`：默认召回方式
- `RERANK_ENABLED`：默认是否启用重排
- `RERANK_METHOD`：默认重排方式
- `RERANK_CANDIDATE_TOP_K`：重排前先取多少个候选
- `KEYWORD_RERANK_RETRIEVAL_WEIGHT`：keyword 重排中原始召回分权重
- `KEYWORD_RERANK_KEYWORD_WEIGHT`：keyword 重排中关键词匹配分权重

## 常见报错

### 向量库为空

说明：

- 还没有建库
- 或 `vector_store/` 中没有有效数据

处理方式：

```bash
python scripts/build_kb.py
```

或者：

```bash
python scripts/test_rag.py -q "RAG 是什么" --refresh-from-processed
```

### 缺少 embedding key

说明：

- 建库和召回都需要 embedding key

### 缺少 LLM key

说明：

- 只有在 `--full-chain` 模式下才需要

## 当前最常用命令清单

```bash
python main.py
python scripts/build_kb.py
python -m core.embedding
python scripts/test_rag.py --list-methods
python scripts/test_rag.py
python scripts/test_rag.py -q "RAG 是什么"
python scripts/test_rag.py -q "RAG 是什么" -k 5
python scripts/test_rag.py -q "RAG 是什么" --retrieval-method dense --rerank-method keyword
python scripts/test_rag.py -q "RAG 是什么" --min-score 0.6
python scripts/test_rag.py -q "RAG 是什么" --no-rerank
python scripts/test_rag.py -q "RAG 是什么" --candidate-top-k 10
python scripts/test_rag.py -q "RAG 是什么" --full-text
python scripts/test_rag.py -q "RAG 是什么" --refresh-from-processed
python scripts/test_rag.py -q "RAG 是什么" --full-chain
python scripts/test_rag.py -q "RAG 是什么" --full-chain --show-context
python scripts/test_rag.py -q "RAG 是什么" --full-chain --retrieval-method dense --rerank-method keyword
```
