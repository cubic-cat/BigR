# BigR

一个正在完善的本地向量库存储版的 RAG 项目。当前代码已经支持：

- 从 `data/processed/` 读取文本文件
- 生成 embedding 并写入本地向量库 `vector_store/`
- 对本地向量库做相似度召回
- 输出召回结果的排名、相似度分数、来源和文本内容

当前最常用的命令行流程只有两步：

1. 先建库
2. 再做召回测试

## 目录约定

```text
configs/         配置文件
core/            embedding / retriever / generator / rag_chain
data/raw/        原始文件
data/processed/  已清洗或已分块的文本，当前默认知识库来源
document/        文档加载模块
scripts/         命令行脚本
vector_store/    本地向量库存储目录
```

## 环境准备

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 初始化环境变量

如果你还没有 `.env`，可以先从模板复制：

```powershell
Copy-Item .env.example .env
```

然后编辑 `.env`，至少保证 embedding 有可用的 key。

### 3. 准备知识库文本

把要入库的文本放进：

```text
data/processed/
```

当前 loader 会默认递归读取下面这些类型：

- `.txt`
- `.md`
- `.markdown`
- `.text`

## RAG 相关命令行指令

### 查看当前配置

```bash
python main.py
```

作用：

- 打印当前 embedding 模型
- 打印当前 LLM 模型
- 打印本地向量库存储路径

### 构建知识库

推荐命令：

```bash
python scripts/build_kb.py
```

执行效果：

- 从 `data/processed/` 读取文本
- 调用 embedding 模型生成向量
- 将结果写入 `vector_store/rag_knowledge_base/store.json`

典型输出包括：

- `Embedded N documents from data/processed`
- `Collection: rag_knowledge_base`
- `Store: vector_store/.../store.json`
- `Dimensions: 1024` 或其他向量维度

### 另一种建库方式

`core/embedding.py` 也带了直接执行入口，效果与建库脚本基本一致：

```bash
python -m core.embedding
```

更推荐使用：

```bash
python scripts/build_kb.py
```

因为脚本语义更明确。

### 测试召回

最常用命令：

```bash
python scripts/test_rag.py -q "RAG 是什么"
```

这个脚本会：

- 检查 `vector_store/` 是否非空
- 对 query 生成 embedding
- 在本地向量库中做相似度召回
- 输出排名、总分、向量分、重排分、来源和召回文本

### 交互式输入 query

如果不传 `-q`，脚本会提示你手动输入：

```bash
python scripts/test_rag.py
```

### 指定召回条数

```bash
python scripts/test_rag.py -q "RAG 是什么" -k 5
```

### 设置最小分数阈值

```bash
python scripts/test_rag.py -q "RAG 是什么" --min-score 0.6
```

### 关闭轻量重排

```bash
python scripts/test_rag.py -q "RAG 是什么" --no-rerank
```

说明：

- 默认会使用“向量相似度 + 简单关键词重排”
- 加上 `--no-rerank` 后，只看向量分

### 输出完整召回文本

```bash
python scripts/test_rag.py -q "RAG 是什么" --full-text
```

### 控制文本预览长度

```bash
python scripts/test_rag.py -q "RAG 是什么" --preview-chars 500
```

## 常见使用顺序

### 首次使用

```bash
pip install -r requirements.txt
python main.py
python scripts/build_kb.py
python scripts/test_rag.py -q "你的问题"
```

### 当 `data/processed/` 内容更新后

重新执行建库即可：

```bash
python scripts/build_kb.py
```

当前建库逻辑会重新生成本地向量库内容。

## 召回结果字段说明

`scripts/test_rag.py` 输出中的主要字段：

- `rank`：当前结果的召回排名
- `id`：文档块 id
- `source`：来源文件名或来源路径
- `score`：最终排序分数
- `vector_score`：纯向量相似度分数
- `rerank_score`：简单关键词重排得分
- `text`：召回文本内容或预览

## 当前配置说明

当前代码是“本地向量库存储”模式，不依赖远程 qdrant 服务：

- 向量库存储目录来自 `VECTOR_DB_PERSIST_DIRECTORY`
- 默认值是 `vector_store`

模型配置按模块分离：

- embedding 读 `EMBEDDING_*`
- llm 读 `LLM_*`
- 也支持按 provider 自动回退到 `QWEN_*`、`OPENAI_*`

例如：

- `EMBEDDING_PROVIDER=qwen`
- `LLM_PROVIDER=openai`

这种“embedding 和 LLM 不是一家”的配置是支持的。

## 常见报错

### 1. 向量库为空

报错含义：

- 还没有执行建库
- 或者 `vector_store/` 目录里没有有效数据

处理方式：

```bash
python scripts/build_kb.py
```

### 2. Embedding API key 缺失

报错含义：

- 没有在 `.env` 中配置可用的 embedding key

处理方式：

- 设置 `EMBEDDING_API_KEY`
- 或设置 provider 级 key，例如 `QWEN_API_KEY`、`OPENAI_API_KEY`

注意：

- 即使只是做“召回测试”，query 也需要先做 embedding
- 所以 `scripts/test_rag.py` 仍然需要 embedding key

## 当前可直接使用的核心命令清单

```bash
python main.py
python scripts/build_kb.py
python -m core.embedding
python scripts/test_rag.py
python scripts/test_rag.py -q "RAG 是什么"
python scripts/test_rag.py -q "RAG 是什么" -k 5
python scripts/test_rag.py -q "RAG 是什么" --min-score 0.6
python scripts/test_rag.py -q "RAG 是什么" --no-rerank
python scripts/test_rag.py -q "RAG 是什么" --full-text
python scripts/test_rag.py -q "RAG 是什么" --preview-chars 500
```
