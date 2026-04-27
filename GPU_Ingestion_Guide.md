# GPU Ingestion Guide — BigR Wikipedia Index Comparison

本文档面向**有本地 GPU**（NVIDIA CUDA 或 Apple Silicon MPS）的机器，指导如何将 Wikipedia JSONL 数据入库到两个 Qdrant collection，并运行索引对比评估。

---

## 目标

在同一份语料上构建两个 Qdrant collection：

| Collection | 索引类型 | 说明 |
|------------|----------|------|
| `wiki_single` | 单向量（dense only） | 每个 chunk 一个 bge-m3 向量，余弦相似度检索 |
| `wiki_dual` | 双向量（dense + sparse） | bge-m3 稠密向量 + TF 稀疏向量，RRF 融合检索 |

评估指标：Recall@5、MRR、Precision@5。

---

## 准备步骤

### Step 0 — 克隆项目

```bash
git clone https://github.com/cubic-cat/BigR.git
cd BigR
git checkout wxl-progress
```

### Step 1 — 安装 Python 依赖

```bash
pip install -r requirements.txt
```

如果用的是 NVIDIA GPU，确认 PyTorch 安装的是 CUDA 版本：

```bash
python -c "import torch; print(torch.cuda.is_available())"
# 期望输出：True
```

如果是 Apple Silicon（M1/M2/M3）：

```bash
python -c "import torch; print(torch.backends.mps.is_available())"
# 期望输出：True
```

如果输出 `False`，需要重装对应版本的 PyTorch，参考 https://pytorch.org/get-started/locally/

### Step 2 — 配置 .env

复制模板并编辑：

```bash
cp .env.example .env
```

入库只需要以下几项，其余保持默认：

```env
VECTOR_DB_PROVIDER=qdrant
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_VECTOR_SIZE=1024
```

RAG 生成和评估不需要，可以留空。

### Step 3 — 下载 bge-m3 模型

```bash
python -c "
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('BAAI/bge-m3')
print('dim:', model.get_embedding_dimension())
"
# 期望输出：dim: 1024
```

如果网络慢，使用镜像：

```bash
HF_ENDPOINT=https://hf-mirror.com python -c "
from sentence_transformers import SentenceTransformer
SentenceTransformer('BAAI/bge-m3')
print('done')
"
```

下载完成后设置离线模式，防止后续运行时触发网络请求：

```bash
# Linux / Mac
export HF_HUB_OFFLINE=1

# Windows PowerShell
$env:HF_HUB_OFFLINE = "1"
```

### Step 4 — 验证 GPU 推理速度

```bash
python -c "
import torch
from sentence_transformers import SentenceTransformer

# 自动选择设备
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'

model = SentenceTransformer('BAAI/bge-m3', device=device)
import time
texts = ['test sentence'] * 64
t0 = time.time()
vecs = model.encode(texts, batch_size=64)
print(f'device={device}  shape={vecs.shape}  time={time.time()-t0:.2f}s')
"
```

参考速度（64 条）：CUDA ~0.2s，MPS ~0.5s，CPU ~5s。

### Step 5 — 安装并启动 Qdrant

从 https://github.com/qdrant/qdrant/releases 下载对应平台的二进制文件，然后：

```bash
# 必须在二进制文件所在目录启动
cd /path/to/qdrant
./qdrant          # Linux / Mac
.\qdrant.exe      # Windows
```

验证：访问 http://localhost:6333/dashboard，能看到空的 collection 列表即可。

---

## 入库

### Step 6 — Smoke test（验证流水线正常）

先用 100 篇验证整个流程跑通，再全量入库：

```bash
# Linux / Mac
HF_HUB_OFFLINE=1 python scripts/ingest_wikipedia.py \
  --file /path/to/enwiki_namespace_0_0.jsonl \
  --collection smoke_test \
  --embed-batch-size 64 \
  --article-batch-size 200 \
  --max-articles 100

# Windows PowerShell
$env:HF_HUB_OFFLINE = "1"
python scripts/ingest_wikipedia.py `
  --file "C:/path/to/enwiki_namespace_0_0.jsonl" `
  --collection smoke_test `
  --embed-batch-size 64 `
  --article-batch-size 200 `
  --max-articles 100
```

期望输出：
```
[LocalEmbeddingClient] Loading BAAI/bge-m3 on cuda...   ← 必须是 cuda 或 mps，不能是 cpu
[ingest] Articles processed : ~98
[ingest] Chunks ingested    : ~250
[ingest] Qdrant points      : ~250
```

**如果看到 `device=cpu`，停下来检查 Step 1 的 PyTorch 安装。**

### Step 7 — 全量入库 wiki_single（dense only）

```bash
# Linux / Mac
HF_HUB_OFFLINE=1 python scripts/ingest_wikipedia.py \
  --file /path/to/enwiki_namespace_0_0.jsonl \
  --collection wiki_single \
  --embed-batch-size 64 \
  --article-batch-size 200

# Windows PowerShell
$env:HF_HUB_OFFLINE = "1"
python scripts/ingest_wikipedia.py `
  --file "C:/path/to/enwiki_namespace_0_0.jsonl" `
  --collection wiki_single `
  --embed-batch-size 64 `
  --article-batch-size 200
```

预计时间：CUDA ~30 min，MPS ~1 h，CPU ~12 h（不推荐）。

完成后验证：

```bash
python -c "
from qdrant_client import QdrantClient
c = QdrantClient(host='localhost', port=6333)
info = c.get_collection('wiki_single')
print('wiki_single points:', info.points_count)
"
# 期望：~600,000
```

**如果中途中断**，用 `--skip N` 断点续传（N = 已处理的文章数）：

```bash
HF_HUB_OFFLINE=1 python scripts/ingest_wikipedia.py \
  --file /path/to/enwiki_namespace_0_0.jsonl \
  --collection wiki_single \
  --embed-batch-size 64 \
  --article-batch-size 200 \
  --skip 50000
```

### Step 8 — 全量入库 wiki_dual（dense + sparse）

```bash
# Linux / Mac
HF_HUB_OFFLINE=1 python scripts/ingest_wikipedia_dual.py \
  --file /path/to/enwiki_namespace_0_0.jsonl

# Windows PowerShell
$env:HF_HUB_OFFLINE = "1"
python scripts/ingest_wikipedia_dual.py `
  --file "C:/path/to/enwiki_namespace_0_0.jsonl"
```

预计时间与 wiki_single 相近（sparse vector 计算开销很小）。

完成后验证：

```bash
python -c "
from qdrant_client import QdrantClient
c = QdrantClient(host='localhost', port=6333)
for name in ['wiki_single', 'wiki_dual']:
    info = c.get_collection(name)
    print(f'{name}: {info.points_count:,} points')
"
# 期望两个都是 ~600,000
```

---

## 评估

### Step 9 — 运行索引对比评估

两个 collection 都入库完成后运行：

```bash
# Linux / Mac
HF_HUB_OFFLINE=1 python scripts/eval_index_comparison.py \
  --file /path/to/enwiki_namespace_0_0.jsonl \
  --max-articles 500 \
  --top-k 5 \
  --output results/index_comparison.json

# Windows PowerShell
$env:HF_HUB_OFFLINE = "1"
python scripts/eval_index_comparison.py `
  --file "C:/path/to/enwiki_namespace_0_0.jsonl" `
  --max-articles 500 `
  --top-k 5 `
  --output results/index_comparison.json
```

`--max-articles 500` 表示用 500 篇文章的 abstract 作为 query 构建测试集，够用于可靠对比。

期望输出格式：

```
=======================================================
Metric               wiki_single       wiki_dual
-------------------------------------------------------
Recall@5                  0.xxxx          0.xxxx ◄
MRR                       0.xxxx          0.xxxx
Precision@5               0.xxxx          0.xxxx
=======================================================
```

结果同时保存到 `results/index_comparison.json`。

---

## 打包传回 Windows（可选）

如果在朋友机器上跑完，需要把 Qdrant storage 传回 Windows 评估：

```bash
# 先停止 Qdrant（Ctrl+C）
# 然后在 Qdrant 二进制所在目录执行
zip -r qdrant_storage_wiki.zip storage/
```

传回 Windows 后：
1. 解压到 `C:\path\to\qdrant\storage\`（替换原有 storage）
2. 启动 Qdrant
3. 访问 http://localhost:6333/dashboard 确认 `wiki_single` 和 `wiki_dual` 都出现
4. 运行 Step 9 的评估命令

---

## 参数速查

| 参数 | 推荐值（GPU） | 说明 |
|------|-------------|------|
| `--embed-batch-size` | 64 | 每次 forward pass 的 chunk 数，GPU 可用 128 |
| `--article-batch-size` | 200 | 每次 flush 前累积的文章数 |
| `--max-articles` | 不设（全量） | 调试时可设 100 |

bge-m3 在不同设备上的参考吞吐量：

| 设备 | 速度 | 300K 篇预计时间 |
|------|------|----------------|
| NVIDIA GPU（RTX 3090 等） | ~100 art/s | ~50 min |
| Apple M3 MPS | ~30–50 art/s | ~2–3 h |
| CPU（8 核） | ~1–7 art/s | ~12–80 h |
