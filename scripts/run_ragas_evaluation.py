#!/usr/bin/env python3
"""在语义QA基准测试上运行RAGAS端到端评估。

用途
-----
补充评估：测试BM25在启发式基准测试中的优势是否在语义QA条件下持续存在，
以及Dense检索或CrossEncoder重排是否提供边际收益。

支持的检索方法
-------------
  - bm25:              客户端BM25（与eval_retrieval_comparison.py相同）
  - dense:             Qdrant HNSW密集搜索
  - bm25+cross_encoder: BM25检索 + CrossEncoder重排

本脚本复用现有的QdrantRetriever、LLMGenerator和CrossEncoderReranker模块，
不引入新的检索框架。

使用方法
--------
# 默认：在LLM生成的QA基准测试上评估BM25
    python scripts/run_ragas_evaluation.py --retrieval_method bm25

# 评估Dense检索
    python scripts/run_ragas_evaluation.py --retrieval_method dense

# 评估BM25 + CrossEncoder重排
    python scripts/run_ragas_evaluation.py --retrieval_method bm25+cross_encoder

# 自定义QA路径和top_k
    python scripts/run_ragas_evaluation.py --retrieval_method bm25 --top_k 10 --qa_path outputs/qa_benchmark/llm_generated_qa.json

要求
----
- Qdrant必须运行，且已加载'documents'集合
- .env中必须设置LLM_API_KEY（或QWEN_API_KEY）（用于生成和RAGAS）
- 需要安装ragas包: pip install ragas
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv
load_dotenv(_ROOT / ".env")

from tqdm import tqdm

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from core.generator import LLMGenerator
from core.qdrant_retriever import QdrantRetriever
from core.embedding import build_embedding_client
from configs.vector_db_config import VectorDBConfig


# ---------------------------------------------------------------------------
# 常量定义
# ---------------------------------------------------------------------------

DEFAULT_COLLECTION = "documents"
DEFAULT_TOP_K = 10
DEFAULT_QA_PATH = "outputs/qa_benchmark/llm_generated_qa.json"
DEFAULT_OUTPUT_DIR = "outputs/ragas_eval"

VALID_METHODS = ["bm25", "dense", "bm25+cross_encoder"]


# ---------------------------------------------------------------------------
# BM25辅助函数（复用自eval_retrieval_comparison.py）
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    """简单的空格+小写tokenizer。"""
    return text.lower().split()


def _bm25_scores(
    query_tokens: list[str],
    corpus: list[dict],
    k1: float = 1.5,
    b: float = 0.75,
) -> list[tuple[str, float, str]]:
    """计算BM25分数。返回(id, score, title)按降序排列。"""
    import math
    n_docs = len(corpus)
    if n_docs == 0:
        return []
    avg_dl = sum(d["doc_len"] for d in corpus) / n_docs
    df: dict[str, int] = {}
    for tok in set(query_tokens):
        df[tok] = sum(1 for d in corpus if tok in d["token_set"])
    scores = []
    for doc in corpus:
        score = 0.0
        dl = doc["doc_len"]
        tf_map = doc["tf"]
        for tok in query_tokens:
            if tok not in tf_map:
                continue
            tf = tf_map[tok]
            n_q = df.get(tok, 0)
            idf = math.log(1 + max(0.0, (n_docs - n_q + 0.5) / (n_q + 0.5)))
            score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avg_dl))
        scores.append((doc["id"], score, doc["title"]))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


# ---------------------------------------------------------------------------
# 检索实现
# ---------------------------------------------------------------------------

def build_bm25_pool(
    client: QdrantClient,
    collection: str,
    max_pool: int = 50000,
) -> list[dict]:
    """从Qdrant集合构建BM25候选池。"""
    print(f"正在构建BM25候选池 (max={max_pool})...")
    pool = []
    offset = None
    while len(pool) < max_pool:
        pts, offset = client.scroll(
            collection_name=collection,
            limit=200,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        if not pts:
            break
        for p in pts:
            meta = p.payload.get("metadata", {})
            text = p.payload.get("text", "")
            oid = str(p.payload.get("original_id", p.id))
            tokens = _tokenize(text)
            pool.append({
                "id": oid,
                "text": text,
                "title": meta.get("title", ""),
                "tokens": tokens,
                "token_set": set(tokens),
                "tf": Counter(tokens),
                "doc_len": len(tokens),
            })
        if offset is None:
            break
    print(f"  BM25候选池: {len(pool)} 文档")
    return pool


def retrieve_bm25(
    query: str,
    bm25_pool: list[dict],
    top_k: int,
) -> list[dict]:
    """BM25检索，返回{id, text, title, score}列表。"""
    query_tokens = _tokenize(query)
    if not query_tokens:
        return []
    scored = _bm25_scores(query_tokens, bm25_pool)
    results = []
    for doc_id, score, title in scored[:top_k]:
        # 从候选池中找到完整文本
        text = ""
        for doc in bm25_pool:
            if doc["id"] == doc_id:
                text = doc["text"]
                break
        results.append({
            "id": doc_id,
            "text": text,
            "title": title,
            "score": score,
        })
    return results


def retrieve_dense(
    retriever: QdrantRetriever,
    query: str,
    top_k: int,
) -> list[dict]:
    """通过Qdrant HNSW进行Dense检索，返回{id, text, title, score}列表。"""
    hits = retriever.similarity_search(
        query, top_k=top_k, rerank=False, retrieval_method="dense",
    )
    results = []
    for h in hits:
        results.append({
            "id": h.id,
            "text": h.text,
            "title": h.metadata.get("title", ""),
            "score": h.score,
        })
    return results


def rerank_cross_encoder(
    query: str,
    results: list[dict],
    top_k: int,
) -> list[dict]:
    """对检索结果应用CrossEncoder重排。"""
    from core.search_types import SearchResult
    from core.reranker import resolve_reranker

    # 转换为SearchResult对象供重排器使用
    search_results = []
    for r in results:
        search_results.append(SearchResult(
            id=r["id"],
            text=r["text"],
            metadata={"title": r["title"]},
            score=r["score"],
            retrieval_score=r["score"],
            vector_score=0.0,
            retrieval_method="bm25",
        ))

    reranker = resolve_reranker("cross_encoder")
    reranked = reranker.rerank(
        query=query,
        results=search_results,
        top_k=top_k,
    )

    output = []
    for r in reranked:
        output.append({
            "id": r.id,
            "text": r.text,
            "title": r.metadata.get("title", ""),
            "score": r.score,
            "rerank_score": r.rerank_score,
        })
    return output


# ---------------------------------------------------------------------------
# RAGAS评分（复用自eval_strategies.py）
# ---------------------------------------------------------------------------

def _init_ragas_metrics(base_url: str, api_key: str) -> dict:
    """使用项目的LLM配置初始化RAGAS指标。"""
    from openai import AsyncOpenAI
    from ragas.llms import llm_factory
    from ragas.embeddings import OpenAIEmbeddings as RagasEmbeddings
    from ragas.metrics.collections import (
        ContextPrecisionWithoutReference,
        ContextRecall,
        Faithfulness,
        AnswerRelevancy,
    )

    async_client = AsyncOpenAI(base_url=base_url, api_key=api_key)
    llm = llm_factory("qwen-plus", client=async_client, max_tokens=4096)
    emb = RagasEmbeddings(client=async_client, model="text-embedding-v4")

    return {
        "context_precision": ContextPrecisionWithoutReference(llm=llm),
        "context_recall": ContextRecall(llm=llm),
        "faithfulness": Faithfulness(llm=llm),
        "answer_relevancy": AnswerRelevancy(llm=llm, embeddings=emb),
    }


def _to_float(result) -> float:
    """从MetricResult或普通数值中提取float。"""
    if hasattr(result, "result"):
        return float(result.result)
    return float(result)


async def _score_one_async(
    metrics: dict,
    question: str,
    reference: str,
    retrieved_contexts: list[str],
    response: str,
) -> dict[str, float]:
    """单个QA样本的异步RAGAS评分。"""
    scores: dict[str, float] = {}

    cp = await metrics["context_precision"].ascore(
        user_input=question, response=response, retrieved_contexts=retrieved_contexts,
    )
    scores["context_precision"] = _to_float(cp)

    cr = await metrics["context_recall"].ascore(
        user_input=question, retrieved_contexts=retrieved_contexts, reference=reference,
    )
    scores["context_recall"] = _to_float(cr)

    fa = await metrics["faithfulness"].ascore(
        user_input=question, response=response, retrieved_contexts=retrieved_contexts,
    )
    scores["faithfulness"] = _to_float(fa)

    ar = await metrics["answer_relevancy"].ascore(
        user_input=question, response=response,
    )
    scores["answer_relevancy"] = _to_float(ar)

    return scores


def score_one(
    metrics: dict,
    question: str,
    reference: str,
    retrieved_contexts: list[str],
    response: str,
) -> dict[str, float]:
    """对单个QA样本进行评分（内部使用异步运行）。"""
    return asyncio.run(
        _score_one_async(metrics, question, reference, retrieved_contexts, response)
    )


# ---------------------------------------------------------------------------
# 主评估流程
# ---------------------------------------------------------------------------

def run_evaluation(args: argparse.Namespace) -> None:
    """完整流程：加载QA → 检索 → 生成 → RAGAS评分 → 保存。"""
    method = args.retrieval_method.strip().lower()
    if method not in VALID_METHODS:
        print(f"[错误] 无效方法: {method}。有效方法: {VALID_METHODS}")
        sys.exit(1)

    # 加载QA基准测试
    qa_path = Path(args.qa_path)
    if not qa_path.exists():
        print(f"[错误] QA基准测试未找到: {qa_path}")
        print("  请先运行generate_llm_qa_dataset.py。")
        sys.exit(1)

    with open(qa_path, "r", encoding="utf-8") as f:
        qa_pairs = json.load(f)

    print(f"从 {qa_path} 加载了 {len(qa_pairs)} 个QA对")

    # 连接到Qdrant
    client = QdrantClient(host="localhost", port=6333)
    collection = args.collection

    existing = {c.name for c in client.get_collections().collections}
    if collection not in existing:
        print(f"[错误] 集合 '{collection}' 不存在。")
        sys.exit(1)

    # 初始化组件
    generator = LLMGenerator()
    if not generator.is_configured():
        print("[错误] LLM未配置。请在.env中设置LLM_API_KEY")
        sys.exit(1)

    # 构建BM25候选池（如果需要）
    bm25_pool = None
    if "bm25" in method:
        bm25_pool = build_bm25_pool(client, collection, max_pool=50000)

    # 构建Dense检索器（如果需要）
    dense_retriever = None
    if "dense" in method:
        embedding_client = build_embedding_client(model="BAAI/bge-small-en-v1.5")
        config = VectorDBConfig(
            provider="qdrant",
            collection_name=collection,
            distance_metric="cosine",
            persist_directory="vector_store",
        )
        dense_retriever = QdrantRetriever(
            embedding_client=embedding_client,
            config=config,
            retrieval_method="dense",
        )

    # 初始化RAGAS指标
    print("\n正在初始化RAGAS指标...")
    ragas_metrics = _init_ragas_metrics(
        base_url=os.getenv("QWEN_BASE_URL"),
        api_key=os.getenv("QWEN_API_KEY"),
    )
    print("RAGAS指标准备就绪:", list(ragas_metrics.keys()))

    # 评估每个QA对
    top_k = args.top_k
    results = []

    print(f"\n正在评估 method='{method}'，top_k={top_k}...")
    print(f"{'='*60}")

    for i, qa in enumerate(tqdm(qa_pairs, desc=f"评估 ({method})", unit="q")):
        question = qa["question"]
        ground_truth = qa["answer"]

        # --- 检索 ---
        t_retrieve_start = time.perf_counter()

        if method == "bm25":
            raw_results = retrieve_bm25(question, bm25_pool, top_k)
        elif method == "dense":
            raw_results = retrieve_dense(dense_retriever, question, top_k)
        elif method == "bm25+cross_encoder":
            # BM25检索2*top_k，然后CrossEncoder重排到top_k
            raw_results = retrieve_bm25(question, bm25_pool, top_k * 2)
            raw_results = rerank_cross_encoder(question, raw_results, top_k)
        else:
            raw_results = []

        t_retrieve = (time.perf_counter() - t_retrieve_start) * 1000

        # 提取上下文文本
        contexts = [r["text"] for r in raw_results]

        if not contexts:
            results.append({
                "question": question,
                "ground_truth": ground_truth,
                "error": "no_retrieval",
            })
            continue

        # --- 生成 ---
        t_gen_start = time.perf_counter()
        try:
            context_text = "\n\n".join(f"[{j+1}] {c}" for j, c in enumerate(contexts))
            gen = generator.generate(question, context_text)
            response = gen.answer
        except Exception as e:
            print(f"  [警告] Q{i+1}生成错误: {e}")
            response = contexts[0][:200]

        t_gen = (time.perf_counter() - t_gen_start) * 1000

        # --- RAGAS评分 ---
        try:
            scores = score_one(ragas_metrics, question, ground_truth, contexts, response)
        except Exception as e:
            print(f"  [警告] Q{i+1} RAGAS错误: {e}")
            scores = {}

        results.append({
            "question": question,
            "ground_truth": ground_truth,
            "response": response,
            "retrieved_contexts": contexts[:3],  # 为简洁保存前3个
            "retrieval_latency_ms": round(t_retrieve, 1),
            "generation_latency_ms": round(t_gen, 1),
            "scores": scores,
            "source_title": qa.get("source_title", ""),
        })

    # --- 保存结果 ---
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存详细结果
    results_path = output_dir / f"results_{method.replace('+', '_')}.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n详细结果已保存: {results_path}")

    # 计算并保存摘要
    metric_names = ["context_precision", "context_recall", "faithfulness", "answer_relevancy"]
    summary = {
        "method": method,
        "top_k": top_k,
        "n_questions": len(qa_pairs),
        "n_successful": len([r for r in results if "error" not in r]),
        "qa_path": str(qa_path),
        "collection": collection,
        "metrics": {},
    }

    for m in metric_names:
        vals = [r["scores"][m] for r in results if "scores" in r and m in r["scores"]]
        if vals:
            summary["metrics"][m] = {
                "mean": round(sum(vals) / len(vals), 4),
                "min": round(min(vals), 4),
                "max": round(max(vals), 4),
            }

    # 延迟摘要
    retrieve_latencies = [r["retrieval_latency_ms"] for r in results if "retrieval_latency_ms" in r]
    gen_latencies = [r["generation_latency_ms"] for r in results if "generation_latency_ms" in r]
    if retrieve_latencies:
        summary["avg_retrieval_latency_ms"] = round(sum(retrieve_latencies) / len(retrieve_latencies), 1)
    if gen_latencies:
        summary["avg_generation_latency_ms"] = round(sum(gen_latencies) / len(gen_latencies), 1)

    # 保存摘要
    summary_path = output_dir / f"summary_{method.replace('+', '_')}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"摘要已保存: {summary_path}")

    # 打印摘要表格
    print(f"\n{'='*60}")
    print(f"RAGAS评估摘要 — {method}")
    print(f"{'='*60}")
    print(f"  问题数量: {summary['n_successful']}/{summary['n_questions']}")
    for m in metric_names:
        if m in summary["metrics"]:
            print(f"  {m:<25}: {summary['metrics'][m]['mean']:.4f}")
    if "avg_retrieval_latency_ms" in summary:
        print(f"  {'平均检索延迟':<25}: {summary['avg_retrieval_latency_ms']:.1f} ms")
    if "avg_generation_latency_ms" in summary:
        print(f"  {'平均生成延迟':<25}: {summary['avg_generation_latency_ms']:.1f} ms")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# 命令行接口
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="在语义QA基准测试上运行RAGAS端到端评估",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--retrieval_method", default="bm25",
                   choices=VALID_METHODS,
                   help="要评估的检索方法")
    p.add_argument("--top_k", type=int, default=DEFAULT_TOP_K,
                   help="要检索的chunk数量")
    p.add_argument("--qa_path", default=DEFAULT_QA_PATH,
                   help="LLM生成的QA基准测试JSON路径")
    p.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR,
                   help="保存评估结果的目录")
    p.add_argument("--collection", default=DEFAULT_COLLECTION,
                   help="Qdrant集合名称")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_evaluation(args)
