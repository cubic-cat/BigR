#!/usr/bin/env python3
"""生成小规模LLM语义QA基准测试数据集。

用途
-----
补充评估：生成模拟真实用户查询的问题，与源文本保持低词法重叠，

输出
-----
JSON文件包含QA对，每个包含：
  - question: 转述/语义风格的问题
  - answer: 从chunk中提取的真实答案
  - source_title: Wikipedia文章标题
  - source_chunk_id: Qdrant中的原始chunk ID
  - supporting_context: 支持答案的chunk文本

使用方法
--------
# 默认：从随机Wikipedia chunk生成30个QA对
    python scripts/generate_llm_qa_dataset.py

# 自定义样本数量和随机种子
    python scripts/generate_llm_qa_dataset.py --num_samples 50 --seed 42

# 自定义输出路径
    python scripts/generate_llm_qa_dataset.py --output_path outputs/qa_benchmark/my_qa.json

要求
----
- Qdrant必须运行，且已加载'documents'集合
- .env中必须设置LLM_API_KEY（或QWEN_API_KEY）
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path

# 解析项目根目录，确保import在任何工作目录下都能正常工作
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv
load_dotenv(_ROOT / ".env")

from tqdm import tqdm

# Qdrant客户端用于采样chunk
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

# LLM生成器用于问题生成（复用现有模块）
from core.generator import LLMGenerator


# ---------------------------------------------------------------------------
# 常量定义
# ---------------------------------------------------------------------------

DEFAULT_COLLECTION = "documents"
DEFAULT_NUM_SAMPLES = 30
DEFAULT_SEED = 42
DEFAULT_OUTPUT = "outputs/qa_benchmark/llm_generated_qa.json"

# 生成语义QA对的Prompt模板
# 关键设计：强制转述风格，禁止复制原文
QA_GENERATION_PROMPT = """You are a QA dataset generator for a RAG evaluation benchmark.

Given the following Wikipedia passage, generate exactly ONE question-answer pair.

CRITICAL REQUIREMENTS:
1. The question must be a PARAPHRASE or SEMANTIC QUERY — do NOT copy phrases from the text.
2. The question should sound like something a real user would ask (natural, conversational).
3. The answer must be directly supported by the passage — do NOT add outside knowledge.
4. The answer should be concise (1-3 sentences).
5. Avoid open-ended or subjective questions.
6. Avoid questions that can be answered with just a name or title from the text.

EXAMPLES of good questions (low lexical overlap):
  - Passage: "The Eiffel Tower was constructed from 1887 to 1889 as the centerpiece of the 1889 World's Fair."
    Good Q: "How long did it take to build the famous iron structure in Paris for the World's Fair?"
    Bad Q:  "When was the Eiffel Tower constructed?" (too similar to original text)

  - Passage: "Bendemeer is a subzone within the planning area of Kallang, as defined by the URA."
    Good Q: "Which planning area does this particular subzone fall under according to Singapore's urban authority?"
    Bad Q:  "What planning area is Bendemeer in?" (uses original proper nouns directly)

Now generate a QA pair for this passage:

PASSAGE:
{passage}

Respond in this exact JSON format (no markdown, no extra text):
{{"question": "<your paraphrased question>", "answer": "<concise answer from passage>"}}"""


# ---------------------------------------------------------------------------
# 从Qdrant采样chunk
# ---------------------------------------------------------------------------

def sample_chunks_from_qdrant(
    client: QdrantClient,
    collection: str,
    num_samples: int,
    seed: int,
    min_text_length: int = 100,
) -> list[dict]:
    """从Qdrant集合中随机采样chunk。

    策略：遍历chunk_index=0的chunk（每篇文章的第一个chunk），
    使用给定种子打乱顺序，然后取前num_samples个。
    这确保了文章的多样性。

    返回包含以下键的字典列表：id, text, title, original_id。
    """
    print(f"正在从集合 '{collection}' 采样 {num_samples} 个chunk...")

    # 收集第一个chunk（chunk_index=0）以确保文章多样性
    candidates = []
    offset = None
    batch_size = 200
    seen_titles = set()

    while len(candidates) < num_samples * 5:  # 过采样以保证多样性
        pts, offset = client.scroll(
            collection_name=collection,
            scroll_filter=qm.Filter(must=[
                qm.FieldCondition(
                    key="metadata.chunk_index",
                    match=qm.MatchValue(value=0),
                )
            ]),
            limit=batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        if not pts:
            break

        for p in pts:
            meta = p.payload.get("metadata", {})
            title = meta.get("title", "")
            text = p.payload.get("text", "")
            oid = str(p.payload.get("original_id", p.id))

            # 跳过短文本或重定向文章
            if not title or title in seen_titles:
                continue
            if len(text) < min_text_length:
                continue
            if "REDIRECT" in text[:30]:
                continue

            seen_titles.add(title)
            candidates.append({
                "id": str(p.id),
                "text": text,
                "title": title,
                "original_id": oid,
            })

        if offset is None:
            break

    print(f"  找到 {len(candidates)} 个候选chunk")

    # 打乱并采样
    rng = random.Random(seed)
    rng.shuffle(candidates)
    sampled = candidates[:num_samples]

    print(f"  已采样 {len(sampled)} 个chunk (seed={seed})")
    return sampled


# ---------------------------------------------------------------------------
# 基于LLM的QA生成
# ---------------------------------------------------------------------------

def generate_qa_pair(
    generator: LLMGenerator,
    chunk_text: str,
    max_retries: int = 2,
) -> dict | None:
    """使用LLM从chunk生成单个QA对。

    返回包含'question'和'answer'键的字典，如果失败则返回None。
    """
    # 从chunk文本中去除"# Title "前缀，使prompt更清晰
    body = chunk_text
    title_match = re.match(r"^#\s+.+?\s+", body)
    if title_match:
        body = body[title_match.end():]

    # 截断过长的段落以避免token限制
    passage = body[:800]

    prompt = QA_GENERATION_PROMPT.format(passage=passage)

    for attempt in range(max_retries + 1):
        try:
            result = generator.generate(
                query=prompt,
                context="",  # 不需要RAG上下文；prompt已包含段落
                system_prompt="You are a precise QA dataset generator. Output valid JSON only.",
            )
            raw = result.answer.strip()

            # 尝试从响应中提取JSON
            # 处理LLM用markdown代码块包裹的情况
            json_match = re.search(r'\{[^{}]*"question"[^{}]*"answer"[^{}]*\}', raw, re.DOTALL)
            if json_match:
                qa = json.loads(json_match.group())
                question = qa.get("question", "").strip()
                answer = qa.get("answer", "").strip()
                if question and answer and len(question) > 10:
                    return {"question": question, "answer": answer}

            # 备选方案：尝试将整个响应解析为JSON
            qa = json.loads(raw)
            question = qa.get("question", "").strip()
            answer = qa.get("answer", "").strip()
            if question and answer and len(question) > 10:
                return {"question": question, "answer": answer}

        except (json.JSONDecodeError, KeyError, AttributeError) as e:
            if attempt < max_retries:
                print(f"    [警告] JSON解析失败 (尝试 {attempt+1}): {e}")
                time.sleep(1)
            else:
                print(f"    [警告] {max_retries+1}次尝试后失败: {e}")
        except Exception as e:
            if attempt < max_retries:
                print(f"    [警告] LLM错误 (尝试 {attempt+1}): {e}")
                time.sleep(2)
            else:
                print(f"    [警告] LLM调用失败: {e}")

    return None


def compute_lexical_overlap(query: str, text: str) -> float:
    """计算查询与文本之间的词法重叠率。

    使用简单的token重叠：|query_tokens ∩ text_tokens| / |query_tokens|。
    较低的值表示较少的词法偏差。
    """
    query_tokens = set(re.findall(r'[a-zA-Z]{3,}', query.lower()))
    text_tokens = set(re.findall(r'[a-zA-Z]{3,}', text.lower()))
    if not query_tokens:
        return 0.0
    overlap = len(query_tokens & text_tokens)
    return overlap / len(query_tokens)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def generate_qa_dataset(
    num_samples: int,
    output_path: str,
    seed: int,
    collection: str,
) -> None:
    """完整流程：采样chunk → 生成QA → 保存到文件。"""
    # 连接到Qdrant
    client = QdrantClient(host="localhost", port=6333)

    # 验证集合存在
    existing = {c.name for c in client.get_collections().collections}
    if collection not in existing:
        print(f"[错误] 集合 '{collection}' 不存在。可用集合: {existing}")
        sys.exit(1)

    # 初始化LLM生成器（复用项目的LLMConfig，从.env读取）
    generator = LLMGenerator()
    if not generator.is_configured():
        print("[错误] LLM未配置。请在.env中设置LLM_API_KEY或QWEN_API_KEY")
        sys.exit(1)
    print(f"LLM已配置: provider={generator.config.provider}, model={generator.config.model_name}")

    # 步骤1：采样chunk
    chunks = sample_chunks_from_qdrant(client, collection, num_samples, seed)

    # 步骤2：生成QA对
    qa_pairs = []
    print(f"\n正在生成 {len(chunks)} 个QA对...")

    for chunk in tqdm(chunks, desc="生成QA", unit="pair"):
        qa = generate_qa_pair(generator, chunk["text"])
        if qa is None:
            print(f"  [跳过] 为以下文章生成QA失败: {chunk['title']}")
            continue

        # 计算词法重叠用于质量监控
        overlap = compute_lexical_overlap(qa["question"], chunk["text"])

        qa_pairs.append({
            "question": qa["question"],
            "answer": qa["answer"],
            "source_title": chunk["title"],
            "source_chunk_id": chunk["original_id"],
            "supporting_context": chunk["text"][:600],
            "lexical_overlap": round(overlap, 4),
        })

    # 步骤3：保存到文件
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, "w", encoding="utf-8") as f:
        json.dump(qa_pairs, f, ensure_ascii=False, indent=2)

    # 打印摘要
    print(f"\n{'='*60}")
    print(f"QA数据集生成完成")
    print(f"{'='*60}")
    print(f"  生成总数: {len(qa_pairs)} / {len(chunks)}")
    print(f"  输出文件: {output}")

    if qa_pairs:
        overlaps = [q["lexical_overlap"] for q in qa_pairs]
        avg_overlap = sum(overlaps) / len(overlaps)
        print(f"  平均词法重叠: {avg_overlap:.3f}")
        print(f"  (越低越好 — 表示较少的关键词复制)")
        print(f"\n  示例问题:")
        for i, qa in enumerate(qa_pairs[:3]):
            print(f"    Q{i+1}: {qa['question'][:80]}...")
            print(f"    A: {qa['answer'][:80]}...")
            print(f"    重叠度: {qa['lexical_overlap']:.3f}")
            print()

    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# 命令行接口
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="生成用于RAG评估的LLM语义QA基准测试",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--num_samples", type=int, default=DEFAULT_NUM_SAMPLES,
                   help="要生成的QA对数量")
    p.add_argument("--output_path", default=DEFAULT_OUTPUT,
                   help="输出JSON文件路径")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED,
                   help="chunk采样的随机种子")
    p.add_argument("--collection", default=DEFAULT_COLLECTION,
                   help="Qdrant集合名称")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_qa_dataset(
        num_samples=args.num_samples,
        output_path=args.output_path,
        seed=args.seed,
        collection=args.collection,
    )
