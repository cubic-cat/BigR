#!/usr/bin/env python3
"""Chunking strategy comparison via RAGAS metrics.

Builds three Qdrant collections (one per strategy) from the same N articles,
runs the same question set against each, scores with RAGAS, and prints a
side-by-side comparison table.

Usage
-----
# Quick smoke test (100 articles, built-in questions):
    python scripts/eval_strategies.py --max-articles 100

# Use your own question file (JSONL, one {"question":"...","reference":"..."} per line):
    python scripts/eval_strategies.py --max-articles 5000 --questions data/eval_questions.jsonl

# Only rebuild collections (skip if already built):
    python scripts/eval_strategies.py --skip-ingest

Strategies compared
-------------------
  section     : section-first + sliding-window for oversized sections (current default)
  fixed512    : fixed 512-token sliding window, 50-token overlap
  fixed256    : fixed 256-token sliding window, 50-token overlap
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv
load_dotenv(_ROOT / ".env")

from document.wiki_loader import iter_articles
from document.splitter import chunk_article, split_text, TextChunk
from core.embedding import EmbeddingClient, build_embedding_client
from core.qdrant_retriever import QdrantRetriever, _doc_id_to_qdrant
from core.generator import LLMGenerator
from configs.vector_db_config import VectorDBConfig


# ---------------------------------------------------------------------------
# Built-in evaluation questions (used when no --questions file is provided)
# These are designed to have clear answers in Wikipedia articles.
# ---------------------------------------------------------------------------
BUILTIN_QUESTIONS = [
    {
        "question": "What was the lowest air pressure recorded during the 1906 Mississippi hurricane?",
        "reference": "The lowest air pressure recorded in Mobile was 977 mbar during the 1906 Mississippi hurricane.",
    },
    {
        "question": "When was asteroid 1214 Richilde discovered and by whom?",
        "reference": "Richilde was discovered on 1 January 1932 by German astronomer Max Wolf at the Heidelberg-Königstuhl State Observatory.",
    },
    {
        "question": "What is the #NotAgainSU movement about?",
        "reference": "NotAgainSU is a hashtag and student-led organization that began after racist incidents at Syracuse University between 2019 and 2021.",
    },
    {
        "question": "What type of asteroid is 1214 Richilde classified as?",
        "reference": "In the SMASS classification, Richilde is an Xk-subtype asteroid that transitions from X-type to the rare K-type.",
    },
    {
        "question": "What storm surge height was recorded in New Orleans during the 1906 hurricane?",
        "reference": "A storm surge of about 6 feet (1.8 m) was recorded at the backwater of the Mississippi River in New Orleans.",
    },
    {
        "question": "How many demands did the NotAgainSU protesters make to Syracuse University?",
        "reference": "The protesters initially made 19 demands to Chancellor Kent Syverud, which was later expanded to 34.",
    },
    {
        "question": "What song by Tame Impala was released in April 2015?",
        "reference": "'Cause I'm a Man' is a song by Tame Impala released on 7 April 2015 as the second single from Currents.",
    },
    {
        "question": "What is the orbital distance range of asteroid 1214 Richilde from the Sun?",
        "reference": "Richilde orbits the Sun in the central asteroid belt at a distance of 2.4 to 3.0 AU.",
    },
]


# ---------------------------------------------------------------------------
# Chunking strategies
# ---------------------------------------------------------------------------

def chunks_section(article, **kwargs) -> list[TextChunk]:
    """Section-first + sliding window for oversized sections."""
    return chunk_article(article, max_tokens=512, overlap_tokens=50)


def chunks_fixed512(article, **kwargs) -> list[TextChunk]:
    """Fixed 512-token sliding window over full article text."""
    full_text = article.abstract + "\n\n"
    for s in article.sections:
        full_text += f"{s.title}\n{s.text}\n\n"
    return split_text(full_text.strip(), max_tokens=512, overlap_tokens=50,
                      source=article.title)


def chunks_fixed256(article, **kwargs) -> list[TextChunk]:
    """Fixed 256-token sliding window over full article text."""
    full_text = article.abstract + "\n\n"
    for s in article.sections:
        full_text += f"{s.title}\n{s.text}\n\n"
    return split_text(full_text.strip(), max_tokens=256, overlap_tokens=30,
                      source=article.title)


STRATEGIES = {
    "section":   chunks_section,
    "fixed512":  chunks_fixed512,
    "fixed256":  chunks_fixed256,
}


# ---------------------------------------------------------------------------
# Ingest helpers
# ---------------------------------------------------------------------------

def _make_retriever(collection_name: str, embedding_client) -> QdrantRetriever:
    config = VectorDBConfig(
        provider="qdrant",
        collection_name=collection_name,
        distance_metric="cosine",
        persist_directory="vector_store",
    )
    return QdrantRetriever(embedding_client=embedding_client, config=config)


def ingest_strategy(
    strategy_name: str,
    chunk_fn,
    jsonl_path: Path,
    max_articles: int,
    embedding_client,
    collection_name: str | None = None,
    embed_batch_size: int | None = None,
    upsert_batch_size: int = 64,
) -> QdrantRetriever:
    """Build a Qdrant collection for one chunking strategy."""
    collection = collection_name or f"eval_{strategy_name}"
    retriever = _make_retriever(collection, embedding_client)
    # Default batch size: 10 for API clients, 32 for local
    if embed_batch_size is None:
        from core.embedding import LocalEmbeddingClient as _LEC
        embed_batch_size = 32 if isinstance(embedding_client, _LEC) else 10

    # Wipe and rebuild
    retriever.clear()
    retriever._collection_ready = False

    from qdrant_client.http import models as qmodels

    print(f"  [ingest:{strategy_name}] building collection '{collection}'...")
    t0 = time.time()
    total_chunks = 0
    pending: list[TextChunk] = []

    def flush():
        nonlocal total_chunks
        if not pending:
            return
        texts = [c.text for c in pending]
        # embed in sub-batches of embed_batch_size
        vectors: list[list[float]] = []
        for start in range(0, len(texts), embed_batch_size):
            vecs = embedding_client.embed_texts(texts[start:start + embed_batch_size])
            vectors.extend(vecs)

        if not retriever._collection_ready:
            retriever._ensure_collection(len(vectors[0]))

        for start in range(0, len(pending), upsert_batch_size):
            batch_chunks = pending[start:start + upsert_batch_size]
            batch_vecs   = vectors[start:start + upsert_batch_size]
            points = [
                qmodels.PointStruct(
                    id=_doc_id_to_qdrant(c.chunk_id),
                    vector=vec,
                    payload={"text": c.text, "metadata": c.metadata, "original_id": c.chunk_id},
                )
                for c, vec in zip(batch_chunks, batch_vecs)
            ]
            retriever.client.upsert(collection_name=collection, points=points)
        total_chunks += len(pending)
        pending.clear()

    for article in iter_articles(jsonl_path, max_articles=max_articles):
        if "REDIRECT" in article.abstract[:20]:
            continue
        chunks = chunk_fn(article)
        pending.extend(chunks)
        if len(pending) >= 200:
            flush()

    flush()
    elapsed = time.time() - t0
    print(f"  [ingest:{strategy_name}] done — {total_chunks:,} chunks in {elapsed:.0f}s")
    return retriever


# ---------------------------------------------------------------------------
# RAGAS scoring
# ---------------------------------------------------------------------------

def _init_ragas_metrics(base_url: str, api_key: str):
    from openai import AsyncOpenAI
    from ragas.llms import llm_factory
    from ragas.embeddings import OpenAIEmbeddings as RagasEmbeddings
    from ragas.metrics.collections import (
        ContextPrecisionWithoutReference,
        ContextRecall,
        Faithfulness,
        AnswerRelevancy,
    )
    # Both LLM and embeddings must use AsyncOpenAI for ascore() to work
    async_client = AsyncOpenAI(base_url=base_url, api_key=api_key)

    # max_tokens must be large enough for RAGAS JSON reasoning chains (~2k–4k tokens)
    llm = llm_factory("qwen-plus", client=async_client, max_tokens=4096)
    emb = RagasEmbeddings(client=async_client, model="text-embedding-v4")
    return {
        "context_precision": ContextPrecisionWithoutReference(llm=llm),
        "context_recall":    ContextRecall(llm=llm),
        "faithfulness":      Faithfulness(llm=llm),
        "answer_relevancy":  AnswerRelevancy(llm=llm, embeddings=emb),
    }


def _to_float(result) -> float:
    """Extract float from a MetricResult or plain numeric."""
    if hasattr(result, "result"):
        return float(result.result)
    return float(result)


async def _score_one_async(metrics: dict, question: str, reference: str,
                           retrieved_contexts: list[str], response: str) -> dict[str, float]:
    """Async scoring — required by RAGAS 0.4.x which uses async LLM calls internally."""
    import asyncio

    scores: dict[str, float] = {}

    cp = await metrics["context_precision"].ascore(
        user_input=question, response=response, retrieved_contexts=retrieved_contexts
    )
    scores["context_precision"] = _to_float(cp)

    cr = await metrics["context_recall"].ascore(
        user_input=question, retrieved_contexts=retrieved_contexts, reference=reference
    )
    scores["context_recall"] = _to_float(cr)

    fa = await metrics["faithfulness"].ascore(
        user_input=question, response=response, retrieved_contexts=retrieved_contexts
    )
    scores["faithfulness"] = _to_float(fa)

    ar = await metrics["answer_relevancy"].ascore(
        user_input=question, response=response
    )
    scores["answer_relevancy"] = _to_float(ar)

    return scores


def score_one(metrics: dict, question: str, reference: str,
              retrieved_contexts: list[str], response: str) -> dict[str, float]:
    """Score a single QA sample. Runs the async scorer in a fresh event loop."""
    import asyncio
    return asyncio.run(_score_one_async(metrics, question, reference, retrieved_contexts, response))


def evaluate_strategy(
    strategy_name: str,
    retriever: QdrantRetriever,
    generator: LLMGenerator,
    questions: list[dict],
    ragas_metrics: dict,
    top_k: int = 5,
) -> list[dict]:
    """Run all questions against one strategy, return per-question score dicts."""
    results = []
    for i, qa in enumerate(questions):
        question = qa["question"]
        reference = qa.get("reference", "")
        print(f"    Q{i+1}/{len(questions)}: {question[:60]}...")

        # Retrieve
        try:
            hits = retriever.similarity_search(question, top_k=top_k, rerank=False)
            contexts = [h.text for h in hits]
        except Exception as e:
            print(f"      [WARN] retrieval error: {e}")
            contexts = []

        if not contexts:
            results.append({"question": question, "error": "no_retrieval"})
            continue

        # Generate
        try:
            context_text = "\n\n".join(f"[{j+1}] {c}" for j, c in enumerate(contexts))
            gen = generator.generate(question, context_text)
            response = gen.answer
        except Exception as e:
            print(f"      [WARN] generation error: {e}")
            response = contexts[0][:200]  # fallback: first context as answer

        # Score
        try:
            scores = score_one(ragas_metrics, question, reference, contexts, response)
        except Exception as e:
            print(f"      [WARN] scoring error: {e}")
            scores = {}

        results.append({
            "question": question,
            "reference": reference,
            "response": response,
            "contexts": contexts,
            "scores": scores,
        })

    return results


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(all_results: dict[str, list[dict]]) -> None:
    metric_names = ["context_precision", "context_recall", "faithfulness", "answer_relevancy"]

    # Compute averages per strategy
    averages: dict[str, dict[str, float]] = {}
    for strategy, results in all_results.items():
        agg: dict[str, list[float]] = {m: [] for m in metric_names}
        for r in results:
            for m in metric_names:
                v = r.get("scores", {}).get(m)
                if v is not None:
                    agg[m].append(v)
        averages[strategy] = {
            m: (sum(vals) / len(vals) if vals else float("nan"))
            for m, vals in agg.items()
        }

    strategies = list(all_results.keys())
    col_w = 14

    print("\n" + "=" * 70)
    print("RAGAS EVALUATION RESULTS")
    print("=" * 70)

    # Header
    header = f"{'Metric':<22}" + "".join(f"{s:>{col_w}}" for s in strategies)
    print(header)
    print("-" * (22 + col_w * len(strategies)))

    for m in metric_names:
        row = f"{m:<22}"
        best_val = max(averages[s][m] for s in strategies)
        for s in strategies:
            val = averages[s][m]
            marker = " *" if abs(val - best_val) < 1e-6 else "  "
            row += f"{val:>{col_w-2}.3f}{marker}"
        print(row)

    print("-" * (22 + col_w * len(strategies)))
    print("(* = best score for this metric)\n")

    # Per-question breakdown
    print("PER-QUESTION SCORES")
    print("-" * 70)
    n_questions = max(len(v) for v in all_results.values())
    for i in range(n_questions):
        q_text = ""
        for results in all_results.values():
            if i < len(results):
                q_text = results[i]["question"]
                break
        print(f"\nQ{i+1}: {q_text[:80]}")
        for m in metric_names:
            row = f"  {m:<20}"
            for s in strategies:
                results = all_results.get(s, [])
                if i < len(results):
                    v = results[i].get("scores", {}).get(m, float("nan"))
                    row += f"  {s}={v:.2f}"
            print(row)

    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)
    # Simple scoring: rank each metric, sum ranks
    rank_sum = {s: 0 for s in strategies}
    for m in metric_names:
        sorted_s = sorted(strategies, key=lambda s: averages[s][m], reverse=True)
        for rank, s in enumerate(sorted_s):
            rank_sum[s] += rank
    best = min(rank_sum, key=rank_sum.get)
    print(f"Best overall strategy: {best}")
    for s in strategies:
        avg_score = sum(averages[s][m] for m in metric_names) / len(metric_names)
        print(f"  {s:<12}: avg={avg_score:.3f}  rank_sum={rank_sum[s]}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _model_tag(model_name: str) -> str:
    """Short tag for collection naming: 'BAAI/bge-m3' → 'bgem3', 'text-embedding-v4' → 'qwen'."""
    name = model_name.lower()
    if "bge-m3" in name:
        return "bgem3"
    if "bge-large" in name:
        return "bgelarge"
    if "bge-small" in name:
        return "bgesmall"
    if "text-embedding" in name:
        return "qwen"
    # Generic: strip non-alphanumeric
    import re
    return re.sub(r"[^a-z0-9]", "", name)[:12]


def parse_args():
    p = argparse.ArgumentParser(
        description="Compare chunking strategies with RAGAS",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--file",
        default="C:/learning/enwiki_namespace_0/enwiki_namespace_0_0.jsonl",
        help="Wikipedia JSONL file to ingest from",
    )
    p.add_argument("--max-articles", type=int, default=500,
                   help="Articles to ingest per strategy")
    p.add_argument("--questions", default=None,
                   help="JSONL file with {question, reference} pairs (uses built-in if omitted)")
    p.add_argument("--top-k", type=int, default=5,
                   help="Chunks to retrieve per question")
    p.add_argument("--strategies", default="section,fixed512,fixed256",
                   help="Comma-separated list of strategies to compare")
    p.add_argument(
        "--embedding",
        default="BAAI/bge-m3",
        help=(
            "Embedding model. HuggingFace IDs (containing '/') use LocalEmbeddingClient "
            "(sentence-transformers). API model names use EmbeddingClient (Qwen/OpenAI). "
            "Use 'text-embedding-v4' for Qwen API."
        ),
    )
    p.add_argument(
        "--embed-batch-size",
        type=int,
        default=None,
        help="Override embedding batch size (default: 32 for local, 10 for API)",
    )
    p.add_argument("--skip-ingest", action="store_true",
                   help="Skip ingest, use existing eval_* collections")
    p.add_argument(
        "--also-qwen-section",
        action="store_true",
        help=(
            "Also include the existing eval_section (Qwen-embedded) collection as a "
            "baseline. Requires Qdrant to have the 'eval_section' collection from the "
            "previous Qwen run."
        ),
    )
    p.add_argument("--output", default=None,
                   help="Save full results to this JSON file")
    return p.parse_args()


def main():
    args = parse_args()
    jsonl_path = Path(args.file)
    strategy_names = [s.strip() for s in args.strategies.split(",")]

    for s in strategy_names:
        if s not in STRATEGIES:
            print(f"Unknown strategy: {s}. Available: {list(STRATEGIES)}", file=sys.stderr)
            sys.exit(1)

    # Load questions
    if args.questions:
        with open(args.questions, encoding="utf-8") as f:
            questions = [json.loads(line) for line in f if line.strip()]
    else:
        questions = BUILTIN_QUESTIONS

    tag = _model_tag(args.embedding)
    print(f"Embedding model : {args.embedding}  (tag: {tag})")
    print(f"Strategies      : {strategy_names}")
    print(f"Articles/strategy: {args.max_articles}")
    print(f"Questions       : {len(questions)}\n")

    embedding_client = build_embedding_client(
        model=args.embedding,
        batch_size=args.embed_batch_size,
    )
    generator = LLMGenerator()

    # Step 1: ingest
    retrievers: dict[str, QdrantRetriever] = {}
    if not args.skip_ingest:
        print("=== STEP 1: INGESTING ===")
        for name in strategy_names:
            collection = f"eval_{name}_{tag}"
            print(f"\nStrategy: {name}  →  collection: {collection}")
            retrievers[name] = ingest_strategy(
                strategy_name=name,
                chunk_fn=STRATEGIES[name],
                jsonl_path=jsonl_path,
                max_articles=args.max_articles,
                embedding_client=embedding_client,
                collection_name=collection,
            )
    else:
        print("=== STEP 1: SKIPPING INGEST (using existing collections) ===")
        for name in strategy_names:
            collection = f"eval_{name}_{tag}"
            print(f"  Using collection: {collection}")
            retrievers[name] = _make_retriever(collection, embedding_client)

    # Optionally add Qwen-embedded section baseline (uses existing eval_section collection)
    if args.also_qwen_section:
        print("\n  Adding Qwen section baseline → collection: eval_section")
        qwen_embedding = EmbeddingClient()
        retrievers["section_qwen"] = _make_retriever("eval_section", qwen_embedding)

    # Step 2: init RAGAS
    print("\n=== STEP 2: INITIALISING RAGAS METRICS ===")
    ragas_metrics = _init_ragas_metrics(
        base_url=os.getenv("QWEN_BASE_URL"),
        api_key=os.getenv("QWEN_API_KEY"),
    )
    print("Metrics ready:", list(ragas_metrics.keys()))

    # Step 3: evaluate
    print("\n=== STEP 3: EVALUATING ===")
    all_results: dict[str, list[dict]] = {}
    for run_name, retriever in retrievers.items():
        label = f"{run_name}+{tag}" if run_name != "section_qwen" else "section+qwen"
        print(f"\n--- {label} ---")
        all_results[label] = evaluate_strategy(
            strategy_name=run_name,
            retriever=retriever,
            generator=generator,
            questions=questions,
            ragas_metrics=ragas_metrics,
            top_k=args.top_k,
        )

    # Step 4: report
    print_report(all_results)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"\nFull results saved to: {args.output}")


if __name__ == "__main__":
    main()
