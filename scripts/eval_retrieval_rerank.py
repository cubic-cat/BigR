#!/usr/bin/env python3
"""Evaluate rerank strategies on top of BM25 retrieval (best retrieval method).

Experiment matrix:
  Retrieval: bm25 (fixed, based on retrieval evaluation findings)
  Rerank:    none | keyword | cross_encoder
  Enhancer:  identity | rewrite | expansion

Metrics: Recall@K, Precision@K, MRR, nDCG@K, Latency

Usage:
    # First generate test set
    python scripts/generate_test_set.py --collection documents --max-articles 100 --output test_set.jsonl
    
    # Then run rerank evaluation
    python scripts/eval_retrieval_rerank.py --questions test_set.jsonl

Outputs:
  1) Console table with progress
  2) JSON artifact (default: output/rerank_eval_results.json)
  3) CSV artifact  (default: output/rerank_eval_results.csv)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if TYPE_CHECKING:
    from core.search_types import SearchResult


@dataclass(slots=True)
class EvalRecord:
    retrieval: str
    rerank: str
    enhancer: str
    recall_at_k: float
    precision_at_k: float
    mrr_at_k: float
    ndcg_at_k: float
    latency_ms_avg: float
    latency_ms_p95: float
    questions: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "retrieval": self.retrieval,
            "rerank": self.rerank,
            "enhancer": self.enhancer,
            "recall_at_k": self.recall_at_k,
            "precision_at_k": self.precision_at_k,
            "mrr_at_k": self.mrr_at_k,
            "ndcg_at_k": self.ndcg_at_k,
            "latency_ms_avg": self.latency_ms_avg,
            "latency_ms_p95": self.latency_ms_p95,
            "questions": self.questions,
        }


BUILTIN_QUESTIONS = [
    {
        "question": "What was the lowest air pressure recorded during the 1906 Mississippi hurricane?",
        "reference": "The lowest air pressure recorded in Mobile was 977 mbar during the 1906 Mississippi hurricane.",
    },
    {
        "question": "When was asteroid 1214 Richilde discovered and by whom?",
        "reference": "Richilde was discovered on 1 January 1932 by German astronomer Max Wolf at the Heidelberg-Konigstuhl State Observatory.",
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
        "question": "How many demands did the NotAgainSU protesters make to Syracuse University?",
        "reference": "The protesters initially made 19 demands to Chancellor Kent Syverud, which was later expanded to 34.",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate rerank strategies on BM25 retrieval.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--questions", default=None, help="JSONL file with {question, reference}")
    parser.add_argument("--top-k", type=int, default=5, help="Evaluation cutoff K")
    parser.add_argument(
        "--rerank-methods",
        default="none,keyword,cross_encoder",
        help="Comma-separated: none,keyword,cross_encoder",
    )
    parser.add_argument(
        "--enhancers",
        default="identity",
        help="Comma-separated: identity,rewrite,expansion",
    )
    parser.add_argument(
        "--min-overlap",
        type=float,
        default=0.12,
        help="Overlap threshold to treat a retrieved doc as relevant",
    )
    parser.add_argument(
        "--output-json",
        default="output/rerank_eval_results.json",
        help="Path to save JSON results",
    )
    parser.add_argument(
        "--output-csv",
        default="output/rerank_eval_results.csv",
        help="Path to save CSV results",
    )
    parser.add_argument(
        "--batch-size", type=int, default=10, help="Progress reporting batch size"
    )
    return parser.parse_args()


def load_questions(path: str | None) -> list[dict[str, str]]:
    if not path:
        print("Warning: No --questions provided, using built-in questions (5 questions)")
        return list(BUILTIN_QUESTIONS)

    questions: list[dict[str, str]] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            question = str(item.get("question", "")).strip()
            reference = str(item.get("reference", "")).strip()
            if question and reference:
                questions.append({"question": question, "reference": reference})
    if not questions:
        raise ValueError("No valid questions loaded from --questions")
    return questions


def tokenize(text: str) -> set[str]:
    normalized = "".join(char.lower() if char.isalnum() else " " for char in (text or ""))
    return {token for token in normalized.split() if token}


def lexical_overlap(reference: str, text: str) -> float:
    reference_tokens = tokenize(reference)
    if not reference_tokens:
        return 0.0
    candidate_tokens = tokenize(text)
    if not candidate_tokens:
        return 0.0
    overlap = len(reference_tokens & candidate_tokens)
    return overlap / len(reference_tokens)


def compute_mrr(relevances: Sequence[float]) -> float:
    for rank, rel in enumerate(relevances, start=1):
        if rel > 0:
            return 1.0 / rank
    return 0.0


def compute_ndcg(relevances: Sequence[float], ideal_relevance: float = 1.0) -> float:
    dcg = 0.0
    for rank, rel in enumerate(relevances, start=1):
        dcg += (2**rel - 1) / math.log2(rank + 1)
    idcg = (2**ideal_relevance - 1) / math.log2(2) if ideal_relevance > 0 else 0.0
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def compute_precision(retrieved_relevances: Sequence[float], min_overlap: float) -> float:
    if not retrieved_relevances:
        return 0.0
    relevant_count = sum(1 for rel in retrieved_relevances if rel >= min_overlap)
    return relevant_count / len(retrieved_relevances)


def dedupe_results(results: Iterable[SearchResult]) -> list[SearchResult]:
    seen: set[str] = set()
    unique: list[SearchResult] = []
    for item in results:
        if item.id in seen:
            continue
        seen.add(item.id)
        unique.append(item)
    return unique


def evaluate_configuration(
    *,
    chain,
    questions: Sequence[dict[str, str]],
    rerank_method: str,
    enhancer_name: str,
    top_k: int,
    min_overlap: float,
    batch_size: int,
) -> EvalRecord:
    from core.query_enhancer import create_query_enhancer
    from core.rerank_experiments import create_rerank_strategy

    enhancer = create_query_enhancer(enhancer_name)
    reranker = create_rerank_strategy(rerank_method)

    recall_scores: list[float] = []
    precision_scores: list[float] = []
    mrr_scores: list[float] = []
    ndcg_scores: list[float] = []
    latencies_ms: list[float] = []

    n = len(questions)
    
    for i, qa in enumerate(questions, 1):
        query = qa["question"]
        reference = qa["reference"]

        started = time.perf_counter()
        expanded_queries = enhancer.enhance(query) or [query]

        merged_candidates: list[SearchResult] = []
        for expanded_query in expanded_queries:
            hits = chain.search(
                expanded_query,
                top_k=20,  # BM25 先召回 Top-20
                retrieval_method="sparse",
                rerank=False,
                enable_query_enhancer=False,
            )
            merged_candidates.extend(hits)

        merged_candidates = dedupe_results(merged_candidates)
        reranked = reranker.rerank(query, merged_candidates, top_k)
        elapsed_ms = (time.perf_counter() - started) * 1000
        latencies_ms.append(elapsed_ms)

        relevances = [lexical_overlap(reference, item.text) for item in reranked]
        is_recalled = 1.0 if any(rel >= min_overlap for rel in relevances) else 0.0
        recall_scores.append(is_recalled)

        precision_scores.append(compute_precision(relevances, min_overlap))

        binary_relevances = [1.0 if rel >= min_overlap else 0.0 for rel in relevances]
        mrr_scores.append(compute_mrr(binary_relevances))
        ndcg_scores.append(compute_ndcg(relevances))

        if i % batch_size == 0 or i == n:
            avg_recall = sum(recall_scores) / i
            avg_mrr = sum(mrr_scores) / i
            avg_latency = sum(latencies_ms) / i
            print(f"  [{i}/{n}] Recall@K={avg_recall:.4f}, MRR@K={avg_mrr:.4f}, Latency={avg_latency:.2f}ms")

    latency_p95 = (
        statistics.quantiles(latencies_ms, n=20)[18]
        if len(latencies_ms) >= 20
        else max(latencies_ms, default=0.0)
    )

    return EvalRecord(
        retrieval="bm25",
        rerank=rerank_method,
        enhancer=enhancer_name,
        recall_at_k=sum(recall_scores) / len(recall_scores),
        precision_at_k=sum(precision_scores) / len(precision_scores),
        mrr_at_k=sum(mrr_scores) / len(mrr_scores),
        ndcg_at_k=sum(ndcg_scores) / len(ndcg_scores),
        latency_ms_avg=sum(latencies_ms) / len(latencies_ms),
        latency_ms_p95=latency_p95,
        questions=len(questions),
    )


def print_table(records: Sequence[EvalRecord]) -> None:
    headers = [
        "retrieval",
        "rerank",
        "enhancer",
        "recall@k",
        "precision@k",
        "mrr@k",
        "ndcg@k",
        "latency_avg(ms)",
        "latency_p95(ms)",
    ]
    print("\n" + "=" * 150)
    print(" | ".join(headers))
    print("-" * 150)
    for item in records:
        print(
            f"{item.retrieval:8} | {item.rerank:12} | {item.enhancer:9} | "
            f"{item.recall_at_k:8.4f} | {item.precision_at_k:10.4f} | "
            f"{item.mrr_at_k:8.4f} | {item.ndcg_at_k:8.4f} | "
            f"{item.latency_ms_avg:14.2f} | {item.latency_ms_p95:14.2f}"
        )
    print("=" * 150)


def save_json(path: str, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def save_csv(path: str, records: Sequence[EvalRecord]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "retrieval",
                "rerank",
                "enhancer",
                "recall_at_k",
                "precision_at_k",
                "mrr_at_k",
                "ndcg_at_k",
                "latency_ms_avg",
                "latency_ms_p95",
                "questions",
            ],
        )
        writer.writeheader()
        for item in records:
            writer.writerow(item.to_dict())


def main() -> None:
    args = parse_args()
    from core.rag_chain import RAGChain

    chain = RAGChain()

    questions = load_questions(args.questions)
    print(f"Loaded {len(questions)} test questions")

    rerank_methods = [item.strip().lower() for item in args.rerank_methods.split(",") if item.strip()]
    enhancers = [item.strip().lower() for item in args.enhancers.split(",") if item.strip()]

    total_configs = len(rerank_methods) * len(enhancers)
    print(f"Testing {total_configs} configurations (rerank: {rerank_methods}, enhancers: {enhancers})")
    print(f"Retrieval method: BM25 (fixed as best retrieval method)")
    print(f"Top-K: {args.top_k}, Min overlap: {args.min_overlap}")
    print()

    records: list[EvalRecord] = []
    config_idx = 0

    for rerank in rerank_methods:
        for enhancer in enhancers:
            config_idx += 1
            print(f"\n[{config_idx}/{total_configs}] Testing: rerank={rerank}, enhancer={enhancer}")
            print("-" * 60)
            
            record = evaluate_configuration(
                chain=chain,
                questions=questions,
                rerank_method=rerank,
                enhancer_name=enhancer,
                top_k=args.top_k,
                min_overlap=args.min_overlap,
                batch_size=args.batch_size,
            )
            records.append(record)

            print(f"  Completed: Recall@K={record.recall_at_k:.4f}, MRR@K={record.mrr_at_k:.4f}, "
                  f"Latency={record.latency_ms_avg:.2f}ms")

    best = sorted(
        records,
        key=lambda item: (
            -item.recall_at_k,
            -item.precision_at_k,
            -item.mrr_at_k,
            -item.ndcg_at_k,
            item.latency_ms_avg,
        ),
    )[0]

    print("\n" + "=" * 150)
    print("FINAL RESULTS")
    print("=" * 150)
    print_table(records)
    
    print("\n[BEST CONFIGURATION]")
    print(f"  retrieval: {best.retrieval}")
    print(f"  rerank: {best.rerank}")
    print(f"  enhancer: {best.enhancer}")
    print(f"  recall@k: {best.recall_at_k:.4f}")
    print(f"  precision@k: {best.precision_at_k:.4f}")
    print(f"  mrr@k: {best.mrr_at_k:.4f}")
    print(f"  ndcg@k: {best.ndcg_at_k:.4f}")
    print(f"  latency_avg: {best.latency_ms_avg:.2f}ms")
    print(f"  latency_p95: {best.latency_ms_p95:.2f}ms")
    print(f"  questions: {best.questions}")

    payload = {
        "config": {
            "top_k": args.top_k,
            "questions": len(questions),
            "retrieval_method": "bm25",
            "rerank_methods": rerank_methods,
            "enhancers": enhancers,
            "min_overlap": args.min_overlap,
        },
        "results": [item.to_dict() for item in records],
        "best": best.to_dict(),
    }
    save_json(args.output_json, payload)
    save_csv(args.output_csv, records)
    print(f"\n[SAVED] JSON: {args.output_json}")
    print(f"[SAVED] CSV: {args.output_csv}")


if __name__ == "__main__":
    main()
