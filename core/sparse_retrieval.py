"""Sparse (BM25) retrieval strategy."""

from __future__ import annotations

import math
from typing import Any, Sequence

import jieba

from .embedding import SupportsEmbedding
from .retriever import RetrievalStrategy
from .search_types import SearchResult, VectorDocument

_STOPWORDS: set[str] = {
    # 中文常用停用词
    "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一", "一个", "上", "也",
    "很", "到", "说", "要", "去", "你", "会", "着", "没有", "看", "好", "自己", "这", "那",
    "之", "与", "及", "等", "或", "但", "而", "如果", "因为", "所以", "可以", "被", "让",
    "把", "给", "为", "于", "以", "及", "但", "对", "将", "还", "并", "从", "到", "向",
    "中", "前", "后", "内", "外", "里", "间", "边", "面", "头", "部", "下", "大", "小",
    "多", "少", "来", "过", "下", "得", "地", "着", "过", "吗", "呢", "吧", "啊", "哦",
    # 英文常用停用词
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "must", "shall", "can", "need", "dare",
    "ought", "used", "to", "of", "in", "for", "on", "with", "at", "by",
    "from", "as", "into", "through", "during", "before", "after", "above",
    "below", "between", "under", "and", "but", "or", "yet", "so", "if",
    "because", "although", "though", "while", "where", "when", "that",
    "which", "who", "whom", "whose", "what", "this", "these", "those",
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her",
    "us", "them", "my", "your", "his", "its", "our", "their", "this",
}


class SparseRetrievalStrategy(RetrievalStrategy):
    """BM25-based sparse retrieval over local documents."""

    name = "sparse"

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b

    def search(
        self,
        *,
        query: str,
        records: Sequence[VectorDocument],
        embedding_client: SupportsEmbedding,
        top_k: int = 4,
        min_score: float | None = None,
        **_: Any,
    ) -> list[SearchResult]:
        """Rank records by BM25 sparse score."""
        if not records:
            return []

        index = _build_index(records)
        query_tokens = tokenize_text(query)
        results: list[SearchResult] = []

        for record in records:
            score = bm25_score(
                query_tokens,
                index["doc_freqs"][record.id],
                index["doc_lens"][record.id],
                index["avgdl"],
                index["n_docs"],
                index["token_doc_counts"],
                self.k1,
                self.b,
            )
            if min_score is not None and score < min_score:
                continue

            results.append(
                SearchResult(
                    id=record.id,
                    text=record.text,
                    metadata=dict(record.metadata),
                    score=score,
                    retrieval_score=score,
                    vector_score=0.0,
                    retrieval_method=self.name,
                    rerank_method="",
                    details={"rerank_enabled": False},
                )
            )

        results.sort(key=lambda item: item.score, reverse=True)
        return results[: max(top_k, 0)]


def tokenize_text(text: str) -> list[str]:
    """Tokenize text using jieba and filter out common stopwords."""
    return [
        token.strip().lower()
        for token in jieba.lcut(text or "")
        if token.strip() and token.strip().lower() not in _STOPWORDS
    ]


def _build_index(records: Sequence[VectorDocument]) -> dict[str, Any]:
    """Build lightweight BM25 index from records."""
    doc_freqs: dict[str, dict[str, int]] = {}
    doc_lens: dict[str, int] = {}
    token_doc_counts: dict[str, int] = {}
    total_len = 0

    for record in records:
        tokens = tokenize_text(record.text)
        freq: dict[str, int] = {}
        for token in tokens:
            freq[token] = freq.get(token, 0) + 1

        doc_freqs[record.id] = freq
        doc_lens[record.id] = len(tokens)
        total_len += len(tokens)

        for token in freq:
            token_doc_counts[token] = token_doc_counts.get(token, 0) + 1

    n_docs = len(records)
    avgdl = total_len / n_docs if n_docs > 0 else 0.0

    return {
        "doc_freqs": doc_freqs,
        "doc_lens": doc_lens,
        "avgdl": avgdl,
        "n_docs": n_docs,
        "token_doc_counts": token_doc_counts,
    }


def bm25_score(
    query_tokens: list[str],
    doc_freqs: dict[str, int],
    doc_len: int,
    avgdl: float,
    n_docs: int,
    token_doc_counts: dict[str, int],
    k1: float,
    b: float,
) -> float:
    """Compute BM25 score for a single document."""
    if not query_tokens or avgdl == 0:
        return 0.0

    score = 0.0
    for token in query_tokens:
        n_q = token_doc_counts.get(token, 0)
        if n_q == 0:
            continue

        idf = math.log((n_docs - n_q + 0.5) / (n_q + 0.5) + 1.0)
        f = doc_freqs.get(token, 0)
        denom = f + k1 * (1 - b + b * (doc_len / avgdl))
        score += idf * (f * (k1 + 1)) / denom

    return score
