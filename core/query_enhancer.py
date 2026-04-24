"""Pluggable query enhancement strategies for retrieval-time optimization.

This module is intentionally decoupled from retriever / reranker logic:
- Query enhancement only produces one or more retrieval queries.
- Retrieval / rerank components decide how to consume those queries.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Protocol, Sequence

from configs import llm_config

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None


class QueryEnhancer(Protocol):
    """Unified interface for query enhancement strategies."""

    name: str

    def enhance(self, query: str) -> list[str]:
        """Return one or multiple retrieval queries."""

    def enhance_batch(self, queries: Sequence[str]) -> list[list[str]]:
        """Batch variant for future throughput optimization."""


@dataclass(slots=True)
class IdentityEnhancer:
    """No-op enhancer used as a strict baseline."""

    name: str = "identity"

    def enhance(self, query: str) -> list[str]:
        cleaned = (query or "").strip()
        return [cleaned] if cleaned else []

    def enhance_batch(self, queries: Sequence[str]) -> list[list[str]]:
        return [self.enhance(query) for query in queries]


class _BaseLLMEnhancer:
    """Shared LLM client helpers for rewrite / expansion enhancers."""

    def __init__(self) -> None:
        self._client: OpenAI | None = None

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            if OpenAI is None:
                raise ImportError(
                    "openai package is required for LLM-based query enhancers. "
                    "Install it with `pip install openai`."
                )
            if not llm_config.api_key:
                raise ValueError(
                    "LLM API key is missing. Set LLM_API_KEY or provider-specific "
                    "key in `.env` before using rewrite/expansion enhancers."
                )
            kwargs: dict[str, object] = {
                "api_key": llm_config.api_key,
                "timeout": llm_config.request_timeout,
            }
            if llm_config.base_url:
                kwargs["base_url"] = llm_config.base_url
            self._client = OpenAI(**kwargs)
        return self._client

    def _chat(self, system_prompt: str, user_prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=llm_config.model_name,
            temperature=0.2,
            top_p=0.9,
            max_tokens=512,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return (response.choices[0].message.content or "").strip()

    @staticmethod
    def _normalize_unique(items: Sequence[str], *, max_items: int) -> list[str]:
        unique: list[str] = []
        seen: set[str] = set()
        for item in items:
            value = " ".join((item or "").split()).strip()
            key = value.lower()
            if not value or key in seen:
                continue
            seen.add(key)
            unique.append(value)
            if len(unique) >= max_items:
                break
        return unique


@dataclass(slots=True)
class SimpleRewriteEnhancer(_BaseLLMEnhancer):
    """Generate one optimized retrieval query with the same intent."""

    name: str = "rewrite"

    def __post_init__(self) -> None:
        _BaseLLMEnhancer.__init__(self)

    def enhance(self, query: str) -> list[str]:
        cleaned = (query or "").strip()
        if not cleaned:
            return []

        raw = self._chat(
            build_rewrite_system_prompt(),
            build_rewrite_user_prompt(cleaned),
        )
        rewritten = parse_single_line_output(raw) or cleaned
        return [rewritten]

    def enhance_batch(self, queries: Sequence[str]) -> list[list[str]]:
        return [self.enhance(query) for query in queries]


@dataclass(slots=True)
class QueryExpansionEnhancer(_BaseLLMEnhancer):
    """Generate 3-5 diversified retrieval queries for multi-query search."""

    name: str = "expansion"
    min_queries: int = 3
    max_queries: int = 5

    def __post_init__(self) -> None:
        _BaseLLMEnhancer.__init__(self)

    def enhance(self, query: str) -> list[str]:
        cleaned = (query or "").strip()
        if not cleaned:
            return []

        raw = self._chat(
            build_expansion_system_prompt(),
            build_expansion_user_prompt(
                cleaned,
                min_queries=self.min_queries,
                max_queries=self.max_queries,
            ),
        )
        candidates = parse_json_or_lines(raw)
        normalized = self._normalize_unique(candidates, max_items=self.max_queries)
        if len(normalized) < self.min_queries:
            # Keep deterministic fallback behavior to protect experiments.
            normalized = [cleaned]
        return normalized

    def enhance_batch(self, queries: Sequence[str]) -> list[list[str]]:
        return [self.enhance(query) for query in queries]


def build_rewrite_system_prompt() -> str:
    """System prompt for single-query rewriting."""
    return (
        "You optimize user queries for retrieval systems. "
        "Preserve intent, entities, numbers, and constraints. "
        "Output exactly one rewritten query without explanation."
    )


def build_rewrite_user_prompt(query: str) -> str:
    """User prompt for single-query rewriting."""
    return (
        "Rewrite the following user query for better document retrieval.\n"
        f"Query: {query}\n"
        "Return one line only."
    )


def build_expansion_system_prompt() -> str:
    """System prompt for query expansion."""
    return (
        "You generate retrieval query expansions. "
        "Produce diverse but intent-consistent phrasings and sub-queries. "
        "Do not fabricate entities. Return valid JSON."
    )


def build_expansion_user_prompt(query: str, *, min_queries: int, max_queries: int) -> str:
    """User prompt for multi-query expansion."""
    return (
        "Given a user query, generate multiple retrieval-oriented variants.\n"
        f"Original query: {query}\n"
        f"Return a JSON array with {min_queries} to {max_queries} strings.\n"
        "Each string should be concise and suitable for semantic/BM25 retrieval."
    )


def parse_single_line_output(raw: str) -> str:
    """Parse one-line text outputs robustly."""
    value = " ".join((raw or "").split()).strip()
    # Remove common bullet/numbering prefixes to keep output clean.
    value = re.sub(r"^\s*[-*\d\.\)]\s*", "", value)
    return value


def parse_json_or_lines(raw: str) -> list[str]:
    """Parse model output as JSON array first, then fallback to line parsing."""
    text = (raw or "").strip()
    if not text:
        return []

    try:
        data = json.loads(text)
        if isinstance(data, list):
            return [str(item) for item in data]
    except json.JSONDecodeError:
        pass

    lines: list[str] = []
    for line in text.splitlines():
        value = parse_single_line_output(line)
        if value:
            lines.append(value)
    return lines


def create_query_enhancer(name: str | None) -> QueryEnhancer:
    """Factory for query enhancers."""
    normalized = (name or "identity").strip().lower()
    if normalized in {"identity", "none"}:
        return IdentityEnhancer()
    if normalized in {"rewrite", "simple_rewrite"}:
        return SimpleRewriteEnhancer()
    if normalized in {"expansion", "query_expansion", "multi_query"}:
        return QueryExpansionEnhancer()
    raise ValueError(
        "Unknown query enhancer: "
        f"{normalized}. Available: identity, rewrite, expansion."
    )

