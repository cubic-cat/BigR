"""Text chunking strategies for RAG document preparation.

Primary strategy: split by section boundaries first.  If a section exceeds
max_tokens, apply a sliding-window secondary split.  Metadata (title, section
name, article_id) is preserved on every chunk so it can surface in citations.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import Iterable, Iterator

from .wiki_loader import WikiArticle, WikiSection


# ---------------------------------------------------------------------------
# Public chunk type
# ---------------------------------------------------------------------------

@dataclass
class TextChunk:
    """A chunk of text ready for embedding."""

    chunk_id: str
    text: str
    metadata: dict = field(default_factory=dict)

    def to_record(self) -> dict:
        """Convert to the dict format expected by LocalVectorRetriever / QdrantRetriever."""
        return {"id": self.chunk_id, "text": self.text, "metadata": self.metadata}


# ---------------------------------------------------------------------------
# Token estimation (no external tokenizer dependency)
# ---------------------------------------------------------------------------

def _approx_tokens(text: str) -> int:
    """Rough token count: ~1 token per 4 characters (English heuristic)."""
    return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Core splitting logic
# ---------------------------------------------------------------------------

def _sliding_window_split(
    text: str,
    max_tokens: int,
    overlap_tokens: int,
) -> list[str]:
    """Split *text* into overlapping windows measured in approximate tokens."""
    # Work at the word level so we don't cut mid-word
    words = text.split()
    if not words:
        return []

    # Estimate words-per-token (inverse of token density)
    chars_per_token = 4
    avg_word_len = max(1, sum(len(w) for w in words) / len(words))
    words_per_token = chars_per_token / avg_word_len  # ≥ 1 on average

    window_words = max(1, int(max_tokens * words_per_token))
    step_words = max(1, window_words - int(overlap_tokens * words_per_token))

    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + window_words, len(words))
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk.strip())
        if end >= len(words):
            break
        start += step_words

    return chunks


def split_section(
    section: WikiSection,
    max_tokens: int = 512,
    overlap_tokens: int = 50,
) -> list[str]:
    """Return a list of text chunks for a single section.

    If the section fits within max_tokens it is returned as-is.  Otherwise a
    sliding-window secondary split is applied.
    """
    text = section.text.strip()
    if not text:
        return []
    if _approx_tokens(text) <= max_tokens:
        return [text]
    return _sliding_window_split(text, max_tokens, overlap_tokens)


# ---------------------------------------------------------------------------
# High-level Wikipedia article → TextChunk list
# ---------------------------------------------------------------------------

def chunk_article(
    article: WikiArticle,
    *,
    max_tokens: int = 512,
    overlap_tokens: int = 50,
    include_abstract: bool = True,
) -> list[TextChunk]:
    """Convert a WikiArticle into a list of TextChunk objects.

    Each chunk carries rich metadata so downstream context assembly can show
    the article title and section name in citations.
    """
    chunks: list[TextChunk] = []
    base_meta = {
        "article_id": article.article_id,
        "title": article.title,
        "url": article.url,
        "language": article.language,
        "categories": article.categories[:10],  # cap to keep metadata small
    }

    # --- abstract as a standalone chunk ---
    if include_abstract and article.abstract:
        if _approx_tokens(article.abstract) <= max_tokens:
            texts = [article.abstract]
        else:
            texts = _sliding_window_split(article.abstract, max_tokens, overlap_tokens)
        for i, text in enumerate(texts):
            chunks.append(
                TextChunk(
                    chunk_id=_make_id(article.article_id, "abstract", i),
                    text=text,
                    metadata={**base_meta, "section": "Abstract", "chunk_index": i},
                )
            )

    # --- per-section chunks ---
    for sec in article.sections:
        sub_texts = split_section(sec, max_tokens=max_tokens, overlap_tokens=overlap_tokens)
        for i, text in enumerate(sub_texts):
            chunks.append(
                TextChunk(
                    chunk_id=_make_id(article.article_id, sec.title, i),
                    text=text,
                    metadata={
                        **base_meta,
                        "section": sec.title or "Section",
                        "section_depth": sec.depth,
                        "chunk_index": i,
                    },
                )
            )

    return chunks


def chunk_articles(
    articles: Iterable[WikiArticle],
    *,
    max_tokens: int = 512,
    overlap_tokens: int = 50,
) -> Iterator[TextChunk]:
    """Yield TextChunk objects for an iterable of WikiArticle objects."""
    for article in articles:
        yield from chunk_article(article, max_tokens=max_tokens, overlap_tokens=overlap_tokens)


# ---------------------------------------------------------------------------
# Plain-text document splitter (for non-Wikipedia plain text files)
# ---------------------------------------------------------------------------

def split_text(
    text: str,
    *,
    max_tokens: int = 512,
    overlap_tokens: int = 50,
    source: str = "",
) -> list[TextChunk]:
    """Split a plain-text string into overlapping chunks.

    Useful for the existing data/processed/ workflow.
    """
    chunks = _sliding_window_split(text, max_tokens, overlap_tokens)
    return [
        TextChunk(
            chunk_id=uuid.uuid4().hex,
            text=chunk,
            metadata={"source": source, "chunk_index": i},
        )
        for i, chunk in enumerate(chunks)
    ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_id(article_id: int, section_title: str, chunk_index: int) -> str:
    slug = re.sub(r"\W+", "_", section_title.lower())[:40].strip("_") or "s"
    return f"{article_id}_{slug}_{chunk_index}"
