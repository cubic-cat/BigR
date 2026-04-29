"""Wikipedia JSONL parser for Wikimedia Enterprise Structured Contents format.

Each line in the input file is a JSON object representing one Wikipedia article.
This loader extracts article sections, cleans prose text, and yields structured
records suitable for downstream chunking and embedding.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator, Iterable


@dataclass
class WikiSection:
    """One section of a Wikipedia article."""

    title: str
    text: str
    depth: int = 1


@dataclass
class WikiArticle:
    """Parsed representation of a single Wikipedia article."""

    article_id: int
    title: str
    url: str
    abstract: str
    sections: list[WikiSection] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)
    language: str = "en"


def _clean_text(text: str) -> str:
    """Remove wiki markup artifacts and collapse whitespace."""
    # Strip template-like curly brace blocks (e.g. {{cite web}})
    text = re.sub(r"\{\{[^}]*\}\}", " ", text)
    # Strip HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Strip leading bullets/hashes used as wiki list markers
    text = re.sub(r"^[*#:;]+", "", text, flags=re.MULTILINE)
    # Collapse multiple spaces and blank lines
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_sections(raw_sections: list[dict]) -> list[WikiSection]:
    """Recursively flatten the nested sections list into a flat list.

    Each entry in raw_sections is a section dict with:
      - "name": section title
      - "has_parts": list of parts, where each part is either:
          * {"type": "paragraph", "value": "<prose text>"}
          * {"type": "section", "name": "...", "has_parts": [...]}  (sub-section)
          * {"type": "list", ...}  (skipped)
    """
    results: list[WikiSection] = []

    def _walk(sec: dict, depth: int) -> None:
        title = sec.get("name", "").strip()
        content_blocks: list[str] = []
        sub_sections: list[dict] = []

        for part in sec.get("has_parts", []):
            part_type = part.get("type", "")
            if part_type == "paragraph":
                value = part.get("value", "")
                if isinstance(value, str) and value.strip():
                    content_blocks.append(value)
            elif part_type == "section":
                sub_sections.append(part)
            # skip: list, image, table, references, etc.

        prose = _clean_text("\n\n".join(content_blocks))
        # Skip the bare "Abstract" section — its text is already in article.abstract
        if prose and title.lower() != "abstract":
            results.append(WikiSection(title=title, text=prose, depth=depth))

        for sub in sub_sections:
            _walk(sub, depth + 1)

    for top_sec in raw_sections:
        _walk(top_sec, depth=1)

    return results


def parse_article(raw: dict) -> WikiArticle | None:
    """Parse one raw JSON dict into a WikiArticle.  Returns None on failure."""
    try:
        article_id = int(raw.get("identifier", 0))
        title = str(raw.get("name", "")).strip()
        url = str(raw.get("url", ""))
        language = str(raw.get("in_language", {}).get("identifier", "en"))
        abstract = _clean_text(str(raw.get("abstract", "")))

        raw_sections = raw.get("article_sections", raw.get("sections", []))
        sections = _extract_sections(raw_sections) if isinstance(raw_sections, list) else []

        categories = [
            cat.get("name", "") for cat in raw.get("categories", []) if isinstance(cat, dict)
        ]

        return WikiArticle(
            article_id=article_id,
            title=title,
            url=url,
            abstract=abstract,
            sections=sections,
            categories=categories,
            language=language,
        )
    except Exception:  # noqa: BLE001
        return None


def iter_articles(
    path: str | Path,
    *,
    max_articles: int | None = None,
    skip: int = 0,
    encoding: str = "utf-8",
) -> Generator[WikiArticle, None, None]:
    """Yield parsed WikiArticle objects from a Wikipedia JSONL file.

    Args:
        path: Path to the `.jsonl` or `.ndjson` file.
        max_articles: Stop after this many successfully parsed articles.
        skip: Skip the first N lines (for resuming interrupted ingestion).
        encoding: File encoding (default UTF-8).
    """
    path = Path(path)
    count = 0
    with path.open(encoding=encoding, errors="replace") as fh:
        for line_no, line in enumerate(fh):
            if line_no < skip:
                continue
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                continue
            article = parse_article(raw)
            if article is None or not article.title:
                continue
            yield article
            count += 1
            if max_articles is not None and count >= max_articles:
                break


def iter_articles_batch(
    path: str | Path,
    batch_size: int = 100,
    **kwargs,
) -> Generator[list[WikiArticle], None, None]:
    """Yield lists of WikiArticle in batches for bulk processing."""
    batch: list[WikiArticle] = []
    for article in iter_articles(path, **kwargs):
        batch.append(article)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch
