"""Load processed text files from the local data directory."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

DEFAULT_PROCESSED_DIR = Path("data/processed")
DEFAULT_TEXT_SUFFIXES = (".txt", ".md", ".markdown", ".text")
DEFAULT_ENCODINGS = ("utf-8", "utf-8-sig", "gb18030", "gbk")


@dataclass(slots=True)
class LoadedDocument:
    """A processed document loaded from disk."""

    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        """Convert to the record shape expected by the retriever."""
        return {
            "id": self.id,
            "text": self.text,
            "metadata": dict(self.metadata),
        }


def load_processed_documents(
    source_dir: str | Path = DEFAULT_PROCESSED_DIR,
    *,
    suffixes: Sequence[str] = DEFAULT_TEXT_SUFFIXES,
) -> list[LoadedDocument]:
    """Load all processed text-like files under the given directory."""
    directory = Path(source_dir)
    if not directory.exists():
        raise FileNotFoundError(f"Processed data directory does not exist: {directory}")
    if not directory.is_dir():
        raise NotADirectoryError(f"Processed data path is not a directory: {directory}")

    normalized_suffixes = {suffix.lower() for suffix in suffixes}
    documents: list[LoadedDocument] = []

    for file_path in sorted(_iter_files(directory)):
        if file_path.suffix.lower() not in normalized_suffixes:
            continue

        text = read_text_file(file_path)
        if not text.strip():
            continue

        relative_path = file_path.relative_to(directory)
        documents.append(
            LoadedDocument(
                id=str(relative_path).replace("\\", "/"),
                text=text,
                metadata={
                    "source": str(relative_path).replace("\\", "/"),
                    "file_name": file_path.name,
                    "relative_path": str(relative_path).replace("\\", "/"),
                    "absolute_path": str(file_path.resolve()),
                    "suffix": file_path.suffix.lower(),
                },
            )
        )

    return documents


def read_text_file(file_path: str | Path) -> str:
    """Read a text file with a small encoding fallback chain."""
    path = Path(file_path)
    last_error: UnicodeDecodeError | None = None

    for encoding in DEFAULT_ENCODINGS:
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError as error:
            last_error = error

    if last_error is not None:
        raise last_error
    return path.read_text()


def _iter_files(directory: Path) -> Iterable[Path]:
    for path in directory.rglob("*"):
        if path.is_file():
            yield path
