"""Build the local vector knowledge base from `data/processed`."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core import RAGChain


def main() -> None:
    chain = RAGChain()
    if not chain.retriever.embedding_client.is_configured():
        raise ValueError(
            "Embedding API key is missing. Set EMBEDDING_API_KEY or the "
            "provider-specific key in `.env` before building the knowledge base."
        )

    documents = chain.build_knowledge_base(replace=True, persist=True)
    stats = chain.retriever.stats()
    print(f"Embedded {len(documents)} documents from {chain.knowledge_base_dir}.")
    print(f"Collection: {stats['collection_name']}")
    print(f"Store: {stats['store_path']}")
    print(f"Dimensions: {stats['dimensions']}")


if __name__ == "__main__":
    main()
