"""Vector store configuration.

Supports two backends:
  - "local"   : JSON file-based store (default, no extra dependencies)
  - "qdrant"  : Qdrant vector database (requires qdrant-client)

Qdrant connection is controlled by environment variables:
  QDRANT_HOST, QDRANT_PORT, QDRANT_URL, QDRANT_API_KEY, QDRANT_VECTOR_SIZE
"""

from __future__ import annotations

import os
from pathlib import Path
from dataclasses import asdict, dataclass


@dataclass(slots=True, frozen=True)
class VectorDBConfig:
    """Settings for the vector store backend."""

    provider: str = "local"           # "local" | "qdrant"
    collection_name: str = "rag_knowledge_base"
    distance_metric: str = "cosine"
    persist_directory: str = "vector_store"

    @classmethod
    def from_env(cls) -> "VectorDBConfig":
        defaults = cls()
        return cls(
            provider=os.getenv("VECTOR_DB_PROVIDER", defaults.provider),
            collection_name=os.getenv(
                "VECTOR_DB_COLLECTION_NAME",
                defaults.collection_name,
            ),
            distance_metric=os.getenv(
                "VECTOR_DB_DISTANCE_METRIC",
                defaults.distance_metric,
            ),
            persist_directory=os.getenv(
                "VECTOR_DB_PERSIST_DIRECTORY",
                defaults.persist_directory,
            ),
        )

    @property
    def storage_path(self) -> str:
        return str(Path(self.persist_directory))

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["storage_path"] = self.storage_path
        return data
