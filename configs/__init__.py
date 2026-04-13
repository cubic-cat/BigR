"""Project configuration exports."""

from __future__ import annotations

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

from .embedding_config import EmbeddingConfig
from .llm_config import LLMConfig
from .vector_db_config import VectorDBConfig

if load_dotenv is not None:
    load_dotenv()

embedding_config = EmbeddingConfig.from_env()
llm_config = LLMConfig.from_env()
vector_db_config = VectorDBConfig.from_env()


def reload_configs() -> tuple[EmbeddingConfig, LLMConfig, VectorDBConfig]:
    """Reload configuration values from environment variables."""
    global embedding_config, llm_config, vector_db_config

    if load_dotenv is not None:
        load_dotenv(override=True)

    embedding_config = EmbeddingConfig.from_env()
    llm_config = LLMConfig.from_env()
    vector_db_config = VectorDBConfig.from_env()
    return embedding_config, llm_config, vector_db_config


__all__ = [
    "EmbeddingConfig",
    "LLMConfig",
    "VectorDBConfig",
    "embedding_config",
    "llm_config",
    "reload_configs",
    "vector_db_config",
]
