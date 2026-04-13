"""Embedding model configuration."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass

from .provider_utils import (
    get_optional_env,
    get_provider_default_base_url,
    mask_secret,
    resolve_provider_value,
)

DEFAULT_EMBEDDING_MODELS: dict[str, str] = {
    "openai": "text-embedding-3-large",
    "qwen": "text-embedding-v4",
}


def _get_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value in (None, ""):
        return default
    return int(value)


def _get_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value in (None, ""):
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass(slots=True, frozen=True)
class EmbeddingConfig:
    """Runtime settings for embedding generation."""

    provider: str = "qwen"
    model_name: str = "text-embedding-v4"
    base_url: str | None = None
    api_key: str | None = None
    dimensions: int = 1024
    batch_size: int = 32
    normalize: bool = True
    request_timeout: int = 30

    @classmethod
    def from_env(cls) -> "EmbeddingConfig":
        defaults = cls()
        provider = get_optional_env("EMBEDDING_PROVIDER", defaults.provider) or defaults.provider
        default_model_name = DEFAULT_EMBEDDING_MODELS.get(
            provider.strip().lower(),
            defaults.model_name,
        )
        return cls(
            provider=provider,
            model_name=get_optional_env(
                "EMBEDDING_MODEL_NAME",
                default_model_name,
            ) or default_model_name,
            base_url=resolve_provider_value(
                provider,
                "BASE_URL",
                explicit_env="EMBEDDING_BASE_URL",
                default=get_provider_default_base_url(provider),
            ),
            api_key=resolve_provider_value(
                provider,
                "API_KEY",
                explicit_env="EMBEDDING_API_KEY",
            ),
            dimensions=_get_int("EMBEDDING_DIMENSIONS", defaults.dimensions),
            batch_size=_get_int("EMBEDDING_BATCH_SIZE", defaults.batch_size),
            normalize=_get_bool("EMBEDDING_NORMALIZE", defaults.normalize),
            request_timeout=_get_int(
                "EMBEDDING_REQUEST_TIMEOUT",
                defaults.request_timeout,
            ),
        )

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["api_key"] = mask_secret(self.api_key)
        return data
