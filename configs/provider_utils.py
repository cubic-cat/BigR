"""Helpers for resolving provider-level settings from environment variables."""

from __future__ import annotations

import os

DEFAULT_BASE_URLS: dict[str, str] = {
    "openai": "https://api.openai.com/v1",
    "qwen": "https://dashscope.aliyuncs.com/compatible-mode/v1",
}


def _provider_prefix(provider: str) -> str:
    return provider.strip().replace("-", "_").replace(" ", "_").upper()


def get_optional_env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value in (None, ""):
        return default
    return value


def get_provider_default_base_url(provider: str) -> str | None:
    return DEFAULT_BASE_URLS.get(provider.strip().lower())


def resolve_provider_value(
    provider: str,
    suffix: str,
    *,
    explicit_env: str | None = None,
    default: str | None = None,
) -> str | None:
    if explicit_env:
        explicit_value = get_optional_env(explicit_env)
        if explicit_value is not None:
            return explicit_value

    provider_env_name = f"{_provider_prefix(provider)}_{suffix}"
    return get_optional_env(provider_env_name, default)


def mask_secret(secret: str | None) -> str | None:
    if not secret:
        return None
    if len(secret) <= 8:
        return "*" * len(secret)
    return f"{secret[:4]}...{secret[-4:]}"
