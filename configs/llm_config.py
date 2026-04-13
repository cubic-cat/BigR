"""LLM generation configuration."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass

from .provider_utils import (
    get_optional_env,
    get_provider_default_base_url,
    mask_secret,
    resolve_provider_value,
)

DEFAULT_LLM_MODELS: dict[str, str] = {
    "openai": "gpt-4o-mini",
    "qwen": "qwen3.5-flash",
}


def _get_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value in (None, ""):
        return default
    return int(value)


def _get_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value in (None, ""):
        return default
    return float(value)


def _get_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value in (None, ""):
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass(slots=True, frozen=True)
class LLMConfig:
    """Runtime settings for the answer generation model."""

    provider: str = "qwen"
    model_name: str = "qwen3.5-flash"
    base_url: str | None = None
    api_key: str | None = None
    temperature: float = 0.2
    top_p: float = 0.9
    max_tokens: int = 1024
    streaming: bool = False
    request_timeout: int = 60

    @classmethod
    def from_env(cls) -> "LLMConfig":
        defaults = cls()
        provider = get_optional_env("LLM_PROVIDER", defaults.provider) or defaults.provider
        default_model_name = DEFAULT_LLM_MODELS.get(
            provider.strip().lower(),
            defaults.model_name,
        )
        return cls(
            provider=provider,
            model_name=get_optional_env(
                "LLM_MODEL_NAME",
                default_model_name,
            ) or default_model_name,
            base_url=resolve_provider_value(
                provider,
                "BASE_URL",
                explicit_env="LLM_BASE_URL",
                default=get_provider_default_base_url(provider),
            ),
            api_key=resolve_provider_value(
                provider,
                "API_KEY",
                explicit_env="LLM_API_KEY",
            ),
            temperature=_get_float("LLM_TEMPERATURE", defaults.temperature),
            top_p=_get_float("LLM_TOP_P", defaults.top_p),
            max_tokens=_get_int("LLM_MAX_TOKENS", defaults.max_tokens),
            streaming=_get_bool("LLM_STREAMING", defaults.streaming),
            request_timeout=_get_int(
                "LLM_REQUEST_TIMEOUT",
                defaults.request_timeout,
            ),
        )

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["api_key"] = mask_secret(self.api_key)
        return data
