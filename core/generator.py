"""LLM answer generation for the RAG pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from configs import LLMConfig, llm_config

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful RAG assistant. Answer the user only from the retrieved "
    "context when the context is relevant. If the context is insufficient, say "
    "what is missing instead of fabricating facts. Keep the answer concise and "
    "use the same language as the user's question."
)


@dataclass(slots=True)
class ChatMessage:
    """Simple chat message structure."""

    role: str
    content: str


@dataclass(slots=True)
class GenerationResult:
    """Output of the answer generation step."""

    answer: str
    model_name: str
    provider: str
    messages: list[dict[str, str]] = field(default_factory=list)
    usage: dict[str, int | None] = field(default_factory=dict)


class LLMGenerator:
    """Thin wrapper over an OpenAI-compatible chat completion API."""

    def __init__(self, config: LLMConfig | None = None) -> None:
        self.config = config or llm_config
        self._client: OpenAI | None = None

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            if OpenAI is None:
                raise ImportError(
                    "openai package is required for LLMGenerator. "
                    "Install it with `pip install openai`."
                )
            if not self.config.api_key:
                raise ValueError(
                    "LLM API key is missing. Set LLM_API_KEY or the "
                    "provider-specific key in `.env`."
                )

            client_kwargs: dict[str, object] = {
                "api_key": self.config.api_key,
                "timeout": self.config.request_timeout,
            }
            if self.config.base_url:
                client_kwargs["base_url"] = self.config.base_url
            self._client = OpenAI(**client_kwargs)
        return self._client

    def is_configured(self) -> bool:
        """Return whether the generator can make API calls."""
        return bool(self.config.api_key)

    def build_messages(
        self,
        query: str,
        context: str,
        *,
        system_prompt: str | None = None,
        history: Sequence[ChatMessage | Mapping[str, str]] | None = None,
    ) -> list[dict[str, str]]:
        """Construct a prompt that conditions the model on retrieved context."""
        messages = [
            {
                "role": "system",
                "content": system_prompt or DEFAULT_SYSTEM_PROMPT,
            }
        ]

        for item in history or []:
            if isinstance(item, Mapping):
                role = str(item.get("role", "user"))
                content = str(item.get("content", ""))
            else:
                role = item.role
                content = item.content
            messages.append({"role": role, "content": content})

        context_block = context.strip() or "No relevant context was retrieved."
        user_prompt = (
            "Retrieved context:\n"
            f"{context_block}\n\n"
            "User question:\n"
            f"{query}\n\n"
            "Answer requirements:\n"
            "1. Prioritize the retrieved context.\n"
            "2. If context is insufficient, say so explicitly.\n"
            "3. Cite chunk numbers like [1], [2] when possible."
        )
        messages.append({"role": "user", "content": user_prompt})
        return messages

    def generate(
        self,
        query: str,
        context: str,
        *,
        system_prompt: str | None = None,
        history: Sequence[ChatMessage | Mapping[str, str]] | None = None,
    ) -> GenerationResult:
        """Generate an answer from the LLM using retrieved context."""
        messages = self.build_messages(
            query,
            context,
            system_prompt=system_prompt,
            history=history,
        )

        request_kwargs: dict[str, object] = {
            "model": self.config.model_name,
            "messages": messages,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "max_tokens": self.config.max_tokens,
        }

        if self.config.streaming:
            response_stream = self.client.chat.completions.create(
                **request_kwargs,
                stream=True,
            )
            chunks: list[str] = []
            for event in response_stream:
                for choice in getattr(event, "choices", []):
                    delta = getattr(choice, "delta", None)
                    content = getattr(delta, "content", None)
                    if content:
                        chunks.append(content)
            answer = "".join(chunks).strip()
            return GenerationResult(
                answer=answer,
                model_name=self.config.model_name,
                provider=self.config.provider,
                messages=messages,
                usage={},
            )

        response = self.client.chat.completions.create(**request_kwargs)
        answer = (response.choices[0].message.content or "").strip()
        return GenerationResult(
            answer=answer,
            model_name=response.model or self.config.model_name,
            provider=self.config.provider,
            messages=messages,
            usage=_usage_to_dict(getattr(response, "usage", None)),
        )


def _usage_to_dict(usage: Any) -> dict[str, int | None]:
    if usage is None:
        return {}
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }
