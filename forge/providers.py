"""Provider abstraction: one factory that returns a LangChain chat model for any supported backend.

The agent loop only relies on the BaseChatModel interface (`bind_tools` +
`astream`), so swapping between a local Ollama model and a hosted endpoint is
a config change. Adding another provider (Groq, OpenAI, Anthropic...) means
adding one entry to PROVIDERS.

Reasoning ("thinking") tokens are surfaced in
`AIMessageChunk.additional_kwargs["reasoning_content"]` for every provider,
so the CLI can show progress while the model thinks.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.outputs import ChatGenerationChunk
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI


class ProviderError(RuntimeError):
    """Raised when a provider is unknown or missing required configuration."""


class ChatOpenAICompatible(ChatOpenAI):
    """ChatOpenAI for OpenAI-compatible servers (llama.cpp, vLLM, ...).

    ChatOpenAI deliberately ignores non-standard fields, but llama.cpp streams
    the model's thinking in `delta.reasoning_content` (or `delta.reasoning`).
    We copy it into additional_kwargs, matching how ChatOllama reports it.
    """

    def _convert_chunk_to_generation_chunk(self, chunk: dict, default_chunk_class: type,
                                           base_generation_info: dict | None) -> ChatGenerationChunk | None:
        generation = super()._convert_chunk_to_generation_chunk(chunk, default_chunk_class, base_generation_info)
        choices = chunk.get("choices") or []
        if generation is not None and choices:
            delta = choices[0].get("delta") or {}
            reasoning = delta.get("reasoning_content") or delta.get("reasoning")
            if reasoning:
                generation.message.additional_kwargs["reasoning_content"] = reasoning
        return generation


def _build_ollama(model: str) -> BaseChatModel:
    return ChatOllama(
        model=model,
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        # Ollama's default context window is too small for multi-step tool use.
        num_ctx=int(os.getenv("OLLAMA_NUM_CTX", "16384")),
        temperature=0.2,
        # Stream "thinking" separately from the answer (for models that think).
        reasoning=os.getenv("OLLAMA_THINK", "true").lower() == "true",
    )


def _build_llamacpp(model: str) -> BaseChatModel:
    return ChatOpenAICompatible(
        model=model,
        base_url=os.environ["LLAMACPP_BASE_URL"],
        api_key=os.getenv("LLAMACPP_API_KEY") or "not-needed",
        temperature=0.2,
        stream_usage=True,
        timeout=300,
        max_retries=2,
    )


@dataclass(frozen=True)
class ProviderSpec:
    name: str
    label: str
    model_env: str                 # env var that overrides the default model
    default_model: str
    required_env: tuple[str, ...]  # env vars that must be set to use this provider
    build: Callable[[str], BaseChatModel]

    @property
    def configured_model(self) -> str:
        return os.getenv(self.model_env) or self.default_model

    def missing_env(self) -> list[str]:
        return [var for var in self.required_env if not os.getenv(var)]


PROVIDERS: dict[str, ProviderSpec] = {
    "ollama": ProviderSpec(
        name="ollama", label="Ollama (local)",
        model_env="OLLAMA_MODEL", default_model="gemma4:e2b",
        required_env=(), build=_build_ollama,
    ),
    "llamacpp": ProviderSpec(
        name="llamacpp", label="llama.cpp server (hosted, OpenAI-compatible)",
        model_env="LLAMACPP_MODEL", default_model="qwen3.8-27b",
        required_env=("LLAMACPP_BASE_URL",), build=_build_llamacpp,
    ),
}


def parse_model_ref(ref: str) -> tuple[str, str | None]:
    """'ollama' -> ('ollama', None); 'ollama:qwen3:8b' -> ('ollama', 'qwen3:8b')."""
    provider, _, model = ref.partition(":")
    return provider.strip().lower(), (model.strip() or None)


def create_chat_model(provider: str, model: str | None = None) -> tuple[BaseChatModel, str]:
    """Build a chat model. Returns (model, resolved model name)."""
    spec = PROVIDERS.get(provider)
    if spec is None:
        raise ProviderError(f"Unknown provider '{provider}'. Choose from: {', '.join(PROVIDERS)}")
    if missing := spec.missing_env():
        raise ProviderError(f"Provider '{provider}' needs {', '.join(missing)} set in .env")
    resolved = model or spec.configured_model
    return spec.build(resolved), resolved
