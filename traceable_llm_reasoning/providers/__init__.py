from __future__ import annotations

import os
import urllib.error
import urllib.request

from traceable_llm_reasoning.providers.base import ProviderConfig, ReasoningProvider
from traceable_llm_reasoning.providers.mock import MockReasoningProvider, RuleBasedProvider
from traceable_llm_reasoning.providers.ollama import OllamaReasoningProvider
from traceable_llm_reasoning.providers.openai_compatible import OpenAICompatibleProvider


def _has_ollama(host: str, timeout_s: int) -> bool:
    try:
        with urllib.request.urlopen(f"{host.rstrip('/')}/api/tags", timeout=timeout_s):
            return True
    except urllib.error.URLError:
        return False


def build_provider(mode: str | None = None) -> ReasoningProvider:
    config = ProviderConfig(
        mode=(mode or os.environ.get("TLR_PROVIDER_MODE", "auto")).strip().lower(),
        ollama_host=os.environ.get("TLR_OLLAMA_HOST", "http://localhost:11434"),
        ollama_model=os.environ.get("TLR_OLLAMA_MODEL", "llama3.1:8b"),
        openai_base_url=os.environ.get("TLR_OPENAI_BASE_URL", "").strip(),
        openai_api_key=os.environ.get("TLR_OPENAI_API_KEY", "").strip(),
        openai_model=os.environ.get("TLR_OPENAI_MODEL", "gpt-4o-mini"),
        timeout_s=int(os.environ.get("TLR_PROVIDER_TIMEOUT_S", "30")),
    )

    if config.mode == "mock":
        return MockReasoningProvider()
    if config.mode == "rule-based":
        return RuleBasedProvider()
    if config.mode == "ollama":
        return OllamaReasoningProvider(config.ollama_host, config.ollama_model, timeout_s=config.timeout_s)
    if config.mode == "openai-compatible" and config.openai_base_url and config.openai_api_key:
        return OpenAICompatibleProvider(
            config.openai_base_url,
            config.openai_api_key,
            config.openai_model,
            timeout_s=config.timeout_s,
        )
    if config.mode == "auto":
        if config.openai_base_url and config.openai_api_key:
            return OpenAICompatibleProvider(
                config.openai_base_url,
                config.openai_api_key,
                config.openai_model,
                timeout_s=config.timeout_s,
            )
        if _has_ollama(config.ollama_host, config.timeout_s):
            return OllamaReasoningProvider(config.ollama_host, config.ollama_model, timeout_s=config.timeout_s)
        return MockReasoningProvider()
    return RuleBasedProvider()


__all__ = [
    "MockReasoningProvider",
    "OllamaReasoningProvider",
    "OpenAICompatibleProvider",
    "ProviderConfig",
    "ReasoningProvider",
    "RuleBasedProvider",
    "build_provider",
]
