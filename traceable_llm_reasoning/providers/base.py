from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from traceable_llm_reasoning.reasoning.types import RetrievedCandidate, TaskSpec


@dataclass(frozen=True)
class ProviderConfig:
    mode: str = "auto"
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"
    openai_base_url: str = ""
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    timeout_s: int = 30


class ReasoningProvider(Protocol):
    name: str
    call_count: int

    def rerank_retrieval(
        self,
        task_spec: TaskSpec,
        casebase,
        candidates: list[RetrievedCandidate],
    ) -> dict[str, float]:
        ...

    def suggest_substitutions(self, ingredient_name: str, task_spec: TaskSpec, recipe, limit: int = 3) -> list[str]:
        ...

    def generate_recipe(self, task_spec: TaskSpec, source_recipe=None, retrieved_cases=None):
        ...
