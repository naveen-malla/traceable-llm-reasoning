from __future__ import annotations

import json
import urllib.error
import urllib.request

from traceable_llm_reasoning.providers.mock import RuleBasedProvider
from traceable_llm_reasoning.providers.prompting import build_generation_prompt, build_retrieval_rerank_prompt, build_substitution_prompt
from traceable_llm_reasoning.reasoning.types import RetrievedCandidate, TaskSpec


class OllamaReasoningProvider(RuleBasedProvider):
    name = "ollama"

    def __init__(self, host: str, model: str, timeout_s: int = 30) -> None:
        super().__init__()
        self.host = host.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s

    def _generate_json(self, prompt: str) -> dict[str, object] | None:
        body = json.dumps({"model": self.model, "prompt": prompt, "stream": False}).encode("utf-8")
        request = urllib.request.Request(
            f"{self.host}/api/generate",
            data=body,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, json.JSONDecodeError):
            return None
        text = payload.get("response", "")
        if not isinstance(text, str):
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None

    def rerank_retrieval(self, task_spec: TaskSpec, casebase, candidates: list[RetrievedCandidate]) -> dict[str, float]:
        self.call_count += 1
        payload = self._generate_json(build_retrieval_rerank_prompt(task_spec, candidates))
        if not payload or not isinstance(payload.get("scores"), dict):
            return super().rerank_retrieval(task_spec, casebase, candidates)
        scores = payload["scores"]
        return {
            item_id: float(score)
            for item_id, score in scores.items()
            if isinstance(item_id, str) and isinstance(score, (float, int))
        } or super().rerank_retrieval(task_spec, casebase, candidates)

    def suggest_substitutions(self, ingredient_name: str, task_spec: TaskSpec, recipe, limit: int = 3) -> list[str]:
        self.call_count += 1
        payload = self._generate_json(build_substitution_prompt(ingredient_name, task_spec, recipe))
        if not payload or not isinstance(payload.get("substitutions"), list):
            return super().suggest_substitutions(ingredient_name, task_spec, recipe, limit=limit)
        suggestions = [item for item in payload["substitutions"] if isinstance(item, str)]
        return suggestions[:limit] or super().suggest_substitutions(ingredient_name, task_spec, recipe, limit=limit)

    def generate_recipe(self, task_spec: TaskSpec, source_recipe=None, retrieved_cases=None):
        self.call_count += 1
        payload = self._generate_json(build_generation_prompt(task_spec, source_recipe=source_recipe, retrieved_cases=retrieved_cases))
        if not payload:
            return super().generate_recipe(task_spec, source_recipe=source_recipe, retrieved_cases=retrieved_cases)
        try:
            ingredients = tuple(payload["ingredients"])
            steps = tuple(payload["steps"])
        except (KeyError, TypeError):
            return super().generate_recipe(task_spec, source_recipe=source_recipe, retrieved_cases=retrieved_cases)
        # The remote path is optional; if it does not return a full recipe object, fall back cleanly.
        if not ingredients or not steps:
            return super().generate_recipe(task_spec, source_recipe=source_recipe, retrieved_cases=retrieved_cases)
        return super().generate_recipe(task_spec, source_recipe=source_recipe, retrieved_cases=retrieved_cases)
