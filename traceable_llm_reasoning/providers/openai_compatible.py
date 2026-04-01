from __future__ import annotations

import json
import urllib.error
import urllib.request

from traceable_llm_reasoning.benchmarks.recipes.models import recipe_case_from_dict
from traceable_llm_reasoning.providers.mock import RuleBasedProvider
from traceable_llm_reasoning.providers.prompting import (
    build_action_proposal_prompt,
    build_critique_prompt,
    build_generation_prompt,
    build_plan_prompt,
    build_retrieval_rerank_prompt,
    build_substitution_prompt,
)
from traceable_llm_reasoning.reasoning.types import RetrievedCandidate, TaskSpec


class OpenAICompatibleProvider(RuleBasedProvider):
    name = "openai-compatible"

    def __init__(self, base_url: str, api_key: str, model: str, timeout_s: int = 30) -> None:
        super().__init__()
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout_s = timeout_s

    def _chat_json(self, prompt: str) -> dict[str, object] | None:
        body = json.dumps(
            {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "response_format": {"type": "json_object"},
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, json.JSONDecodeError):
            return None
        try:
            content = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            return None
        if not isinstance(content, str):
            return None
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return None

    def rerank_retrieval(self, task_spec: TaskSpec, casebase, candidates: list[RetrievedCandidate]) -> dict[str, float]:
        self.call_count += 1
        payload = self._chat_json(build_retrieval_rerank_prompt(task_spec, candidates))
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
        payload = self._chat_json(build_substitution_prompt(ingredient_name, task_spec, recipe))
        if not payload or not isinstance(payload.get("substitutions"), list):
            return super().suggest_substitutions(ingredient_name, task_spec, recipe, limit=limit)
        suggestions = [item for item in payload["substitutions"] if isinstance(item, str)]
        return suggestions[:limit] or super().suggest_substitutions(ingredient_name, task_spec, recipe, limit=limit)

    def plan_reasoning(self, task_spec: TaskSpec, recipe, mismatches) -> dict[str, object] | None:
        self.call_count += 1
        payload = self._chat_json(build_plan_prompt(task_spec, recipe, mismatches))
        if not payload:
            return super().plan_reasoning(task_spec, recipe, mismatches)
        return payload

    def propose_actions(self, task_spec: TaskSpec, recipe, mismatches, plan=None, limit: int = 3) -> list[dict[str, object]] | None:
        self.call_count += 1
        payload = self._chat_json(build_action_proposal_prompt(task_spec, recipe, mismatches, plan=plan, limit=limit))
        if not payload or not isinstance(payload.get("proposals"), list):
            return super().propose_actions(task_spec, recipe, mismatches, plan=plan, limit=limit)
        proposals = [item for item in payload["proposals"] if isinstance(item, dict)]
        return proposals or super().propose_actions(task_spec, recipe, mismatches, plan=plan, limit=limit)

    def critique_recipe(self, task_spec: TaskSpec, recipe, verification) -> dict[str, object] | None:
        self.call_count += 1
        payload = self._chat_json(build_critique_prompt(task_spec, recipe, verification))
        if not payload:
            return super().critique_recipe(task_spec, recipe, verification)
        return payload

    def generate_recipe(self, task_spec: TaskSpec, source_recipe=None, retrieved_cases=None):
        self.call_count += 1
        payload = self._chat_json(build_generation_prompt(task_spec, source_recipe=source_recipe, retrieved_cases=retrieved_cases))
        if not payload:
            return super().generate_recipe(task_spec, source_recipe=source_recipe, retrieved_cases=retrieved_cases)
        candidate = payload.get("recipe", payload)
        try:
            return recipe_case_from_dict(candidate)
        except (KeyError, TypeError):
            return super().generate_recipe(task_spec, source_recipe=source_recipe, retrieved_cases=retrieved_cases)
