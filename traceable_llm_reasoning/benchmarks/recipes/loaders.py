from __future__ import annotations

import json
from importlib import resources
from typing import Any

from traceable_llm_reasoning.reasoning.types import TaskSpec

from .models import RecipeCase, RecipeCaseBase, recipe_case_from_dict


def _fixture_text(name: str) -> str:
    package = resources.files(__package__) / "fixtures" / name
    return package.read_text(encoding="utf-8")


def load_recipe_cases(name: str = "cases.json") -> list[RecipeCase]:
    raw = json.loads(_fixture_text(name))
    return [recipe_case_from_dict(item) for item in raw["cases"]]


def load_recipe_case_base(name: str = "cases.json") -> RecipeCaseBase:
    return RecipeCaseBase(tuple(load_recipe_cases(name)), source_name=name)


def _task_from_legacy_query(item: dict[str, Any]) -> TaskSpec:
    return TaskSpec(
        task_id=item.get("query_id", "task-000"),
        domain="recipes",
        instruction=item.get("title") or item.get("instruction") or "Unnamed recipe task",
        constraints={
            "category": item.get("category"),
            "include_ingredients": tuple(item.get("include_ingredients", ())),
            "required_ingredients": tuple(item.get("required_ingredients", ())),
            "exclude_ingredients": tuple(item.get("exclude_ingredients", ())),
            "dietary_requirements": tuple(item.get("dietary_requirements", ())),
            "preferred_tags": tuple(item.get("preferred_tags", ())),
        },
        source_hint=item.get("source_case_id"),
        metadata={
            "notes": item.get("notes"),
            "minimal_edit": item.get("minimal_edit", False),
            "style_goals": tuple(item.get("style_goals", ())),
            "impossible": item.get("impossible", False),
            "expected_outcome": item.get("expected_outcome"),
        },
    )


def load_task_specs(name: str = "tasks.json") -> list[TaskSpec]:
    raw = json.loads(_fixture_text(name))
    if "tasks" in raw:
        tasks = raw["tasks"]
    else:
        tasks = raw.get("queries", [])
    result: list[TaskSpec] = []
    for item in tasks:
        if "task_id" in item and "instruction" in item:
            result.append(
                TaskSpec(
                    task_id=item["task_id"],
                    domain=item.get("domain", "recipes"),
                    instruction=item["instruction"],
                    constraints=dict(item.get("constraints", {})),
                    source_hint=item.get("source_hint"),
                    metadata=dict(item.get("metadata", {})),
                )
            )
        else:
            result.append(_task_from_legacy_query(item))
    return result
