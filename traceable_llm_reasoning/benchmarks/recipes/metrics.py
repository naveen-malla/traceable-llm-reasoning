from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

from traceable_llm_reasoning.benchmarks.recipes.models import Ingredient
from traceable_llm_reasoning.benchmarks.recipes.verification import verify_recipe
from traceable_llm_reasoning.reasoning.types import TaskSpec

from .models import RecipeCase, RecipeTaskView, recipe_task_from_task_spec


def summarize_runs(runs: Iterable) -> dict[str, dict[str, float]]:
    grouped: dict[str, list] = defaultdict(list)
    for run in runs:
        grouped[run.system_name].append(run)
    summary: dict[str, dict[str, float]] = {}
    for system_name, system_runs in grouped.items():
        count = len(system_runs)
        summary[system_name] = {
            "runs": count,
            "success_rate": round(sum(1 for run in system_runs if run.success) / count, 4),
            "avg_constraint_pass_rate": round(sum(run.constraint_pass_rate for run in system_runs) / count, 4),
            "avg_minimal_edit_score": round(sum(run.minimal_edit_score for run in system_runs) / count, 4),
            "avg_runtime_ms": round(sum(run.runtime_ms for run in system_runs) / count, 3),
            "avg_model_calls": round(sum(run.model_call_count for run in system_runs) / count, 3),
            "avg_trace_completeness": round(sum(run.trace.trace_completeness() for run in system_runs) / count, 4),
        }
    return summary


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    recipe: RecipeCase
    task: RecipeTaskView
    broken: bool


def build_verifier_benchmark(casebase) -> list[BenchmarkCase]:
    reference = casebase.by_id("recipe-003")
    if reference is None:
        return []
    task_spec = TaskSpec(
        task_id="verifier_recipe_pesto_guardrail",
        domain="recipes",
        instruction="Verify a vegan, nut-free, gluten-free pesto pasta recipe.",
        constraints={"category": "pasta", "dietary_requirements": ["vegan", "nut-free", "gluten-free"]},
        metadata={"notes": "Verifier benchmark fixture."},
    )
    task = recipe_task_from_task_spec(task_spec)
    valid_case = BenchmarkCase("valid_reference", reference, task, broken=False)
    broken_constraint = BenchmarkCase(
        "broken_constraint",
        reference.with_updates(
            ingredients=(*reference.ingredients, Ingredient(name="parmesan", quantity=30, unit="g", roles=("topping",), tags=("dairy",))),
        ),
        task,
        broken=True,
    )
    broken_dependency = BenchmarkCase(
        "broken_dependency",
        reference.with_updates(ingredients=tuple(item for item in reference.ingredients if item.name != "lemon")),
        task,
        broken=True,
    )
    broken_structure = BenchmarkCase(
        "broken_structure",
        reference.with_updates(steps=(reference.steps[0], reference.steps[0], *reference.steps[1:])),
        task,
        broken=True,
    )
    return [valid_case, broken_constraint, broken_dependency, broken_structure]


def verifier_accuracy(casebase) -> dict[str, float]:
    benchmark = build_verifier_benchmark(casebase)
    if not benchmark:
        return {"precision": 0.0, "recall": 0.0, "detection_rate": 0.0}
    true_positive = false_positive = false_negative = detected = 0
    broken_total = sum(1 for item in benchmark if item.broken)
    for item in benchmark:
        result = verify_recipe(item.recipe, item.task)
        predicted_broken = not result.passed
        if predicted_broken and item.broken:
            true_positive += 1
        if predicted_broken and not item.broken:
            false_positive += 1
        if not predicted_broken and item.broken:
            false_negative += 1
        if predicted_broken and item.broken:
            detected += 1
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    detection_rate = detected / max(broken_total, 1)
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "detection_rate": round(detection_rate, 4),
    }
