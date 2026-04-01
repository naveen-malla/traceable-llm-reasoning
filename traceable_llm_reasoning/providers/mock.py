from __future__ import annotations

from traceable_llm_reasoning.benchmarks.recipes.knowledge import allowed_candidate, query_constraints, substitution_candidates
from traceable_llm_reasoning.benchmarks.recipes.models import (
    Ingredient,
    RecipeCase,
    StepAction,
    WorkflowStep,
    normalize_text,
    recipe_task_from_task_spec,
)
from traceable_llm_reasoning.benchmarks.recipes.operators import SubstituteIngredient
from traceable_llm_reasoning.providers.base import ReasoningProvider
from traceable_llm_reasoning.reasoning.types import RetrievedCandidate, TaskSpec


def _find_ingredient(recipe: RecipeCase, name: str) -> Ingredient | None:
    wanted = normalize_text(name)
    for ingredient in recipe.ingredients:
        if normalize_text(ingredient.name) == wanted:
            return ingredient
    for ingredient in recipe.ingredients:
        if wanted in normalize_text(ingredient.name):
            return ingredient
    return None


def _apply_direct_substitutions(recipe: RecipeCase, task_spec: TaskSpec, provider: "RuleBasedProvider", prefix: str) -> RecipeCase:
    updated = recipe
    task = recipe_task_from_task_spec(task_spec)
    for ingredient in list(updated.ingredients):
        if not provider._ingredient_is_compatible(ingredient.name, task_spec):
            suggestions = provider.suggest_substitutions(ingredient.name, task_spec, updated, limit=1)
            if suggestions:
                updated = SubstituteIngredient(ingredient.name, suggestions[0], rationale="direct_generation").apply(updated).recipe
    labels = tuple(dict.fromkeys((*updated.dietary_labels, *query_constraints(task))))
    tags = tuple(dict.fromkeys((*updated.tags, *task.dietary_requirements)))
    return updated.with_updates(
        case_id=f"{prefix}-{task_spec.task_id}",
        title=f"{updated.title} ({', '.join(task.dietary_requirements)})" if task.dietary_requirements else updated.title,
        tags=tags,
        dietary_labels=labels,
    )


def _generic_pasta_recipe(task_spec: TaskSpec) -> RecipeCase:
    task = recipe_task_from_task_spec(task_spec)
    ingredients = (
        Ingredient("pasta", quantity=250, unit="g", roles=("base",), tags=("gluten",)),
        Ingredient("vegan pesto", quantity=4, unit="tbsp", roles=("sauce",), tags=("vegan", "nuts")),
        Ingredient("chickpeas", quantity=1, unit="can", roles=("protein",)),
        Ingredient("oat cream", quantity=120, unit="ml", roles=("sauce",), tags=("vegan",)),
        Ingredient("garlic", quantity=2, unit="cloves", roles=("aromatic",)),
        Ingredient("olive oil", quantity=1, unit="tbsp", roles=("fat",)),
    )
    steps = (
        WorkflowStep("s1", "Cook the pasta in salted water until tender.", (StepAction("boil", ("pasta",)),), ("pasta",)),
        WorkflowStep("s2", "Warm garlic, olive oil, chickpeas, and oat cream until lightly thickened.", (StepAction("saute", ("garlic", "olive oil", "chickpeas", "oat cream")),), ("garlic", "olive oil", "chickpeas", "oat cream")),
        WorkflowStep("s3", "Toss the pasta with vegan pesto and the creamy chickpea sauce.", (StepAction("toss", ("pasta", "vegan pesto", "chickpeas", "oat cream")),), ("pasta", "vegan pesto", "chickpeas", "oat cream")),
    )
    return RecipeCase(
        case_id=f"generated-{task_spec.task_id}",
        title="Generated Weeknight Pesto Pasta",
        category="pasta",
        ingredients=ingredients,
        steps=steps,
        tags=tuple(dict.fromkeys(("generated", *task.dietary_requirements))),
        dietary_labels=tuple(dict.fromkeys(task.dietary_requirements)),
        summary=task.notes or task_spec.instruction,
    )


def _generic_curry_recipe(task_spec: TaskSpec) -> RecipeCase:
    task = recipe_task_from_task_spec(task_spec)
    ingredients = (
        Ingredient("chickpeas", quantity=2, unit="can", roles=("protein",)),
        Ingredient("coconut milk", quantity=400, unit="ml", roles=("sauce",), tags=("vegan", "gluten-free")),
        Ingredient("rice", quantity=250, unit="g", roles=("base",), tags=("gluten-free",)),
        Ingredient("onion", quantity=1, unit="piece", roles=("aromatic",)),
        Ingredient("garlic", quantity=2, unit="cloves", roles=("aromatic",)),
        Ingredient("curry powder", quantity=2, unit="tsp", roles=("seasoning",)),
    )
    steps = (
        WorkflowStep("s1", "Cook the rice separately.", (StepAction("boil", ("rice",)),), ("rice",)),
        WorkflowStep("s2", "Saute onion and garlic, then bloom the curry powder.", (StepAction("saute", ("onion", "garlic", "curry powder")),), ("onion", "garlic", "curry powder")),
        WorkflowStep("s3", "Add chickpeas and coconut milk and simmer until thick.", (StepAction("simmer", ("chickpeas", "coconut milk")),), ("chickpeas", "coconut milk")),
    )
    return RecipeCase(
        case_id=f"generated-{task_spec.task_id}",
        title="Constraint-Aware Coconut Curry",
        category="curry",
        ingredients=ingredients,
        steps=steps,
        tags=tuple(dict.fromkeys(("generated", *task.dietary_requirements))),
        dietary_labels=tuple(dict.fromkeys(task.dietary_requirements)),
        summary=task.notes or task_spec.instruction,
    )


class RuleBasedProvider(ReasoningProvider):
    name = "rule-based"

    def __init__(self) -> None:
        self.call_count = 0

    def _ingredient_is_compatible(self, ingredient_name: str, task_spec: TaskSpec) -> bool:
        return allowed_candidate(ingredient_name, recipe_task_from_task_spec(task_spec))

    def rerank_retrieval(
        self,
        task_spec: TaskSpec,
        casebase,
        candidates: list[RetrievedCandidate],
    ) -> dict[str, float]:
        self.call_count += 1
        scores: dict[str, float] = {}
        for candidate in candidates:
            mismatch_penalty = float(candidate.metadata.get("mismatch_count", 0)) * 0.1
            source_bonus = 0.3 if task_spec.source_hint and task_spec.source_hint == candidate.item_id else 0.0
            scores[candidate.item_id] = round(max(0.0, min(1.0, candidate.score_stage1 + source_bonus - mismatch_penalty)), 4)
        return scores

    def suggest_substitutions(self, ingredient_name: str, task_spec: TaskSpec, recipe: RecipeCase, limit: int = 3) -> list[str]:
        self.call_count += 1
        ingredient = _find_ingredient(recipe, ingredient_name) or Ingredient(ingredient_name)
        return substitution_candidates(ingredient, recipe_task_from_task_spec(task_spec))[:limit]

    def generate_recipe(self, task_spec: TaskSpec, source_recipe: RecipeCase | None = None, retrieved_cases=None) -> RecipeCase:
        self.call_count += 1
        task = recipe_task_from_task_spec(task_spec)
        if source_recipe is not None:
            return _apply_direct_substitutions(source_recipe, task_spec, self, "retrieve-generate")
        if task.category and normalize_text(task.category) == "curry":
            return _generic_curry_recipe(task_spec)
        return _generic_pasta_recipe(task_spec)


class MockReasoningProvider(RuleBasedProvider):
    name = "mock"
