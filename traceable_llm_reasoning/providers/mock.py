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

    def plan_reasoning(self, task_spec: TaskSpec, recipe: RecipeCase, mismatches) -> dict[str, object] | None:
        self.call_count += 1
        steps: list[dict[str, object]] = []
        risks: list[str] = []
        for mismatch in mismatches:
            if mismatch.kind in {"constraint_violation", "excluded_ingredient"}:
                risk = "Replacement may drift away from the original dish identity."
                steps.append(
                    {
                        "title": f"Replace {mismatch.subject}",
                        "purpose": mismatch.detail,
                        "expected_check": "Verify that the replacement clears the relevant dietary or exclusion constraint.",
                        "risk": risk,
                    }
                )
                if risk not in risks:
                    risks.append(risk)
            elif mismatch.kind == "missing_required_ingredient":
                risk = "The required ingredient may be inserted too late in the workflow."
                steps.append(
                    {
                        "title": f"Add {mismatch.subject}",
                        "purpose": mismatch.detail,
                        "expected_check": "Verify that the ingredient list and workflow both reference the new ingredient.",
                        "risk": risk,
                    }
                )
                if risk not in risks:
                    risks.append(risk)
        if not steps:
            steps.append(
                {
                    "title": "Verify source recipe",
                    "purpose": "No explicit mismatches were detected before execution.",
                    "expected_check": "Confirm that the source already satisfies the task.",
                    "risk": "A subtle mismatch may still exist in text or tags.",
                }
            )
            risks.append("A subtle mismatch may still exist in text or tags.")
        return {
            "summary": f"Resolve {len(mismatches)} mismatch(es) while keeping the source recipe recognizable.",
            "steps": steps,
            "target_edits": [step["title"] for step in steps],
            "risks": risks,
        }

    def propose_actions(self, task_spec: TaskSpec, recipe: RecipeCase, mismatches, plan=None, limit: int = 3) -> list[dict[str, object]] | None:
        self.call_count += 1
        proposals: list[dict[str, object]] = []
        for mismatch in mismatches:
            if mismatch.kind in {"constraint_violation", "excluded_ingredient"}:
                candidates = self.suggest_substitutions(mismatch.subject, task_spec, recipe, limit=limit)
                for index, candidate in enumerate(candidates):
                    proposals.append(
                        {
                            "operator_name": "SubstituteIngredient",
                            "arguments": {"old": mismatch.subject, "new": candidate},
                            "confidence": round(max(0.4, 0.9 - (index * 0.1)), 2),
                            "rationale": mismatch.detail,
                            "source_refs": [mismatch.subject],
                        }
                    )
                proposals.append(
                    {
                        "operator_name": "RemoveIngredient",
                        "arguments": {"old": mismatch.subject},
                        "confidence": 0.3,
                        "rationale": f"Fallback removal for {mismatch.subject} when no safe substitution exists.",
                        "source_refs": [mismatch.subject],
                    }
                )
            elif mismatch.kind == "missing_required_ingredient":
                step_ref = recipe.steps[-1].step_id if recipe.steps else "s1"
                proposals.append(
                    {
                        "operator_name": "AddIngredient",
                        "arguments": {"new": mismatch.subject, "step_ref": step_ref},
                        "confidence": 0.55,
                        "rationale": mismatch.detail,
                        "source_refs": [step_ref],
                    }
                )
        return proposals

    def critique_recipe(self, task_spec: TaskSpec, recipe: RecipeCase, verification) -> dict[str, object] | None:
        self.call_count += 1
        repair_proposals: list[dict[str, object]] = []
        notes: list[str] = []
        if verification.passed:
            notes.append("The candidate passes structural, hard-constraint, and dependency checks.")
        else:
            notes.append("The candidate still fails at least one verifier stage.")
            for issue in verification.hard_constraint_issues:
                if issue.get("kind") == "missing_required_ingredient":
                    step_ref = recipe.steps[-1].step_id if recipe.steps else "s1"
                    repair_proposals.append(
                        {
                            "operator_name": "AddIngredient",
                            "arguments": {"new": issue["subject"], "step_ref": step_ref},
                            "confidence": 0.45,
                            "rationale": issue["detail"],
                            "source_refs": [step_ref],
                        }
                    )
        return {
            "approved": verification.passed,
            "notes": notes,
            "repair_proposals": repair_proposals,
        }

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
