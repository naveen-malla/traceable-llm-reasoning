from __future__ import annotations

from traceable_llm_reasoning.reasoning.types import SemanticCheck, VerificationResult

from .knowledge import ingredient_violations, preserved_overlap
from .models import RecipeCase, RecipeTaskView, normalize_text
from .retrieval import Mismatch


def verify_structure(recipe: RecipeCase) -> tuple[Mismatch, ...]:
    issues: list[Mismatch] = []
    if not recipe.title.strip():
        issues.append(Mismatch(kind="structural", subject="title", detail="Recipe title is empty.", severity=4))
    if not recipe.steps:
        issues.append(Mismatch(kind="structural", subject="steps", detail="Recipe has no workflow steps.", severity=4))
    seen_step_ids: set[str] = set()
    for step in recipe.steps:
        if not step.step_id.strip():
            issues.append(Mismatch(kind="structural", subject="step_id", detail="A workflow step is missing its identifier.", severity=4))
        if step.step_id in seen_step_ids:
            issues.append(Mismatch(kind="structural", subject=step.step_id, detail=f"Duplicate step id '{step.step_id}'.", severity=4))
        seen_step_ids.add(step.step_id)
        if not step.text.strip():
            issues.append(Mismatch(kind="structural", subject=step.step_id, detail="A workflow step has empty text.", severity=4, step_id=step.step_id))
        if not step.actions and not step.ingredient_refs:
            issues.append(
                Mismatch(
                    kind="structural",
                    subject=step.step_id,
                    detail="A workflow step has no ingredient references or explicit actions left after editing.",
                    severity=3,
                    step_id=step.step_id,
                )
            )
    roles = {normalize_text(role) for ingredient in recipe.ingredients for role in ingredient.roles}
    category = normalize_text(recipe.category)
    if category == "pasta":
        if "base" not in roles:
            issues.append(Mismatch(kind="structural", subject="base", detail="A pasta recipe must keep a base ingredient.", severity=3))
        if "sauce" not in roles:
            issues.append(Mismatch(kind="structural", subject="sauce", detail="A pasta recipe must keep a sauce component.", severity=3))
    if category == "curry":
        if "base" not in roles:
            issues.append(Mismatch(kind="structural", subject="base", detail="A curry recipe must keep a serving base.", severity=3))
        if "sauce" not in roles:
            issues.append(Mismatch(kind="structural", subject="sauce", detail="A curry recipe must keep a sauce component.", severity=3))
    return tuple(issues)


def verify_hard_constraints(recipe: RecipeCase, task: RecipeTaskView) -> tuple[Mismatch, ...]:
    issues: list[Mismatch] = []
    for ingredient in recipe.ingredients:
        for violation in ingredient_violations(ingredient, task):
            if violation.startswith("exclude:"):
                issues.append(
                    Mismatch(
                        kind="excluded_ingredient",
                        subject=ingredient.name,
                        detail=f"Ingredient '{ingredient.name}' still matches excluded term '{violation.split(':', 1)[1]}'.",
                        severity=3,
                    )
                )
            else:
                issues.append(
                    Mismatch(
                        kind="constraint_violation",
                        subject=ingredient.name,
                        detail=f"Ingredient '{ingredient.name}' violates '{violation}'.",
                        severity=3,
                        constraint=violation,
                    )
                )
    ingredient_names = recipe.ingredient_names()
    for required in task.required_ingredients:
        required_norm = normalize_text(required)
        if not any(required_norm == name or required_norm in name for name in ingredient_names):
            issues.append(
                Mismatch(
                    kind="missing_required_ingredient",
                    subject=required,
                    detail=f"Required ingredient '{required}' is missing.",
                    severity=2,
                )
            )
    return tuple(issues)


def verify_dependencies(recipe: RecipeCase) -> tuple[Mismatch, ...]:
    issues: list[Mismatch] = []
    step_ids = {step.step_id for step in recipe.steps}
    ingredient_names = recipe.ingredient_names()
    for index, step in enumerate(recipe.steps):
        for dependency in step.depends_on:
            if dependency not in step_ids:
                issues.append(
                    Mismatch(
                        kind="dependency",
                        subject=dependency,
                        detail=f"Step '{step.step_id}' depends on missing step '{dependency}'.",
                        severity=4,
                        step_id=step.step_id,
                    )
                )
                continue
            dep_index = next(i for i, candidate in enumerate(recipe.steps) if candidate.step_id == dependency)
            if dep_index >= index:
                issues.append(
                    Mismatch(
                        kind="dependency",
                        subject=dependency,
                        detail=f"Step '{step.step_id}' depends on step '{dependency}' that appears too late.",
                        severity=4,
                        step_id=step.step_id,
                    )
                )
        for ingredient_name in step.referenced_ingredients():
            if normalize_text(ingredient_name) not in ingredient_names:
                issues.append(
                    Mismatch(
                        kind="dependency",
                        subject=ingredient_name,
                        detail=f"Step '{step.step_id}' references ingredient '{ingredient_name}' that is not present in the ingredient list.",
                        severity=3,
                        step_id=step.step_id,
                    )
                )
    return tuple(issues)


def semantic_review(recipe: RecipeCase, task: RecipeTaskView) -> SemanticCheck:
    notes: list[str] = []
    repair: list[str] = []
    if task.category and normalize_text(task.category) != normalize_text(recipe.category):
        repair.append("Preserve the requested dish category.")
    if task.preferred_tags and not ({normalize_text(tag) for tag in recipe.tags} & {normalize_text(tag) for tag in task.preferred_tags}):
        notes.append("The final recipe does not preserve any preferred tag from the task.")
    return SemanticCheck(passed=not repair, notes=tuple(notes), repair_request=tuple(repair))


def verify_recipe(recipe: RecipeCase, task: RecipeTaskView) -> VerificationResult:
    structural_issues = verify_structure(recipe)
    hard_constraint_issues = verify_hard_constraints(recipe, task)
    dependency_issues = verify_dependencies(recipe)
    semantic_check = semantic_review(recipe, task)
    passed = not structural_issues and not hard_constraint_issues and not dependency_issues and semantic_check.passed
    return VerificationResult(
        passed=passed,
        structural_issues=tuple(issue.to_dict() for issue in structural_issues),
        hard_constraint_issues=tuple(issue.to_dict() for issue in hard_constraint_issues),
        dependency_issues=tuple(issue.to_dict() for issue in dependency_issues),
        semantic_check=semantic_check,
    )


def reuse_faithfulness(source: RecipeCase, candidate: RecipeCase) -> float:
    ingredient_score = preserved_overlap((item.name for item in source.ingredients), (item.name for item in candidate.ingredients))
    source_steps = [step.step_id for step in source.steps]
    candidate_steps = [step.step_id for step in candidate.steps]
    if not source_steps:
        step_score = 1.0
    else:
        step_score = len(set(source_steps) & set(candidate_steps)) / len(source_steps)
    return round((0.6 * ingredient_score) + (0.4 * step_score), 4)
