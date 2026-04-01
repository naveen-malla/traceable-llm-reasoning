from __future__ import annotations

import json

from traceable_llm_reasoning.reasoning.types import RetrievedCandidate, TaskSpec


def _mismatch_lines(mismatches) -> list[str]:
    lines: list[str] = []
    for mismatch in mismatches:
        lines.append(
            json.dumps(
                {
                    "kind": mismatch.kind,
                    "subject": mismatch.subject,
                    "detail": mismatch.detail,
                    "severity": mismatch.severity,
                    "constraint": mismatch.constraint,
                },
                ensure_ascii=True,
            )
        )
    return lines or ['{"kind":"none","subject":"none","detail":"no explicit mismatches"}']


def build_substitution_prompt(ingredient_name: str, task_spec: TaskSpec, recipe) -> str:
    return (
        "You are assisting with a constrained recipe adaptation task.\n"
        "Return strict JSON with the shape {\"substitutions\": [\"candidate 1\", \"candidate 2\"]}.\n"
        f"Task: {task_spec.instruction}\n"
        f"Constraints: {json.dumps(task_spec.constraints, ensure_ascii=True)}\n"
        f"Source recipe: {recipe.title}\n"
        f"Ingredient to replace: {ingredient_name}\n"
        "Prefer substitutions that preserve the role of the ingredient in the dish."
    )


def build_retrieval_rerank_prompt(task_spec: TaskSpec, candidates: list[RetrievedCandidate]) -> str:
    lines = [
        "Score how relevant each candidate recipe is for the task.",
        "Return JSON: {\"scores\": {\"item_id\": 0.0}} with values between 0 and 1.",
        f"Task: {task_spec.instruction}",
        f"Constraints: {json.dumps(task_spec.constraints, ensure_ascii=True)}",
        "Candidates:",
    ]
    for candidate in candidates:
        lines.append(f"- {candidate.item_id}: {candidate.title}")
    return "\n".join(lines)


def build_plan_prompt(task_spec: TaskSpec, recipe, mismatches) -> str:
    lines = [
        "Create a compact reasoning plan for a constrained recipe adaptation task.",
        "Return strict JSON with keys: summary, steps, target_edits, risks.",
        "Each step should contain title, purpose, expected_check, and risk.",
        f"Task: {task_spec.instruction}",
        f"Constraints: {json.dumps(task_spec.constraints, ensure_ascii=True)}",
        f"Source recipe: {recipe.title}",
        "Detected mismatches:",
        *_mismatch_lines(mismatches),
    ]
    return "\n".join(lines)


def build_action_proposal_prompt(task_spec: TaskSpec, recipe, mismatches, plan=None, limit: int = 3) -> str:
    lines = [
        "Propose explicit action primitives for a constrained recipe adaptation task.",
        "Return strict JSON with the shape {\"proposals\": [...]}",
        "Each proposal must contain operator_name, arguments, confidence, rationale, and source_refs.",
        f"Allowed operators: SubstituteIngredient, RemoveIngredient, AddIngredient, ReplaceAction, AdjustParameter, ReorderSteps.",
        f"Maximum proposals per mismatch: {limit}",
        f"Task: {task_spec.instruction}",
        f"Constraints: {json.dumps(task_spec.constraints, ensure_ascii=True)}",
        f"Source recipe: {recipe.title}",
    ]
    if plan is not None:
        lines.append(f"Plan summary: {plan.summary}")
    lines.append("Detected mismatches:")
    lines.extend(_mismatch_lines(mismatches))
    return "\n".join(lines)


def build_critique_prompt(task_spec: TaskSpec, recipe, verification) -> str:
    return "\n".join(
        [
            "Critique a candidate recipe produced for a constrained adaptation task.",
            "Return strict JSON with keys: approved, notes, repair_proposals.",
            "Each repair proposal must contain operator_name, arguments, confidence, rationale, and source_refs.",
            f"Task: {task_spec.instruction}",
            f"Constraints: {json.dumps(task_spec.constraints, ensure_ascii=True)}",
            f"Candidate recipe: {json.dumps(recipe.to_dict(), ensure_ascii=True)}",
            f"Verification: {json.dumps(verification.to_dict(), ensure_ascii=True)}",
        ]
    )


def build_generation_prompt(task_spec: TaskSpec, source_recipe=None, retrieved_cases=None) -> str:
    prompt = [
        "Generate a structured recipe in JSON.",
        "Return an object with keys: title, category, ingredients, steps, tags, dietary_labels, summary.",
        f"Task: {task_spec.instruction}",
        f"Constraints: {json.dumps(task_spec.constraints, ensure_ascii=True)}",
    ]
    if source_recipe is not None:
        prompt.append(f"Start from source recipe '{source_recipe.title}'.")
    if retrieved_cases:
        prompt.append("Retrieved examples: " + ", ".join(recipe.title for recipe in retrieved_cases))
    return "\n".join(prompt)
