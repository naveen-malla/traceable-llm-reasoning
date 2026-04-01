from __future__ import annotations

from dataclasses import dataclass

from traceable_llm_reasoning.reasoning.types import RetrievedCandidate, RetrievedContext, TaskSpec

from .knowledge import ingredient_violations
from .models import Ingredient, RecipeCase, RecipeCaseBase, RecipeTaskView, normalize_text, recipe_task_from_task_spec, tokenize


@dataclass(frozen=True)
class Mismatch:
    kind: str
    subject: str
    detail: str
    severity: int = 1
    constraint: str | None = None
    step_id: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "subject": self.subject,
            "detail": self.detail,
            "severity": self.severity,
            "constraint": self.constraint,
            "step_id": self.step_id,
        }


def _token_overlap_score(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _stage1_score(task: RecipeTaskView, recipe: RecipeCase) -> float:
    title_score = _token_overlap_score(tokenize(task.instruction), tokenize(recipe.title))
    category_score = 1.0 if task.category and normalize_text(task.category) == normalize_text(recipe.category) else 0.0
    preferred_tag_score = _token_overlap_score({normalize_text(tag) for tag in task.preferred_tags}, {normalize_text(tag) for tag in recipe.tags})
    source_bonus = 1.0 if task.source_case_id and task.source_case_id == recipe.case_id else 0.0
    return round((0.45 * title_score) + (0.15 * category_score) + (0.1 * preferred_tag_score) + (0.3 * source_bonus), 4)


def detect_mismatches(task: RecipeTaskView, recipe: RecipeCase) -> tuple[Mismatch, ...]:
    mismatches: list[Mismatch] = []
    if task.category and normalize_text(task.category) != normalize_text(recipe.category):
        mismatches.append(Mismatch("category_mismatch", recipe.category, f"Recipe category '{recipe.category}' differs from requested '{task.category}'."))

    ingredient_names = recipe.ingredient_names()
    for ingredient in recipe.ingredients:
        for violation in ingredient_violations(ingredient, task):
            if violation.startswith("exclude:"):
                mismatches.append(
                    Mismatch("excluded_ingredient", ingredient.name, f"Ingredient '{ingredient.name}' matches excluded term '{violation.split(':', 1)[1]}'.", severity=3)
                )
            else:
                mismatches.append(
                    Mismatch("constraint_violation", ingredient.name, f"Ingredient '{ingredient.name}' violates '{violation}'.", severity=3, constraint=violation)
                )
    for required in task.required_ingredients:
        required_norm = normalize_text(required)
        if not any(required_norm == name or required_norm in name for name in ingredient_names):
            mismatches.append(Mismatch("missing_required_ingredient", required, f"Required ingredient '{required}' is missing.", severity=2))
    return tuple(mismatches)


def build_retrieved_context(task_spec: TaskSpec, casebase: RecipeCaseBase, provider, *, top_k: int = 5) -> RetrievedContext:
    task = recipe_task_from_task_spec(task_spec)
    candidates = [
        RetrievedCandidate(
            item_id=recipe.case_id,
            title=recipe.title,
            score_stage1=_stage1_score(task, recipe),
            score_stage2=0.0,
            rationale=f"Matched category={recipe.category}; tokens_overlap={round(_token_overlap_score(task.tokens(), recipe.all_tokens()), 3)}",
            metadata={"mismatch_count": len(detect_mismatches(task, recipe))},
        )
        for recipe in casebase.cases
    ]
    candidates.sort(key=lambda item: (item.score_stage1, item.item_id), reverse=True)
    candidates = candidates[:top_k]

    used_reranker = False
    if provider is not None:
        reranked_scores = provider.rerank_retrieval(task_spec, casebase, candidates)
        if reranked_scores:
            used_reranker = True
            reranked: list[RetrievedCandidate] = []
            for item in candidates:
                reranked.append(
                    RetrievedCandidate(
                        item_id=item.item_id,
                        title=item.title,
                        score_stage1=item.score_stage1,
                        score_stage2=reranked_scores.get(item.item_id, 0.0),
                        rationale=item.rationale,
                        metadata=item.metadata,
                    )
                )
            candidates = sorted(reranked, key=lambda item: (item.final_score(), item.item_id), reverse=True)

    source_hint_respected = bool(task.source_case_id and candidates and candidates[0].item_id == task.source_case_id)
    return RetrievedContext(
        task_id=task_spec.task_id,
        stage1_query=task_spec.instruction,
        candidates=tuple(candidates),
        used_reranker=used_reranker,
        source_hint_respected=source_hint_respected,
    )


def get_source_case(task_spec: TaskSpec, casebase: RecipeCaseBase, context: RetrievedContext) -> RecipeCase:
    task = recipe_task_from_task_spec(task_spec)
    if task.source_case_id:
        source_case = casebase.by_id(task.source_case_id)
        if source_case is not None:
            return source_case
    top = context.candidates[0]
    source_case = casebase.by_id(top.item_id)
    if source_case is None:
        raise ValueError(f"Missing recipe case '{top.item_id}'.")
    return source_case
