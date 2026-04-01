from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass, field
from typing import Any

from traceable_llm_reasoning.benchmarks.recipes.knowledge import substitution_candidates
from traceable_llm_reasoning.benchmarks.recipes.models import Ingredient, RecipeCase, RecipeTaskView, normalize_text
from traceable_llm_reasoning.benchmarks.recipes.operators import (
    ActionLog,
    AddIngredient,
    AdjustParameter,
    RemoveIngredient,
    ReorderSteps,
    ReplaceAction,
    SubstituteIngredient,
    TraceableOperator,
)
from traceable_llm_reasoning.benchmarks.recipes.retrieval import Mismatch, detect_mismatches
from traceable_llm_reasoning.benchmarks.recipes.verification import reuse_faithfulness, verify_recipe
from traceable_llm_reasoning.reasoning.types import OperatorProposal, TaskSpec, VerificationResult


@dataclass(frozen=True)
class SearchConfig:
    beam_width: int = 6
    max_depth: int = 6
    max_expansions: int = 120
    max_candidates_per_mismatch: int = 3


@dataclass(frozen=True)
class SearchState:
    recipe: RecipeCase
    operator_trace: tuple[ActionLog, ...] = field(default_factory=tuple)
    mismatches: tuple[Mismatch, ...] = field(default_factory=tuple)
    verification: VerificationResult | None = None
    score: float = 0.0
    cost: int = 0
    depth: int = 0

    @property
    def is_valid(self) -> bool:
        return bool(self.verification and self.verification.passed)


@dataclass(frozen=True)
class SearchResult:
    best_state: SearchState
    explored_states: int
    pruned_states: int
    frontier_peak: int
    reason: str
    rejected_candidates: tuple[dict[str, Any], ...] = field(default_factory=tuple)


def _source_mismatch_ingredient(recipe: RecipeCase, mismatch: Mismatch) -> Ingredient | None:
    subject_norm = normalize_text(mismatch.subject)
    for ingredient in recipe.ingredients:
        if normalize_text(ingredient.name) == subject_norm:
            return ingredient
    for ingredient in recipe.ingredients:
        if subject_norm in normalize_text(ingredient.name):
            return ingredient
    return None


def _proposal_to_action(proposal: OperatorProposal) -> TraceableOperator | None:
    if proposal.operator_name == "SubstituteIngredient":
        return SubstituteIngredient(
            old=str(proposal.arguments["old"]),
            new=str(proposal.arguments["new"]),
            rationale=proposal.rationale,
        )
    if proposal.operator_name == "RemoveIngredient":
        return RemoveIngredient(old=str(proposal.arguments["old"]), rationale=proposal.rationale)
    if proposal.operator_name == "AddIngredient":
        return AddIngredient(
            new=str(proposal.arguments["new"]),
            step_ref=str(proposal.arguments["step_ref"]),
            rationale=proposal.rationale,
        )
    if proposal.operator_name == "ReplaceAction":
        return ReplaceAction(
            old_action=str(proposal.arguments["old_action"]),
            new_action=str(proposal.arguments["new_action"]),
            step_id=proposal.arguments.get("step_id"),
            rationale=proposal.rationale,
        )
    if proposal.operator_name == "AdjustParameter":
        return AdjustParameter(
            target=str(proposal.arguments["target"]),
            attr=str(proposal.arguments["attr"]),
            value=str(proposal.arguments["value"]),
            rationale=proposal.rationale,
        )
    if proposal.operator_name == "ReorderSteps":
        return ReorderSteps(
            step_a=str(proposal.arguments["step_a"]),
            step_b=str(proposal.arguments["step_b"]),
            rationale=proposal.rationale,
        )
    return None


def _candidate_actions(
    recipe: RecipeCase,
    task_spec: TaskSpec,
    task: RecipeTaskView,
    mismatches: tuple[Mismatch, ...],
    provider,
    max_candidates_per_mismatch: int,
    seed_proposals: tuple[OperatorProposal, ...],
) -> list[TraceableOperator]:
    actions: list[TraceableOperator] = []
    for proposal in seed_proposals:
        action = _proposal_to_action(proposal)
        if action is not None:
            actions.append(action)
    for mismatch in mismatches:
        if mismatch.kind in {"constraint_violation", "excluded_ingredient"}:
            ingredient = _source_mismatch_ingredient(recipe, mismatch)
            if ingredient is None:
                continue
            suggestions = substitution_candidates(ingredient, task)
            provider_suggestions = provider.suggest_substitutions(ingredient.name, task_spec, recipe, limit=max_candidates_per_mismatch)
            for candidate in provider_suggestions:
                if candidate not in suggestions:
                    suggestions.insert(0, candidate)
            for candidate in suggestions[:max_candidates_per_mismatch]:
                actions.append(SubstituteIngredient(old=ingredient.name, new=candidate, rationale=mismatch.detail))
            actions.append(RemoveIngredient(old=ingredient.name, rationale=mismatch.detail))
        elif mismatch.kind == "missing_required_ingredient":
            target_step = recipe.steps[-1].step_id if recipe.steps else "s1"
            actions.append(AddIngredient(new=mismatch.subject, step_ref=target_step, rationale=mismatch.detail))
    deduped: list[TraceableOperator] = []
    seen: set[tuple[str, str, str]] = set()
    for action in actions:
        if isinstance(action, SubstituteIngredient):
            key = (action.name, normalize_text(action.old), normalize_text(action.new))
        elif isinstance(action, RemoveIngredient):
            key = (action.name, normalize_text(action.old), "")
        elif isinstance(action, AddIngredient):
            key = (action.name, normalize_text(action.new), action.step_ref)
        elif isinstance(action, ReplaceAction):
            key = (action.name, normalize_text(action.old_action), normalize_text(action.new_action))
        elif isinstance(action, AdjustParameter):
            key = (action.name, action.target, action.attr)
        else:
            key = (action.name, getattr(action, "step_a", ""), getattr(action, "step_b", ""))
        if key not in seen:
            seen.add(key)
            deduped.append(action)
    return deduped


def _score_state(state: SearchState, source_recipe: RecipeCase) -> float:
    verification = state.verification
    hard_issues = len(verification.hard_constraint_issues) if verification else 0
    structural_issues = len(verification.structural_issues) if verification else 0
    dependency_issues = len(verification.dependency_issues) if verification else 0
    faithfulness = reuse_faithfulness(source_recipe, state.recipe)
    return (
        hard_issues * 100.0
        + structural_issues * 80.0
        + dependency_issues * 60.0
        + len(state.operator_trace) * 2.0
        - faithfulness * 10.0
    )


def _state_key(recipe: RecipeCase) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    ingredient_key = tuple(sorted(normalize_text(ingredient.name) for ingredient in recipe.ingredients))
    step_order = tuple(step.step_id for step in recipe.steps)
    step_refs = tuple("|".join(sorted(normalize_text(ref) for ref in step.referenced_ingredients())) for step in recipe.steps)
    return ingredient_key, step_order, step_refs


def adapt_recipe(
    source_recipe: RecipeCase,
    task_spec: TaskSpec,
    task: RecipeTaskView,
    provider,
    *,
    config: SearchConfig | None = None,
    seed_proposals: tuple[OperatorProposal, ...] = (),
) -> SearchResult:
    config = config or SearchConfig()
    initial_verification = verify_recipe(source_recipe, task)
    initial_mismatches = detect_mismatches(task, source_recipe)
    initial_state = SearchState(
        recipe=source_recipe,
        operator_trace=(),
        mismatches=initial_mismatches,
        verification=initial_verification,
        score=0.0,
        cost=0,
        depth=0,
    )
    initial_state = SearchState(
        recipe=initial_state.recipe,
        operator_trace=initial_state.operator_trace,
        mismatches=initial_state.mismatches,
        verification=initial_state.verification,
        score=_score_state(initial_state, source_recipe),
        cost=initial_state.cost,
        depth=initial_state.depth,
    )

    best_state = initial_state
    frontier: list[tuple[float, int, SearchState]] = []
    counter = itertools.count()
    heapq.heappush(frontier, (initial_state.score, next(counter), initial_state))
    seen = {_state_key(source_recipe)}
    explored_states = 0
    pruned_states = 0
    frontier_peak = 1
    rejected_candidates: list[dict[str, Any]] = []

    while frontier and explored_states < config.max_expansions:
        _, _, current = heapq.heappop(frontier)
        explored_states += 1

        if current.is_valid and current.score <= best_state.score:
            best_state = current
            if not current.mismatches:
                return SearchResult(
                    best_state=current,
                    explored_states=explored_states,
                    pruned_states=pruned_states,
                    frontier_peak=frontier_peak,
                    reason="verified_solution_found",
                    rejected_candidates=tuple(rejected_candidates),
                )

        if current.depth >= config.max_depth:
            continue

        next_states: list[SearchState] = []
        for action in _candidate_actions(current.recipe, task_spec, task, current.mismatches, provider, config.max_candidates_per_mismatch, seed_proposals):
            operation = action.apply(current.recipe)
            if not operation.success:
                rejected_candidates.append({"operator": operation.log.to_dict(), "reason": list(operation.notes)})
                pruned_states += 1
                continue
            key = _state_key(operation.recipe)
            if key in seen:
                rejected_candidates.append({"operator": operation.log.to_dict(), "reason": ["duplicate_state"]})
                pruned_states += 1
                continue
            seen.add(key)
            verification = verify_recipe(operation.recipe, task)
            if not verification.structural_ok or not verification.dependency_ok:
                rejected_candidates.append({"operator": operation.log.to_dict(), "reason": ["workflow_invalid"], "verification": verification.to_dict()})
                pruned_states += 1
                continue
            mismatches = detect_mismatches(task, operation.recipe)
            next_state = SearchState(
                recipe=operation.recipe,
                operator_trace=(*current.operator_trace, operation.log),
                mismatches=mismatches,
                verification=verification,
                score=0.0,
                cost=current.cost + 1,
                depth=current.depth + 1,
            )
            next_states.append(
                SearchState(
                    recipe=next_state.recipe,
                    operator_trace=next_state.operator_trace,
                    mismatches=next_state.mismatches,
                    verification=next_state.verification,
                    score=_score_state(next_state, source_recipe),
                    cost=next_state.cost,
                    depth=next_state.depth,
                )
            )
        next_states.sort(key=lambda state: state.score)
        for state in next_states[: config.beam_width]:
            if state.is_valid and state.score < best_state.score:
                best_state = state
            heapq.heappush(frontier, (state.score, next(counter), state))
        frontier_peak = max(frontier_peak, len(frontier))

    rescue_state = _greedy_rescue(source_recipe, task_spec, task, provider, config.max_candidates_per_mismatch, seed_proposals)
    if rescue_state and rescue_state.is_valid:
        return SearchResult(
            best_state=rescue_state,
            explored_states=explored_states,
            pruned_states=pruned_states,
            frontier_peak=frontier_peak,
            reason="greedy_rescue",
            rejected_candidates=tuple(rejected_candidates),
        )

    reason = "search_budget_exhausted" if explored_states >= config.max_expansions else "frontier_exhausted"
    return SearchResult(
        best_state=best_state,
        explored_states=explored_states,
        pruned_states=pruned_states,
        frontier_peak=frontier_peak,
        reason=reason,
        rejected_candidates=tuple(rejected_candidates),
    )


def _greedy_rescue(
    source_recipe: RecipeCase,
    task_spec: TaskSpec,
    task: RecipeTaskView,
    provider,
    max_candidates_per_mismatch: int,
    seed_proposals: tuple[OperatorProposal, ...],
) -> SearchState | None:
    current_recipe = source_recipe
    trace: tuple[ActionLog, ...] = ()
    for depth in range(8):
        mismatches = tuple(
            mismatch
            for mismatch in detect_mismatches(task, current_recipe)
            if mismatch.kind in {"constraint_violation", "excluded_ingredient", "missing_required_ingredient"}
        )
        verification = verify_recipe(current_recipe, task)
        current_state = SearchState(
            recipe=current_recipe,
            operator_trace=trace,
            mismatches=mismatches,
            verification=verification,
            score=0.0,
            cost=len(trace),
            depth=depth,
        )
        if current_state.is_valid:
            return SearchState(
                recipe=current_state.recipe,
                operator_trace=current_state.operator_trace,
                mismatches=current_state.mismatches,
                verification=current_state.verification,
                score=_score_state(current_state, source_recipe),
                cost=current_state.cost,
                depth=current_state.depth,
            )
        progress = False
        for mismatch in mismatches:
            for action in _candidate_actions(current_recipe, task_spec, task, (mismatch,), provider, max_candidates_per_mismatch, seed_proposals):
                operation = action.apply(current_recipe)
                if not operation.success:
                    continue
                verification = verify_recipe(operation.recipe, task)
                if not verification.structural_ok or not verification.dependency_ok:
                    continue
                current_recipe = operation.recipe
                trace = (*trace, operation.log)
                progress = True
                break
            if progress:
                break
        if not progress:
            break
    return None
