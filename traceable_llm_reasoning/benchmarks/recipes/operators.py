from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .knowledge import TEXT_ALIASES
from .models import Ingredient, RecipeCase, StepAction, WorkflowStep, normalize_text


@dataclass(frozen=True)
class ActionLog:
    name: str
    arguments: dict[str, Any]
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "arguments": dict(self.arguments), "rationale": self.rationale}

    def describe(self) -> str:
        params = ", ".join(f"{key}={value}" for key, value in self.arguments.items())
        return f"{self.name}({params})"


@dataclass(frozen=True)
class ActionResult:
    success: bool
    recipe: RecipeCase
    log: ActionLog
    notes: tuple[str, ...] = field(default_factory=tuple)


def _replacement_terms(old: str) -> tuple[str, ...]:
    normalized = normalize_text(old)
    return (old, *TEXT_ALIASES.get(normalized, ()))


def _replace_phrase(text: str, old: str, new: str) -> str:
    updated = text
    for term in _replacement_terms(old):
        updated = re.sub(rf"\b{re.escape(term)}\b", new, updated, flags=re.IGNORECASE)
    return " ".join(updated.split())


def _replace_values(values: tuple[str, ...], old: str, new: str) -> tuple[str, ...]:
    old_norm = normalize_text(old)
    updated: list[str] = []
    for value in values:
        updated.append(new if normalize_text(value) == old_norm else value)
    deduped: list[str] = []
    for value in updated:
        if value not in deduped:
            deduped.append(value)
    return tuple(deduped)


def _remove_values(values: tuple[str, ...], old: str) -> tuple[str, ...]:
    old_norm = normalize_text(old)
    return tuple(value for value in values if normalize_text(value) != old_norm)


def _replace_action_targets(actions: tuple[StepAction, ...], old: str, new: str) -> tuple[StepAction, ...]:
    return tuple(action.with_updates(targets=_replace_values(action.targets, old, new)) for action in actions)


def _remove_action_targets(actions: tuple[StepAction, ...], old: str) -> tuple[StepAction, ...]:
    return tuple(action.with_updates(targets=_remove_values(action.targets, old)) for action in actions)


class TraceableAction:
    name: str

    def describe(self) -> str:
        raise NotImplementedError

    def apply(self, recipe: RecipeCase) -> ActionResult:
        raise NotImplementedError


@dataclass(frozen=True)
class SubstituteIngredient(TraceableAction):
    old: str
    new: str
    rationale: str = ""
    name: str = "SubstituteIngredient"

    def describe(self) -> str:
        return f"{self.name}(old={self.old}, new={self.new})"

    def apply(self, recipe: RecipeCase) -> ActionResult:
        log = ActionLog(self.name, {"old": self.old, "new": self.new}, self.rationale)
        if normalize_text(self.old) not in recipe.ingredient_names():
            return ActionResult(False, recipe, log, notes=(f"Ingredient '{self.old}' was not found.",))
        ingredients = tuple(
            ingredient.with_updates(name=self.new, tags=())
            if normalize_text(ingredient.name) == normalize_text(self.old)
            else ingredient
            for ingredient in recipe.ingredients
        )
        steps = tuple(
            step.with_updates(
                text=_replace_phrase(step.text, self.old, self.new),
                actions=_replace_action_targets(step.actions, self.old, self.new),
                ingredient_refs=_replace_values(step.ingredient_refs, self.old, self.new),
            )
            for step in recipe.steps
        )
        return ActionResult(True, recipe.with_updates(ingredients=ingredients, steps=steps), log)


@dataclass(frozen=True)
class RemoveIngredient(TraceableAction):
    old: str
    rationale: str = ""
    name: str = "RemoveIngredient"

    def describe(self) -> str:
        return f"{self.name}(old={self.old})"

    def apply(self, recipe: RecipeCase) -> ActionResult:
        log = ActionLog(self.name, {"old": self.old}, self.rationale)
        if normalize_text(self.old) not in recipe.ingredient_names():
            return ActionResult(False, recipe, log, notes=(f"Ingredient '{self.old}' was not found.",))
        ingredients = tuple(item for item in recipe.ingredients if normalize_text(item.name) != normalize_text(self.old))
        steps = tuple(
            step.with_updates(
                text=_replace_phrase(step.text, self.old, "").strip(),
                actions=_remove_action_targets(step.actions, self.old),
                ingredient_refs=_remove_values(step.ingredient_refs, self.old),
            )
            for step in recipe.steps
        )
        return ActionResult(True, recipe.with_updates(ingredients=ingredients, steps=steps), log)


@dataclass(frozen=True)
class AddIngredient(TraceableAction):
    new: str
    step_ref: str
    rationale: str = ""
    name: str = "AddIngredient"

    def describe(self) -> str:
        return f"{self.name}(new={self.new}, step_ref={self.step_ref})"

    def apply(self, recipe: RecipeCase) -> ActionResult:
        log = ActionLog(self.name, {"new": self.new, "step_ref": self.step_ref}, self.rationale)
        if recipe.get_step(self.step_ref) is None:
            return ActionResult(False, recipe, log, notes=(f"Step '{self.step_ref}' was not found.",))
        ingredients = recipe.ingredients
        if normalize_text(self.new) not in recipe.ingredient_names():
            ingredients = (*recipe.ingredients, Ingredient(name=self.new))
        steps: list[WorkflowStep] = []
        for step in recipe.steps:
            if step.step_id == self.step_ref:
                actions = step.actions
                if actions:
                    first = actions[0]
                    actions = (first.with_updates(targets=(*first.targets, self.new)), *actions[1:])
                steps.append(step.with_updates(text=f"{step.text.rstrip('.')} and add {self.new}.", actions=actions, ingredient_refs=(*step.ingredient_refs, self.new)))
            else:
                steps.append(step)
        return ActionResult(True, recipe.with_updates(ingredients=tuple(ingredients), steps=tuple(steps)), log)


@dataclass(frozen=True)
class ReplaceAction(TraceableAction):
    old_action: str
    new_action: str
    step_id: str | None = None
    rationale: str = ""
    name: str = "ReplaceAction"

    def describe(self) -> str:
        return f"{self.name}(old_action={self.old_action}, new_action={self.new_action}, step_id={self.step_id})"

    def apply(self, recipe: RecipeCase) -> ActionResult:
        log = ActionLog(self.name, {"old_action": self.old_action, "new_action": self.new_action, "step_id": self.step_id}, self.rationale)
        steps: list[WorkflowStep] = []
        matched = False
        for step in recipe.steps:
            if self.step_id and step.step_id != self.step_id:
                steps.append(step)
                continue
            if normalize_text(self.old_action) in {normalize_text(action.verb) for action in step.actions}:
                matched = True
                actions = tuple(
                    action.with_updates(verb=self.new_action) if normalize_text(action.verb) == normalize_text(self.old_action) else action
                    for action in step.actions
                )
                steps.append(step.with_updates(text=_replace_phrase(step.text, self.old_action, self.new_action), actions=actions))
            else:
                steps.append(step)
        if not matched:
            return ActionResult(False, recipe, log, notes=(f"Action '{self.old_action}' was not found.",))
        return ActionResult(True, recipe.with_updates(steps=tuple(steps)), log)


@dataclass(frozen=True)
class AdjustParameter(TraceableAction):
    target: str
    attr: str
    value: str
    rationale: str = ""
    name: str = "AdjustParameter"

    def describe(self) -> str:
        return f"{self.name}(target={self.target}, attr={self.attr}, value={self.value})"

    def apply(self, recipe: RecipeCase) -> ActionResult:
        log = ActionLog(self.name, {"target": self.target, "attr": self.attr, "value": self.value}, self.rationale)
        if recipe.get_step(self.target) is None:
            return ActionResult(False, recipe, log, notes=(f"Step '{self.target}' was not found.",))
        steps = tuple(step.with_updates(parameters={**step.parameters, self.attr: self.value}) if step.step_id == self.target else step for step in recipe.steps)
        return ActionResult(True, recipe.with_updates(steps=steps), log)


@dataclass(frozen=True)
class ReorderSteps(TraceableAction):
    step_a: str
    step_b: str
    rationale: str = ""
    name: str = "ReorderSteps"

    def describe(self) -> str:
        return f"{self.name}(step_a={self.step_a}, step_b={self.step_b})"

    def apply(self, recipe: RecipeCase) -> ActionResult:
        log = ActionLog(self.name, {"step_a": self.step_a, "step_b": self.step_b}, self.rationale)
        step_a = recipe.get_step(self.step_a)
        step_b = recipe.get_step(self.step_b)
        if step_a is None or step_b is None or self.step_a in step_b.depends_on or self.step_b in step_a.depends_on:
            return ActionResult(False, recipe, log, notes=("Step reorder violates dependency safety.",))
        steps = list(recipe.steps)
        index_a = next(index for index, step in enumerate(steps) if step.step_id == self.step_a)
        index_b = next(index for index, step in enumerate(steps) if step.step_id == self.step_b)
        steps[index_a], steps[index_b] = steps[index_b], steps[index_a]
        return ActionResult(True, recipe.with_updates(steps=tuple(steps)), log)


TraceableOperator = SubstituteIngredient | RemoveIngredient | AddIngredient | ReplaceAction | AdjustParameter | ReorderSteps
