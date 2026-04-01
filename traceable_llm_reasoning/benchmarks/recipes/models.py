from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any, Mapping

from traceable_llm_reasoning.reasoning.types import TaskSpec


def normalize_text(value: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() or ch.isspace() else " " for ch in value)
    return " ".join(cleaned.split())


def tokenize(value: str) -> set[str]:
    return set(normalize_text(value).split())


@dataclass(frozen=True)
class Ingredient:
    name: str
    quantity: float | int | None = None
    unit: str | None = None
    roles: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    notes: str | None = None

    @property
    def normalized_name(self) -> str:
        return normalize_text(self.name)

    def tokens(self) -> set[str]:
        tokens = tokenize(self.name)
        tokens.update(normalize_text(tag) for tag in self.tags)
        tokens.update(normalize_text(role) for role in self.roles)
        return tokens

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def with_updates(self, **changes: Any) -> "Ingredient":
        return replace(self, **changes)


@dataclass(frozen=True)
class StepAction:
    verb: str
    targets: tuple[str, ...] = ()
    parameters: dict[str, Any] = field(default_factory=dict)

    def normalized_targets(self) -> tuple[str, ...]:
        return tuple(normalize_text(target) for target in self.targets)

    def to_dict(self) -> dict[str, Any]:
        return {"verb": self.verb, "targets": list(self.targets), "parameters": dict(self.parameters)}

    def with_updates(self, **changes: Any) -> "StepAction":
        return replace(self, **changes)


@dataclass(frozen=True)
class WorkflowStep:
    step_id: str
    text: str
    actions: tuple[StepAction, ...] = ()
    ingredient_refs: tuple[str, ...] = ()
    depends_on: tuple[str, ...] = ()
    parameters: dict[str, Any] = field(default_factory=dict)

    def tokens(self) -> set[str]:
        tokens = tokenize(self.text)
        for action in self.actions:
            tokens.add(normalize_text(action.verb))
            tokens.update(action.normalized_targets())
        tokens.update(normalize_text(item) for item in self.ingredient_refs)
        return tokens

    def referenced_ingredients(self) -> tuple[str, ...]:
        seen: list[str] = []
        for ingredient in self.ingredient_refs:
            if ingredient not in seen:
                seen.append(ingredient)
        for action in self.actions:
            for target in action.targets:
                if target not in seen:
                    seen.append(target)
        return tuple(seen)

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "text": self.text,
            "actions": [action.to_dict() for action in self.actions],
            "ingredient_refs": list(self.ingredient_refs),
            "depends_on": list(self.depends_on),
            "parameters": dict(self.parameters),
        }

    def with_updates(self, **changes: Any) -> "WorkflowStep":
        return replace(self, **changes)


@dataclass(frozen=True)
class RecipeCase:
    case_id: str
    title: str
    category: str
    ingredients: tuple[Ingredient, ...]
    steps: tuple[WorkflowStep, ...]
    tags: tuple[str, ...] = ()
    dietary_labels: tuple[str, ...] = ()
    summary: str | None = None
    default_servings: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def ingredient_names(self) -> set[str]:
        return {ingredient.normalized_name for ingredient in self.ingredients}

    def ingredient_tokens(self) -> set[str]:
        tokens: set[str] = set()
        for ingredient in self.ingredients:
            tokens.update(ingredient.tokens())
        return tokens

    def step_tokens(self) -> set[str]:
        tokens: set[str] = set()
        for step in self.steps:
            tokens.update(step.tokens())
        return tokens

    def all_tokens(self) -> set[str]:
        tokens = tokenize(self.title) | tokenize(self.category)
        if self.summary:
            tokens.update(tokenize(self.summary))
        tokens.update(tokenize(" ".join(self.tags)))
        tokens.update(self.ingredient_tokens())
        tokens.update(self.step_tokens())
        return tokens

    def get_step(self, step_id: str) -> WorkflowStep | None:
        for step in self.steps:
            if step.step_id == step_id:
                return step
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "title": self.title,
            "category": self.category,
            "ingredients": [ingredient.to_dict() for ingredient in self.ingredients],
            "steps": [step.to_dict() for step in self.steps],
            "tags": list(self.tags),
            "dietary_labels": list(self.dietary_labels),
            "summary": self.summary,
            "default_servings": self.default_servings,
            "metadata": dict(self.metadata),
        }

    def with_updates(self, **changes: Any) -> "RecipeCase":
        return replace(self, **changes)


@dataclass(frozen=True)
class RecipeCaseBase:
    cases: tuple[RecipeCase, ...]
    source_name: str = "recipes-benchmark"

    def by_id(self, case_id: str) -> RecipeCase | None:
        for case in self.cases:
            if case.case_id == case_id:
                return case
        return None


@dataclass(frozen=True)
class RecipeTaskView:
    task_id: str
    instruction: str
    category: str | None = None
    include_ingredients: tuple[str, ...] = ()
    required_ingredients: tuple[str, ...] = ()
    exclude_ingredients: tuple[str, ...] = ()
    dietary_requirements: tuple[str, ...] = ()
    preferred_tags: tuple[str, ...] = ()
    notes: str | None = None
    source_case_id: str | None = None
    minimal_edit: bool = False
    style_goals: tuple[str, ...] = ()
    impossible: bool = False

    def tokens(self) -> set[str]:
        tokens = tokenize(self.instruction)
        if self.category:
            tokens.update(tokenize(self.category))
        if self.notes:
            tokens.update(tokenize(self.notes))
        for collection in (
            self.include_ingredients,
            self.required_ingredients,
            self.exclude_ingredients,
            self.dietary_requirements,
            self.preferred_tags,
            self.style_goals,
        ):
            for item in collection:
                tokens.update(tokenize(item))
        return tokens

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "instruction": self.instruction,
            "category": self.category,
            "include_ingredients": list(self.include_ingredients),
            "required_ingredients": list(self.required_ingredients),
            "exclude_ingredients": list(self.exclude_ingredients),
            "dietary_requirements": list(self.dietary_requirements),
            "preferred_tags": list(self.preferred_tags),
            "notes": self.notes,
            "source_case_id": self.source_case_id,
            "minimal_edit": self.minimal_edit,
            "style_goals": list(self.style_goals),
            "impossible": self.impossible,
        }


def recipe_task_from_task_spec(task: TaskSpec) -> RecipeTaskView:
    constraints = task.constraints
    return RecipeTaskView(
        task_id=task.task_id,
        instruction=task.instruction,
        category=constraints.get("category"),
        include_ingredients=tuple(constraints.get("include_ingredients", ())),
        required_ingredients=tuple(constraints.get("required_ingredients", ())),
        exclude_ingredients=tuple(constraints.get("exclude_ingredients", ())),
        dietary_requirements=tuple(constraints.get("dietary_requirements", ())),
        preferred_tags=tuple(constraints.get("preferred_tags", ())),
        notes=task.metadata.get("notes"),
        source_case_id=task.source_hint,
        minimal_edit=bool(task.metadata.get("minimal_edit", False)),
        style_goals=tuple(task.metadata.get("style_goals", ())),
        impossible=bool(task.metadata.get("impossible", False)),
    )


def _to_ingredient(raw: Mapping[str, Any]) -> Ingredient:
    return Ingredient(
        name=raw["name"],
        quantity=raw.get("quantity"),
        unit=raw.get("unit"),
        roles=tuple(raw.get("roles", ())),
        tags=tuple(raw.get("tags", ())),
        notes=raw.get("notes"),
    )


def _to_action(raw: Mapping[str, Any]) -> StepAction:
    return StepAction(
        verb=raw["verb"],
        targets=tuple(raw.get("targets", ())),
        parameters=dict(raw.get("parameters", {})),
    )


def _to_step(raw: Mapping[str, Any]) -> WorkflowStep:
    actions = tuple(_to_action(action) for action in raw.get("actions", ()))
    ingredient_refs = tuple(raw.get("ingredient_refs", ()))
    if not ingredient_refs:
        refs: list[str] = []
        for action in actions:
            for target in action.targets:
                if target not in refs:
                    refs.append(target)
        ingredient_refs = tuple(refs)
    return WorkflowStep(
        step_id=raw["step_id"],
        text=raw["text"],
        actions=actions,
        ingredient_refs=ingredient_refs,
        depends_on=tuple(raw.get("depends_on", ())),
        parameters=dict(raw.get("parameters", {})),
    )


def recipe_case_from_dict(raw: Mapping[str, Any]) -> RecipeCase:
    return RecipeCase(
        case_id=raw["case_id"],
        title=raw["title"],
        category=raw["category"],
        ingredients=tuple(_to_ingredient(ingredient) for ingredient in raw.get("ingredients", ())),
        steps=tuple(_to_step(step) for step in raw.get("steps", ())),
        tags=tuple(raw.get("tags", ())),
        dietary_labels=tuple(raw.get("dietary_labels", ())),
        summary=raw.get("summary"),
        default_servings=raw.get("default_servings"),
        metadata=dict(raw.get("metadata", {})),
    )
