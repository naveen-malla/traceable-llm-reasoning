from __future__ import annotations

from typing import Iterable

from .models import Ingredient, RecipeTaskView, normalize_text


CONSTRAINT_PATTERNS: dict[str, tuple[str, ...]] = {
    "vegan": ("chicken", "beef", "pork", "fish", "shrimp", "egg", "cream", "milk", "butter", "cheese", "parmesan", "yogurt", "honey"),
    "vegetarian": ("chicken", "beef", "pork", "fish", "shrimp", "anchovy", "gelatin"),
    "nut free": ("almond", "cashew", "walnut", "peanut", "hazelnut", "pecan", "pistachio", "pine nut", "macadamia"),
    "gluten free": ("wheat", "semolina", "pasta", "spaghetti", "linguine", "fettuccine", "breadcrumbs", "panko", "flour tortilla", "soy sauce"),
}

SAFE_PATTERN_OVERRIDES: dict[str, tuple[str, ...]] = {
    "vegan": ("vegan pesto", "vegan cheese", "nutritional yeast", "oat cream", "coconut milk", "tofu"),
    "nut free": ("sunflower seed pesto", "sunflower seed butter", "seed pesto"),
    "gluten free": ("gluten-free pasta", "chickpea pasta", "rice noodles", "corn tortilla", "tamari"),
}

SUBSTITUTION_CATALOG: dict[str, tuple[str, ...]] = {
    "chicken breast": ("white beans", "chickpeas", "tofu"),
    "chicken": ("white beans", "chickpeas", "tofu"),
    "heavy cream": ("coconut milk", "oat cream"),
    "cream": ("coconut milk", "oat cream"),
    "parmesan": ("nutritional yeast", "vegan parmesan"),
    "cheese": ("nutritional yeast", "vegan cheese"),
    "basil pesto": ("sunflower seed pesto", "herb sauce"),
    "vegan pesto": ("sunflower seed pesto",),
    "pasta": ("chickpea pasta", "gluten-free pasta", "rice noodles"),
    "spaghetti": ("chickpea pasta", "gluten-free spaghetti"),
    "linguine": ("chickpea pasta", "gluten-free pasta"),
    "flour tortilla": ("corn tortilla",),
    "eggs": ("tofu scramble",),
    "egg": ("tofu scramble",),
    "cheddar cheese": ("nutritional yeast", "vegan cheese"),
    "butter": ("olive oil", "vegan butter"),
    "soy sauce": ("tamari", "coconut aminos"),
    "peanut butter": ("sunflower seed butter", "tahini"),
}

ROLE_FALLBACKS: dict[str, tuple[str, ...]] = {
    "protein": ("white beans", "chickpeas", "tofu"),
    "base": ("chickpea pasta", "gluten-free pasta", "rice noodles"),
    "sauce": ("sunflower seed pesto", "oat cream", "coconut milk"),
    "topping": ("nutritional yeast",),
}

TEXT_ALIASES: dict[str, tuple[str, ...]] = {
    "chicken breast": ("chicken",),
    "heavy cream": ("cream",),
    "basil pesto": ("pesto",),
    "parmesan": ("cheese",),
}


def query_constraints(task: RecipeTaskView) -> tuple[str, ...]:
    return tuple(normalize_text(item) for item in task.dietary_requirements)


def matches_constraint(ingredient: Ingredient, constraint: str) -> bool:
    normalized_constraint = normalize_text(constraint)
    name = normalize_text(ingredient.name)
    tags = {normalize_text(tag) for tag in ingredient.tags}
    if normalized_constraint in tags:
        return True
    if any(override in name for override in SAFE_PATTERN_OVERRIDES.get(normalized_constraint, ())):
        return True
    patterns = CONSTRAINT_PATTERNS.get(normalized_constraint, ())
    return not any(pattern in name for pattern in patterns)


def matches_term(ingredient: Ingredient, term: str) -> bool:
    normalized_term = normalize_text(term)
    name = normalize_text(ingredient.name)
    if normalized_term == name or normalized_term in name:
        return True
    if any(normalized_term == normalize_text(tag) or normalized_term in normalize_text(tag) for tag in ingredient.tags):
        return True
    if any(normalized_term == normalize_text(role) or normalized_term in normalize_text(role) for role in ingredient.roles):
        return True
    if normalized_term == "dairy":
        return any(pattern in name for pattern in ("cream", "milk", "cheese", "parmesan", "butter", "yogurt")) and not matches_constraint(ingredient, "vegan")
    if normalized_term == "cream":
        return "heavy cream" in name
    if normalized_term == "cheese":
        return ("cheese" in name or "parmesan" in name) and "nutritional yeast" not in name
    if normalized_term in {"nuts", "nut"}:
        return any(pattern in name for pattern in CONSTRAINT_PATTERNS["nut free"])
    if normalized_term == "pesto with nuts":
        return "pesto" in name and not matches_constraint(ingredient, "nut free")
    if normalized_term == "wheat pasta":
        return "pasta" in name and not matches_constraint(ingredient, "gluten free")
    return False


def ingredient_violations(ingredient: Ingredient, task: RecipeTaskView) -> list[str]:
    violations: list[str] = []
    for requirement in query_constraints(task):
        if not matches_constraint(ingredient, requirement):
            violations.append(requirement)
    for exclusion in task.exclude_ingredients:
        if matches_term(ingredient, exclusion):
            violations.append(f"exclude:{normalize_text(exclusion)}")
    return violations


def allowed_candidate(name: str, task: RecipeTaskView) -> bool:
    ingredient = Ingredient(name=name)
    return not ingredient_violations(ingredient, task)


def substitution_candidates(ingredient: Ingredient, task: RecipeTaskView) -> list[str]:
    normalized_name = normalize_text(ingredient.name)
    candidates: list[str] = []
    for source_name, replacements in SUBSTITUTION_CATALOG.items():
        if source_name == normalized_name or source_name in normalized_name or normalized_name in source_name:
            candidates.extend(replacements)
    if not candidates:
        for role in ingredient.roles:
            candidates.extend(ROLE_FALLBACKS.get(normalize_text(role), ()))
    filtered: list[str] = []
    for candidate in candidates:
        if allowed_candidate(candidate, task) and candidate not in filtered:
            filtered.append(candidate)
    return filtered


def preserved_overlap(source_names: Iterable[str], candidate_names: Iterable[str]) -> float:
    source_set = {normalize_text(name) for name in source_names}
    candidate_set = {normalize_text(name) for name in candidate_names}
    if not source_set:
        return 1.0
    return len(source_set & candidate_set) / len(source_set)
