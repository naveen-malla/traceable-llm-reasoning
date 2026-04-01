from __future__ import annotations

from traceable_llm_reasoning.benchmarks.recipes.models import RecipeCase
from traceable_llm_reasoning.benchmarks.recipes.retrieval import Mismatch
from traceable_llm_reasoning.reasoning.types import ModelCall, OperatorProposal, ReasoningPlan, TaskSpec


def _fallback_proposals(
    task_spec: TaskSpec,
    recipe: RecipeCase,
    mismatches: tuple[Mismatch, ...],
    provider,
    *,
    max_candidates_per_mismatch: int = 3,
) -> tuple[OperatorProposal, ...]:
    proposals: list[OperatorProposal] = []
    for mismatch in mismatches:
        if mismatch.kind in {"constraint_violation", "excluded_ingredient"}:
            candidates = provider.suggest_substitutions(mismatch.subject, task_spec, recipe, limit=max_candidates_per_mismatch)
            for index, candidate in enumerate(candidates):
                proposals.append(
                    OperatorProposal(
                        operator_name="SubstituteIngredient",
                        arguments={"old": mismatch.subject, "new": candidate},
                        confidence=round(max(0.4, 0.9 - (index * 0.1)), 2),
                        rationale=mismatch.detail,
                        source_refs=(mismatch.subject,),
                    )
                )
            proposals.append(
                OperatorProposal(
                    operator_name="RemoveIngredient",
                    arguments={"old": mismatch.subject},
                    confidence=0.3,
                    rationale=f"Fallback removal for {mismatch.subject} when no safe substitution exists.",
                    source_refs=(mismatch.subject,),
                )
            )
        elif mismatch.kind == "missing_required_ingredient":
            step_ref = recipe.steps[-1].step_id if recipe.steps else "s1"
            proposals.append(
                OperatorProposal(
                    operator_name="AddIngredient",
                    arguments={"new": mismatch.subject, "step_ref": step_ref},
                    confidence=0.55,
                    rationale=mismatch.detail,
                    source_refs=(step_ref,),
                )
            )
    return tuple(proposals)


def _coerce_proposals(payload: list[dict[str, object]] | None, fallback: tuple[OperatorProposal, ...]) -> tuple[OperatorProposal, ...]:
    if not payload:
        return fallback
    proposals: list[OperatorProposal] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        operator_name = item.get("operator_name")
        arguments = item.get("arguments")
        confidence = item.get("confidence", 0.5)
        rationale = item.get("rationale", "")
        source_refs = item.get("source_refs", ())
        if not isinstance(operator_name, str) or not isinstance(arguments, dict):
            continue
        if not isinstance(confidence, (int, float)) or not isinstance(rationale, str):
            continue
        proposals.append(
            OperatorProposal(
                operator_name=operator_name,
                arguments=dict(arguments),
                confidence=float(confidence),
                rationale=rationale,
                source_refs=tuple(ref for ref in source_refs if isinstance(ref, str)),
            )
        )
    return tuple(proposals) or fallback


def build_operator_proposals(
    task_spec: TaskSpec,
    recipe: RecipeCase,
    mismatches: tuple[Mismatch, ...],
    provider,
    plan: ReasoningPlan,
    *,
    max_candidates_per_mismatch: int = 3,
) -> tuple[tuple[OperatorProposal, ...], ModelCall]:
    fallback = _fallback_proposals(
        task_spec,
        recipe,
        mismatches,
        provider,
        max_candidates_per_mismatch=max_candidates_per_mismatch,
    )
    payload = provider.propose_actions(task_spec, recipe, mismatches, plan=plan, limit=max_candidates_per_mismatch)
    proposals = list(_coerce_proposals(payload if isinstance(payload, list) else None, fallback))
    deduped: list[OperatorProposal] = []
    seen: set[tuple[str, tuple[tuple[str, str], ...]]] = set()
    for proposal in proposals:
        key = (
            proposal.operator_name,
            tuple(sorted((key, str(value)) for key, value in proposal.arguments.items())),
        )
        if key not in seen:
            seen.add(key)
            deduped.append(proposal)
    call = ModelCall(
        provider=provider.name,
        operation="propose_actions",
        prompt_summary=f"task={task_spec.task_id}; plan_steps={len(plan.steps)}",
        response_summary=f"{len(deduped)} structured action proposals",
        metadata={"proposal_count": len(deduped)},
    )
    return tuple(deduped), call
