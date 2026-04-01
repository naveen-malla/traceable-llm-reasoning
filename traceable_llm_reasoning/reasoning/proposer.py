from __future__ import annotations

from traceable_llm_reasoning.benchmarks.recipes.models import RecipeCase
from traceable_llm_reasoning.benchmarks.recipes.retrieval import Mismatch
from traceable_llm_reasoning.reasoning.types import ModelCall, OperatorProposal, ReasoningPlan, TaskSpec


def build_operator_proposals(
    task_spec: TaskSpec,
    recipe: RecipeCase,
    mismatches: tuple[Mismatch, ...],
    provider,
    plan: ReasoningPlan,
    *,
    max_candidates_per_mismatch: int = 3,
) -> tuple[tuple[OperatorProposal, ...], ModelCall]:
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
