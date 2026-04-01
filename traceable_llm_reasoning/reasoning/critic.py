from __future__ import annotations

from traceable_llm_reasoning.reasoning.types import CritiqueResult, ModelCall, OperatorProposal, TaskSpec, VerificationResult


def critique_result(task_spec: TaskSpec, recipe, verification: VerificationResult, provider_name: str) -> tuple[CritiqueResult, ModelCall]:
    repair_proposals: list[OperatorProposal] = []
    notes: list[str] = []
    if verification.passed:
        notes.append("The candidate passes structural, hard-constraint, and dependency checks.")
    else:
        notes.append("The candidate still fails at least one verifier stage.")
        for issue in verification.hard_constraint_issues:
            if issue.get("kind") == "missing_required_ingredient":
                step_ref = recipe.steps[-1].step_id if recipe.steps else "s1"
                repair_proposals.append(
                    OperatorProposal(
                        operator_name="AddIngredient",
                        arguments={"new": issue["subject"], "step_ref": step_ref},
                        confidence=0.45,
                        rationale=issue["detail"],
                        source_refs=(step_ref,),
                    )
                )
    critique = CritiqueResult(
        approved=verification.passed,
        notes=tuple(notes),
        repair_proposals=tuple(repair_proposals),
    )
    call = ModelCall(
        provider=provider_name,
        operation="critique",
        prompt_summary=f"task={task_spec.task_id}; verification_passed={verification.passed}",
        response_summary="approved" if critique.approved else "needs_repair",
        metadata={"repair_count": len(repair_proposals)},
    )
    return critique, call
