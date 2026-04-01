from __future__ import annotations

from traceable_llm_reasoning.reasoning.types import CritiqueResult, ModelCall, OperatorProposal, TaskSpec, VerificationResult


def _fallback_critique(recipe, verification: VerificationResult) -> CritiqueResult:
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
    return CritiqueResult(
        approved=verification.passed,
        notes=tuple(notes),
        repair_proposals=tuple(repair_proposals),
    )


def _coerce_critique(payload: dict[str, object] | None, fallback: CritiqueResult) -> CritiqueResult:
    if not payload:
        return fallback
    approved = payload.get("approved")
    notes = payload.get("notes")
    repair_payload = payload.get("repair_proposals", ())
    if not isinstance(approved, bool) or not isinstance(notes, list):
        return fallback
    repair_proposals: list[OperatorProposal] = []
    for item in repair_payload:
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
        repair_proposals.append(
            OperatorProposal(
                operator_name=operator_name,
                arguments=dict(arguments),
                confidence=float(confidence),
                rationale=rationale,
                source_refs=tuple(ref for ref in source_refs if isinstance(ref, str)),
            )
        )
    return CritiqueResult(
        approved=approved,
        notes=tuple(note for note in notes if isinstance(note, str)),
        repair_proposals=tuple(repair_proposals),
    )


def critique_result(task_spec: TaskSpec, recipe, verification: VerificationResult, provider) -> tuple[CritiqueResult, ModelCall]:
    fallback = _fallback_critique(recipe, verification)
    payload = provider.critique_recipe(task_spec, recipe, verification)
    critique = _coerce_critique(payload if isinstance(payload, dict) else None, fallback)
    call = ModelCall(
        provider=provider.name,
        operation="critique",
        prompt_summary=f"task={task_spec.task_id}; verification_passed={verification.passed}",
        response_summary="approved" if critique.approved else "needs_repair",
        metadata={"repair_count": len(critique.repair_proposals)},
    )
    return critique, call
