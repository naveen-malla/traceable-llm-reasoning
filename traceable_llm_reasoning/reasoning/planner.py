from __future__ import annotations

from traceable_llm_reasoning.benchmarks.recipes.retrieval import Mismatch
from traceable_llm_reasoning.reasoning.types import ModelCall, PlanStep, ReasoningPlan, TaskSpec


def _step_for_mismatch(mismatch: Mismatch) -> PlanStep:
    if mismatch.kind in {"constraint_violation", "excluded_ingredient"}:
        return PlanStep(
            title=f"Replace {mismatch.subject}",
            purpose=mismatch.detail,
            expected_check="Verify that the replacement clears the dietary and exclusion constraints.",
            risk="Replacement may drift away from the original dish identity.",
        )
    if mismatch.kind == "missing_required_ingredient":
        return PlanStep(
            title=f"Add {mismatch.subject}",
            purpose=mismatch.detail,
            expected_check="Verify that the required ingredient is reflected in both ingredients and steps.",
            risk="The added ingredient can create an incoherent step order if inserted too late.",
        )
    return PlanStep(
        title=f"Resolve {mismatch.subject}",
        purpose=mismatch.detail,
        expected_check="Re-run structure and dependency checks.",
        risk="Unknown interaction with existing workflow.",
    )


def build_reasoning_plan(task_spec: TaskSpec, mismatches: tuple[Mismatch, ...], provider_name: str) -> tuple[ReasoningPlan, ModelCall]:
    unique_risks: list[str] = []
    steps = tuple(_step_for_mismatch(mismatch) for mismatch in mismatches) or (
        PlanStep(
            title="Verify source recipe",
            purpose="No explicit mismatches were found before execution.",
            expected_check="Confirm that the source already satisfies the task.",
            risk="A silent constraint violation may still be hiding in tags or steps.",
        ),
    )
    for step in steps:
        if step.risk and step.risk not in unique_risks:
            unique_risks.append(step.risk)
    target_edits = tuple(step.title for step in steps)
    summary = f"Resolve {len(mismatches)} mismatch(es) while preserving the original dish structure."
    plan = ReasoningPlan(
        summary=summary,
        steps=steps,
        target_edits=target_edits,
        risks=tuple(unique_risks),
    )
    call = ModelCall(
        provider=provider_name,
        operation="plan",
        prompt_summary=f"task={task_spec.task_id}; mismatches={len(mismatches)}",
        response_summary=summary,
        metadata={"target_edits": list(target_edits)},
    )
    return plan, call
