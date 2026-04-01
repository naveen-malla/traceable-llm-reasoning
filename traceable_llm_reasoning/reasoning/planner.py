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


def _fallback_plan(task_spec: TaskSpec, mismatches: tuple[Mismatch, ...]) -> ReasoningPlan:
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
    return ReasoningPlan(
        summary=f"Resolve {len(mismatches)} mismatch(es) while preserving the original dish structure.",
        steps=steps,
        target_edits=target_edits,
        risks=tuple(unique_risks),
    )


def _coerce_plan(payload: dict[str, object] | None, fallback: ReasoningPlan) -> ReasoningPlan:
    if not payload:
        return fallback
    summary = payload.get("summary")
    raw_steps = payload.get("steps")
    raw_target_edits = payload.get("target_edits", ())
    raw_risks = payload.get("risks", ())
    if not isinstance(summary, str) or not isinstance(raw_steps, list):
        return fallback
    steps: list[PlanStep] = []
    for item in raw_steps:
        if not isinstance(item, dict):
            continue
        title = item.get("title")
        purpose = item.get("purpose")
        expected_check = item.get("expected_check")
        risk = item.get("risk", "")
        if isinstance(title, str) and isinstance(purpose, str) and isinstance(expected_check, str) and isinstance(risk, str):
            steps.append(PlanStep(title=title, purpose=purpose, expected_check=expected_check, risk=risk))
    if not steps:
        return fallback
    target_edits = tuple(item for item in raw_target_edits if isinstance(item, str)) or tuple(step.title for step in steps)
    risks = tuple(item for item in raw_risks if isinstance(item, str))
    return ReasoningPlan(summary=summary, steps=tuple(steps), target_edits=target_edits, risks=risks)


def build_reasoning_plan(task_spec: TaskSpec, recipe, mismatches: tuple[Mismatch, ...], provider) -> tuple[ReasoningPlan, ModelCall]:
    fallback = _fallback_plan(task_spec, mismatches)
    payload = provider.plan_reasoning(task_spec, recipe, mismatches)
    plan = _coerce_plan(payload if isinstance(payload, dict) else None, fallback)
    call = ModelCall(
        provider=provider.name,
        operation="plan",
        prompt_summary=f"task={task_spec.task_id}; mismatches={len(mismatches)}",
        response_summary=plan.summary,
        metadata={"target_edits": list(plan.target_edits)},
    )
    return plan, call
