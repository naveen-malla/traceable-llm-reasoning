from __future__ import annotations

import time

from traceable_llm_reasoning.benchmarks.recipes.models import recipe_task_from_task_spec
from traceable_llm_reasoning.benchmarks.recipes.retrieval import build_retrieved_context, detect_mismatches, get_source_case
from traceable_llm_reasoning.benchmarks.recipes.verification import reuse_faithfulness, verify_recipe
from traceable_llm_reasoning.reasoning.critic import critique_result
from traceable_llm_reasoning.reasoning.executor import SearchConfig, adapt_recipe
from traceable_llm_reasoning.reasoning.planner import build_reasoning_plan
from traceable_llm_reasoning.reasoning.proposer import build_operator_proposals
from traceable_llm_reasoning.reasoning.types import (
    ModelCall,
    ReasoningTrace,
    RetrievedContext,
    SystemRun,
    TaskSpec,
    VerificationResult,
)


def _empty_context(task_spec: TaskSpec) -> RetrievedContext:
    return RetrievedContext(
        task_id=task_spec.task_id,
        stage1_query=task_spec.instruction,
        candidates=(),
        used_reranker=False,
        source_hint_respected=False,
    )


def _constraint_pass_rate(verification: VerificationResult) -> float:
    checks = [verification.structural_ok, verification.hard_constraints_ok, verification.dependency_ok]
    return round(sum(1 for item in checks if item) / len(checks), 4)


def _build_run(
    *,
    system_name: str,
    task_spec: TaskSpec,
    trace: ReasoningTrace,
    runtime_ms: float,
    final_recipe,
    source_recipe=None,
) -> SystemRun:
    verification = trace.verification
    minimal_edit_score = reuse_faithfulness(source_recipe, final_recipe) if source_recipe is not None else 0.0
    return SystemRun(
        system_name=system_name,
        task=task_spec,
        success=verification.passed,
        runtime_ms=runtime_ms,
        constraint_pass_rate=_constraint_pass_rate(verification),
        minimal_edit_score=minimal_edit_score,
        model_call_count=len(trace.model_calls),
        trace=trace,
    )


def run_direct_generation(casebase, task_spec: TaskSpec, provider) -> SystemRun:
    task = recipe_task_from_task_spec(task_spec)
    started = time.perf_counter()
    recipe = provider.generate_recipe(task_spec)
    runtime_ms = (time.perf_counter() - started) * 1000
    verification = verify_recipe(recipe, task)
    model_calls = (
        ModelCall(
            provider=provider.name,
            operation="generate",
            prompt_summary=f"task={task_spec.task_id}; no retrieval",
            response_summary=recipe.title,
        ),
    )
    trace = ReasoningTrace(
        task=task_spec,
        retrieved_context=_empty_context(task_spec),
        plan=None,
        proposals=(),
        applied_actions=(),
        rejected_actions=(),
        verification=verification,
        critique=None,
        model_calls=model_calls,
        final_output={"recipe": recipe.to_dict()},
        notes=("No retrieval. No explicit action trace.",),
    )
    return _build_run(system_name="direct_generation", task_spec=task_spec, trace=trace, runtime_ms=runtime_ms, final_recipe=recipe)


def run_retrieve_and_generate(casebase, task_spec: TaskSpec, provider, *, top_k: int = 3) -> SystemRun:
    task = recipe_task_from_task_spec(task_spec)
    started = time.perf_counter()
    retrieved_context = build_retrieved_context(task_spec, casebase, provider, top_k=top_k)
    source_recipe = get_source_case(task_spec, casebase, retrieved_context)
    retrieved_cases = [casebase.by_id(candidate.item_id) for candidate in retrieved_context.candidates]
    recipe = provider.generate_recipe(task_spec, source_recipe=source_recipe, retrieved_cases=[case for case in retrieved_cases if case is not None])
    runtime_ms = (time.perf_counter() - started) * 1000
    verification = verify_recipe(recipe, task)
    model_calls = (
        ModelCall(
            provider=provider.name,
            operation="generate",
            prompt_summary=f"task={task_spec.task_id}; retrieved={len(retrieved_context.candidates)}",
            response_summary=recipe.title,
        ),
    )
    trace = ReasoningTrace(
        task=task_spec,
        retrieved_context=retrieved_context,
        plan=None,
        proposals=(),
        applied_actions=(),
        rejected_actions=(),
        verification=verification,
        critique=None,
        model_calls=model_calls,
        final_output={"recipe": recipe.to_dict()},
        notes=("One-shot generation from retrieved context.",),
    )
    return _build_run(
        system_name="retrieve_and_generate",
        task_spec=task_spec,
        trace=trace,
        runtime_ms=runtime_ms,
        final_recipe=recipe,
        source_recipe=source_recipe,
    )


def _run_reasoned_execution(
    *,
    system_name: str,
    casebase,
    task_spec: TaskSpec,
    provider,
    top_k: int = 3,
    search_config: SearchConfig | None = None,
    critique: bool = False,
) -> SystemRun:
    task = recipe_task_from_task_spec(task_spec)
    search_config = search_config or SearchConfig()
    started = time.perf_counter()
    retrieved_context = build_retrieved_context(task_spec, casebase, provider, top_k=top_k)
    source_recipe = get_source_case(task_spec, casebase, retrieved_context)
    mismatches = detect_mismatches(task, source_recipe)
    plan, plan_call = build_reasoning_plan(task_spec, source_recipe, mismatches, provider)
    proposals, proposal_call = build_operator_proposals(
        task_spec,
        source_recipe,
        mismatches,
        provider,
        plan,
        max_candidates_per_mismatch=search_config.max_candidates_per_mismatch,
    )
    result = adapt_recipe(source_recipe, task_spec, task, provider, config=search_config, seed_proposals=proposals)
    final_recipe = result.best_state.recipe
    final_verification = result.best_state.verification
    critique_result_payload = None
    critique_call = None

    if critique and final_verification is not None:
        critique_result_payload, critique_call = critique_result(task_spec, final_recipe, final_verification, provider)
        if critique_result_payload.repair_proposals and not critique_result_payload.approved:
            repaired = adapt_recipe(
                source_recipe,
                task_spec,
                task,
                provider,
                config=search_config,
                seed_proposals=(*proposals, *critique_result_payload.repair_proposals),
            )
            if repaired.best_state.score < result.best_state.score:
                result = repaired
                final_recipe = repaired.best_state.recipe
                final_verification = repaired.best_state.verification

    runtime_ms = (time.perf_counter() - started) * 1000
    model_calls = [plan_call, proposal_call]
    if critique_call is not None:
        model_calls.append(critique_call)
    trace = ReasoningTrace(
        task=task_spec,
        retrieved_context=retrieved_context,
        plan=plan,
        proposals=proposals,
        applied_actions=tuple(log.to_dict() for log in result.best_state.operator_trace),
        rejected_actions=result.rejected_candidates,
        verification=final_verification,
        critique=critique_result_payload,
        model_calls=tuple(model_calls),
        final_output={"recipe": final_recipe.to_dict()},
        notes=(result.reason,),
    )
    return _build_run(
        system_name=system_name,
        task_spec=task_spec,
        trace=trace,
        runtime_ms=runtime_ms,
        final_recipe=final_recipe,
        source_recipe=source_recipe,
    )


def run_llm_plan_then_execute(casebase, task_spec: TaskSpec, provider, *, top_k: int = 3, search_config: SearchConfig | None = None) -> SystemRun:
    return _run_reasoned_execution(
        system_name="llm_plan_then_execute",
        casebase=casebase,
        task_spec=task_spec,
        provider=provider,
        top_k=top_k,
        search_config=search_config,
        critique=False,
    )


def run_traceable_reasoning(casebase, task_spec: TaskSpec, provider, *, top_k: int = 3, search_config: SearchConfig | None = None) -> SystemRun:
    return _run_reasoned_execution(
        system_name="traceable_reasoning",
        casebase=casebase,
        task_spec=task_spec,
        provider=provider,
        top_k=top_k,
        search_config=search_config,
        critique=True,
    )
