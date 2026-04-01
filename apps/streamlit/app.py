from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from traceable_llm_reasoning.benchmarks.recipes.loaders import load_recipe_case_base, load_task_specs
from traceable_llm_reasoning.providers import build_provider
from traceable_llm_reasoning.reasoning.executor import SearchConfig
from traceable_llm_reasoning.reasoning.pipeline import (
    run_direct_generation,
    run_llm_plan_then_execute,
    run_retrieve_and_generate,
    run_traceable_reasoning,
)


SYSTEM_RUNNERS = {
    "direct_generation": run_direct_generation,
    "retrieve_and_generate": run_retrieve_and_generate,
    "llm_plan_then_execute": run_llm_plan_then_execute,
    "traceable_reasoning": run_traceable_reasoning,
}


st.set_page_config(
    page_title="Traceable LLM Reasoning",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def get_casebase():
    return load_recipe_case_base()


@st.cache_resource
def get_tasks():
    return load_task_specs()


def _task_lookup():
    return {task.task_id: task for task in get_tasks()}


def _render_recipe(recipe: dict) -> None:
    st.subheader(recipe["title"])
    st.caption(f"Category: {recipe['category']}")
    ingredients = recipe.get("ingredients", [])
    steps = recipe.get("steps", [])
    col1, col2 = st.columns((1, 1.4))
    with col1:
        st.markdown("**Ingredients**")
        st.table(
            [
                {
                    "ingredient": item["name"],
                    "quantity": item.get("quantity"),
                    "unit": item.get("unit"),
                    "roles": ", ".join(item.get("roles", [])),
                }
                for item in ingredients
            ]
        )
    with col2:
        st.markdown("**Workflow**")
        for step in steps:
            st.markdown(f"**{step['step_id']}**  {step['text']}")


def _run_selected_task(task_id: str, system_name: str, provider_name: str, top_k: int) -> dict:
    casebase = get_casebase()
    task = _task_lookup()[task_id]
    provider = build_provider(provider_name)
    runner = SYSTEM_RUNNERS[system_name]
    if system_name in {"llm_plan_then_execute", "traceable_reasoning"}:
        run = runner(casebase, task, provider, top_k=top_k, search_config=SearchConfig())
    elif system_name == "retrieve_and_generate":
        run = runner(casebase, task, provider, top_k=top_k)
    else:
        run = runner(casebase, task, provider)
    return run.to_dict()


st.title("Traceable LLM Reasoning")
st.write("A trace-first demo for constrained task adaptation. The model does not just produce an answer. It retrieves context, proposes explicit actions, executes them, and verifies the result.")

with st.sidebar:
    st.header("Demo Controls")
    tasks = get_tasks()
    default_task_index = next(index for index, task in enumerate(tasks) if task.task_id == "recipe_pesto_vegan")
    selected_task = st.selectbox("Task", [task.task_id for task in tasks], index=default_task_index)
    selected_system = st.selectbox(
        "System",
        ["traceable_reasoning", "llm_plan_then_execute", "retrieve_and_generate", "direct_generation"],
        index=0,
    )
    selected_provider = st.selectbox("Provider", ["mock", "rule-based", "auto", "ollama", "openai-compatible"], index=0)
    top_k = st.slider("Retrieved candidates", min_value=1, max_value=5, value=3)
    run_clicked = st.button("Run Demo", type="primary", use_container_width=True)

if "last_run" not in st.session_state:
    st.session_state["last_run"] = _run_selected_task("recipe_pesto_vegan", "traceable_reasoning", "mock", 3)

if run_clicked:
    with st.spinner("Running reasoning pipeline..."):
        st.session_state["last_run"] = _run_selected_task(selected_task, selected_system, selected_provider, top_k)

run = st.session_state["last_run"]
trace = run["trace"]
verification = trace["verification"]

metric_cols = st.columns(5)
metric_cols[0].metric("System", run["system_name"])
metric_cols[1].metric("Success", "pass" if run["success"] else "fail")
metric_cols[2].metric("Constraint Pass Rate", f"{run['constraint_pass_rate']:.2f}")
metric_cols[3].metric("Trace Completeness", f"{trace['trace_completeness']:.2f}")
metric_cols[4].metric("Runtime (ms)", f"{run['runtime_ms']:.1f}")

tabs = st.tabs(
    [
        "Task",
        "Retrieved Context",
        "Plan",
        "Proposals",
        "Execution Trace",
        "Verification",
        "Final Recipe",
        "Raw JSON",
    ]
)

with tabs[0]:
    st.json(trace["task"], expanded=True)

with tabs[1]:
    candidates = trace["retrieved_context"]["candidates"]
    st.caption(f"Used reranker: {trace['retrieved_context']['used_reranker']}")
    if candidates:
        st.table(
            [
                {
                    "item_id": item["item_id"],
                    "title": item["title"],
                    "stage1": item["score_stage1"],
                    "stage2": item["score_stage2"],
                    "final": item["final_score"],
                    "mismatches": item["metadata"].get("mismatch_count"),
                }
                for item in candidates
            ]
        )
    else:
        st.info("This system does not use retrieval.")

with tabs[2]:
    if trace["plan"]:
        st.markdown(f"**Summary**  {trace['plan']['summary']}")
        st.table(trace["plan"]["steps"])
    else:
        st.info("This system does not create an explicit plan.")

with tabs[3]:
    if trace["proposals"]:
        st.table(
            [
                {
                    "operator": proposal["operator_name"],
                    "confidence": proposal["confidence"],
                    "arguments": json.dumps(proposal["arguments"]),
                    "rationale": proposal["rationale"],
                }
                for proposal in trace["proposals"]
            ]
        )
    else:
        st.info("No structured action proposals for this system.")

with tabs[4]:
    if trace["applied_actions"]:
        st.markdown("**Applied actions**")
        st.table(trace["applied_actions"])
    else:
        st.info("No explicit action trace for this system.")
    if trace["rejected_actions"]:
        st.markdown("**Rejected actions**")
        st.json(trace["rejected_actions"][:10], expanded=False)

with tabs[5]:
    st.json(verification, expanded=True)
    critique = trace.get("critique")
    if critique:
        st.markdown("**Critique**")
        st.json(critique, expanded=True)

with tabs[6]:
    _render_recipe(trace["final_output"]["recipe"])

with tabs[7]:
    st.json(run, expanded=False)
