from __future__ import annotations

import argparse
import json

from traceable_llm_reasoning.benchmarks.recipes.benchmark import run_benchmark
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


def _task_by_id(task_id: str):
    for task in load_task_specs():
        if task.task_id == task_id:
            return task
    raise SystemExit(f"Unknown task id: {task_id}")


def _run_single(system_name: str, task_id: str, provider_mode: str, top_k: int) -> dict:
    casebase = load_recipe_case_base()
    task_spec = _task_by_id(task_id)
    provider = build_provider(provider_mode)
    runner = SYSTEM_RUNNERS[system_name]
    if system_name in {"llm_plan_then_execute", "traceable_reasoning"}:
        run = runner(casebase, task_spec, provider, top_k=top_k, search_config=SearchConfig())
    elif system_name == "retrieve_and_generate":
        run = runner(casebase, task_spec, provider, top_k=top_k)
    else:
        run = runner(casebase, task_spec, provider)
    return run.to_dict()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run traceable reasoning benchmarks and demos.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    benchmark_parser = subparsers.add_parser("benchmark", help="Run the benchmark suite.")
    benchmark_parser.add_argument("--config", default="experiments/configs/default.json")

    demo_parser = subparsers.add_parser("demo", help="Run a single demo task.")
    demo_parser.add_argument("--task", default="recipe_pesto_vegan")
    demo_parser.add_argument(
        "--system",
        default="traceable_reasoning",
        choices=["direct_generation", "retrieve_and_generate", "llm_plan_then_execute", "traceable_reasoning"],
    )
    demo_parser.add_argument("--provider", default="mock")
    demo_parser.add_argument("--top-k", type=int, default=3)

    args = parser.parse_args()
    if args.command == "benchmark":
        payload = run_benchmark(args.config)
        print(
            json.dumps(
                {
                    "run_dir": payload["run_dir"],
                    "summary": payload["summary"],
                    "verifier_accuracy": payload["verifier_accuracy"],
                },
                indent=2,
            )
        )
        return
    payload = _run_single(args.system, args.task, args.provider, args.top_k)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
