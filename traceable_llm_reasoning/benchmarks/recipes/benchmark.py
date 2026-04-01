from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from traceable_llm_reasoning.benchmarks.recipes.loaders import load_recipe_case_base, load_task_specs
from traceable_llm_reasoning.benchmarks.recipes.metrics import summarize_runs, verifier_accuracy
from traceable_llm_reasoning.paths import RUN_OUTPUT_ROOT
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


def load_config(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _select_tasks(all_tasks, task_ids: list[str] | None):
    if not task_ids:
        return [task for task in all_tasks if task.metadata.get("include_in_default", True)]
    wanted = set(task_ids)
    return [task for task in all_tasks if task.task_id in wanted]


def run_benchmark(config_path: str | Path) -> dict:
    config = load_config(config_path)
    casebase = load_recipe_case_base()
    task_ids = config.get("task_ids") or config.get("query_ids")
    tasks = _select_tasks(load_task_specs(), task_ids)
    search_config = SearchConfig(
        beam_width=config.get("search", {}).get("beam_width", 6),
        max_depth=config.get("search", {}).get("max_depth", 6),
        max_expansions=config.get("search", {}).get("max_expansions", 120),
        max_candidates_per_mismatch=config.get("search", {}).get("max_candidates_per_mismatch", 3),
    )
    provider_mode = config.get("provider_mode", "auto")
    top_k = config.get("retrieval", {}).get("top_k", 3)
    systems = config.get(
        "systems",
        ["direct_generation", "retrieve_and_generate", "llm_plan_then_execute", "traceable_reasoning"],
    )

    runs = []
    for task_spec in tasks:
        for system_name in systems:
            provider = build_provider(provider_mode)
            runner = SYSTEM_RUNNERS[system_name]
            if system_name in {"llm_plan_then_execute", "traceable_reasoning"}:
                runs.append(runner(casebase, task_spec, provider, top_k=top_k, search_config=search_config))
            elif system_name == "retrieve_and_generate":
                runs.append(runner(casebase, task_spec, provider, top_k=top_k))
            else:
                runs.append(runner(casebase, task_spec, provider))

    summary = summarize_runs(runs)
    verifier_summary = verifier_accuracy(casebase)
    run_dir = RUN_OUTPUT_ROOT / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "config": config,
        "tasks": [task.to_dict() for task in tasks],
        "runs": [run.to_dict() for run in runs],
        "summary": summary,
        "verifier_accuracy": verifier_summary,
        "run_dir": str(run_dir),
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (run_dir / "tasks.json").write_text(json.dumps(payload["tasks"], indent=2), encoding="utf-8")
    (run_dir / "results.json").write_text(json.dumps(payload["runs"], indent=2), encoding="utf-8")
    (run_dir / "summary.json").write_text(
        json.dumps({"summary": summary, "verifier_accuracy": verifier_summary}, indent=2),
        encoding="utf-8",
    )
    return payload
