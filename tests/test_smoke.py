import unittest

from traceable_llm_reasoning.benchmarks.recipes.loaders import load_recipe_case_base, load_task_specs
from traceable_llm_reasoning.providers import build_provider
from traceable_llm_reasoning.reasoning.executor import SearchConfig
from traceable_llm_reasoning.reasoning.pipeline import (
    run_llm_plan_then_execute,
    run_traceable_reasoning,
)


class SmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.casebase = load_recipe_case_base()
        cls.tasks = {task.task_id: task for task in load_task_specs()}
        cls.search_config = SearchConfig()

    def test_traceable_reasoning_flagship_task_passes(self) -> None:
        run = run_traceable_reasoning(
            self.casebase,
            self.tasks["recipe_pesto_vegan"],
            build_provider("mock"),
            search_config=self.search_config,
        )
        self.assertTrue(run.success)
        ingredient_names = {ingredient["name"] for ingredient in run.trace.final_output["recipe"]["ingredients"]}
        self.assertIn("chickpea pasta", ingredient_names)
        self.assertIn("sunflower seed pesto", ingredient_names)

    def test_plan_then_execute_flagship_task_passes(self) -> None:
        run = run_llm_plan_then_execute(
            self.casebase,
            self.tasks["recipe_pesto_vegan"],
            build_provider("mock"),
            search_config=self.search_config,
        )
        self.assertTrue(run.success)

    def test_impossible_task_fails(self) -> None:
        run = run_traceable_reasoning(
            self.casebase,
            self.tasks["recipe_pesto_impossible"],
            build_provider("mock"),
            search_config=self.search_config,
        )
        self.assertFalse(run.success)


if __name__ == "__main__":
    unittest.main()
