import unittest

from traceable_llm_reasoning.benchmarks.recipes.loaders import load_recipe_case_base, load_task_specs
from traceable_llm_reasoning.benchmarks.recipes.models import recipe_task_from_task_spec
from traceable_llm_reasoning.benchmarks.recipes.retrieval import detect_mismatches
from traceable_llm_reasoning.benchmarks.recipes.verification import verify_recipe
from traceable_llm_reasoning.providers import build_provider
from traceable_llm_reasoning.reasoning.planner import build_reasoning_plan
from traceable_llm_reasoning.reasoning.proposer import build_operator_proposals
from traceable_llm_reasoning.reasoning.critic import critique_result


class PhaseTwoReasoningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.casebase = load_recipe_case_base()
        cls.tasks = {task.task_id: task for task in load_task_specs()}
        cls.provider = build_provider("mock")

    def test_benchmark_fixture_is_expanded(self) -> None:
        all_tasks = list(self.tasks.values())
        self.assertGreaterEqual(len(all_tasks), 20)
        self.assertGreaterEqual(sum(1 for task in all_tasks if task.metadata.get("impossible", False)), 5)

    def test_provider_backed_plan_and_proposals_are_structured(self) -> None:
        task_spec = self.tasks["recipe_pesto_vegan"]
        source_recipe = self.casebase.by_id(task_spec.source_hint)
        self.assertIsNotNone(source_recipe)
        task = recipe_task_from_task_spec(task_spec)
        mismatches = detect_mismatches(task, source_recipe)

        plan, _ = build_reasoning_plan(task_spec, source_recipe, mismatches, self.provider)
        proposals, _ = build_operator_proposals(task_spec, source_recipe, mismatches, self.provider, plan)

        self.assertTrue(plan.steps)
        self.assertTrue(proposals)
        self.assertIn("Replace chicken breast", plan.target_edits)
        self.assertEqual(proposals[0].operator_name, "SubstituteIngredient")

    def test_provider_backed_critique_returns_repair_structure(self) -> None:
        task_spec = self.tasks["recipe_spaghetti_gluten_free_add_protein"]
        recipe = self.casebase.by_id("recipe-004")
        self.assertIsNotNone(recipe)
        task = recipe_task_from_task_spec(task_spec)
        verification = verify_recipe(recipe, task)

        critique, _ = critique_result(task_spec, recipe, verification, self.provider)

        self.assertFalse(critique.approved)
        self.assertTrue(critique.repair_proposals)
        self.assertEqual(critique.repair_proposals[0].operator_name, "AddIngredient")


if __name__ == "__main__":
    unittest.main()
