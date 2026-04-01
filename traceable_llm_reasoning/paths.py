from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_ROOT = Path(__file__).resolve().parent
BENCHMARK_ROOT = PACKAGE_ROOT / "benchmarks"
RECIPE_BENCHMARK_ROOT = BENCHMARK_ROOT / "recipes"
FIXTURE_ROOT = RECIPE_BENCHMARK_ROOT / "fixtures"
EXPERIMENT_ROOT = PROJECT_ROOT / "experiments"
RUN_OUTPUT_ROOT = EXPERIMENT_ROOT / "runs"
APP_ROOT = PROJECT_ROOT / "apps"
