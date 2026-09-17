"""FlyBrain Snake Benchmark & Evaluation Package.

Exports:
- BenchmarkEvaluator: Comparative multi-episode evaluation engine.
- run_benchmark: High-level comparative paired benchmark runner.
- run_headless_benchmark: Headless benchmark runner alias.
- welch_t_test: Welch's two-sample t-test for unequal variances.
- mann_whitney_u: Mann-Whitney U test with tie correction.
- mean, std_dev, median: Statistical summary helpers.
"""

from src.benchmark.stats import (
    welch_t_test,
    mann_whitney_u,
    mean,
    std_dev,
    median,
)
from src.benchmark.evaluator import (
    BenchmarkEvaluator,
    run_benchmark,
    run_headless_benchmark,
)

__all__ = [
    "BenchmarkEvaluator",
    "run_benchmark",
    "run_headless_benchmark",
    "welch_t_test",
    "mann_whitney_u",
    "mean",
    "std_dev",
    "median",
]
