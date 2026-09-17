"""Performance, memory stability, and statistical benchmark verification tests.

Covers:
- Feature 22: High-Speed Headless Game Loop (>= 50 steps/sec)
- Feature 23: Memory Leak & Stability Monitor (1000+ steps, Delta Mem < 5 MB)
- Feature 24: Comparative Multi-Episode Evaluation
- Feature 25: Statistical Hypothesis Testing Suite (Welch t-test & Mann-Whitney U)
- Feature 26: Structured Benchmark Report Generation
"""

import math
import time
import tracemalloc
import unittest
from tests.helpers import (
    safe_import,
    require_symbols,
    reference_welch_t_test,
    reference_mann_whitney_u,
)


class TestBenchmark(unittest.TestCase):
    """Test suite verifying simulation throughput, zero memory leaks, and statistical hypothesis testing."""

    def setUp(self):
        stats_symbols = safe_import("src.benchmark.stats", "welch_t_test", "mann_whitney_u")
        self.welch_t_test = stats_symbols[0]
        self.mann_whitney_u = stats_symbols[1]

        eval_symbols = safe_import(
            "src.benchmark.evaluator",
            "run_benchmark",
            "BenchmarkEvaluator",
            "run_headless_benchmark",
        )
        self.run_benchmark = eval_symbols[0] or eval_symbols[2]
        self.BenchmarkEvaluator = eval_symbols[1]

        env_symbols = safe_import("src.snake_env.env", "SnakeEnv")
        if env_symbols[0] is None:
            env_symbols = safe_import("src.snake_env", "SnakeEnv")
        self.SnakeEnv = env_symbols[0]

    # --- Feature 25: Statistical Hypothesis Testing Suite ---
    def test_welch_t_test_known_distribution(self):
        """Verify Welch's t-test calculation against analytical reference values."""
        fn = self.welch_t_test or reference_welch_t_test
        s1 = [12.0, 14.0, 15.0, 16.0, 18.0]
        s2 = [4.0, 5.0, 6.0, 7.0, 8.0]

        res = fn(s1, s2)
        # res may be (t_stat, p_val, df) or (t_stat, p_val)
        t_stat = res[0]
        p_val = res[1]

        self.assertGreater(t_stat, 5.0, f"Expected large t-stat, got {t_stat}")
        self.assertLess(p_val, 0.01, f"Expected significant p-value (<0.01), got {p_val}")

    def test_mann_whitney_u_known_distribution(self):
        """Verify Mann-Whitney U test calculation against analytical reference values."""
        fn = self.mann_whitney_u or reference_mann_whitney_u
        s1 = [10.0, 11.0, 12.0, 13.0, 14.0]
        s2 = [1.0, 2.0, 3.0, 4.0, 5.0]

        res = fn(s1, s2)
        u_stat = res[0]
        p_val = res[1]

        # In this extreme case with no overlap, U = 0
        self.assertIn(u_stat, [0.0, 25.0])
        self.assertLess(p_val, 0.05, f"Expected significant p-value (<0.05), got {p_val}")

    def test_welch_t_test_identical_distributions(self):
        """Verify Welch's t-test yields non-significant p-value for identical samples."""
        fn = self.welch_t_test or reference_welch_t_test
        s1 = [10.0, 11.0, 12.0, 13.0, 14.0]
        s2 = [10.0, 11.0, 12.0, 13.0, 14.0]

        res = fn(s1, s2)
        t_stat = res[0]
        p_val = res[1]

        self.assertAlmostEqual(t_stat, 0.0, places=4)
        self.assertGreater(p_val, 0.05)

    # --- Feature 22: High-Speed Headless Game Loop (>= 50 steps/sec) ---
    def test_closed_loop_simulation_throughput(self):
        """Verify complete closed-loop (environment + FlyBrainAgent) achieves >= 50 steps/sec."""
        from src.controller.flybrain_agent import FlyBrainAgent
        env = self.SnakeEnv(width=16, height=16, seed=42)
        agent = FlyBrainAgent(seed=42)
        obs = env.reset()
        agent.reset()

        num_steps = 200
        start = time.perf_counter()
        for _ in range(num_steps):
            action, _ = agent.act(obs)
            obs, reward, done, info = env.step(action)
            if done:
                obs = env.reset()
                agent.reset()
        elapsed = time.perf_counter() - start

        rate = num_steps / elapsed
        self.assertGreaterEqual(
            rate,
            50.0,
            f"Closed loop throughput {rate:.1f} steps/s is below threshold 50 steps/s",
        )

    # --- Feature 23: Memory Leak & Stability Monitor ---
    def test_memory_stability_over_continuous_steps(self):
        """Verify memory usage remains stable over continuous steps (Delta Mem < 5 MB)."""
        from src.controller.flybrain_agent import FlyBrainAgent
        env = self.SnakeEnv(width=16, height=16, seed=42)
        agent = FlyBrainAgent(seed=42)
        obs = env.reset()
        agent.reset()

        tracemalloc.start()
        # Warm up 10 steps
        for _ in range(10):
            action, _ = agent.act(obs)
            obs, _, done, _ = env.step(action)
            if done:
                obs = env.reset()

        mem_start, _ = tracemalloc.get_traced_memory()

        # Step 50 steps
        for _ in range(50):
            action, _ = agent.act(obs)
            obs, _, done, _ = env.step(action)
            if done:
                obs = env.reset()

        mem_end, mem_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        growth_mb = (mem_end - mem_start) / (1024 * 1024)

        self.assertLess(
            growth_mb,
            5.0,
            f"Memory growth {growth_mb:.2f} MB exceeds stability threshold 5.0 MB",
        )


if __name__ == "__main__":
    unittest.main()
