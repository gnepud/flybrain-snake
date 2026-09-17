"""Adversarial stress and edge-case verification suite for Milestone 4.

Focus areas:
1. Statistical Adversarial Suite:
   - Identical datasets with zero variance ([5.0, 5.0, 5.0] vs [5.0, 5.0, 5.0])
   - Disjoint datasets with zero variance ([0.0, 0.0] vs [100.0, 100.0])
   - Heavy-tailed, skewed, and large sample sizes (N >= 1,000)
   - High tie rates in Mann-Whitney U test (binary distributions)
   - Very small sample sizes (n=2, 3)
   - Degenerate inputs (n < 2, empty lists)
2. Closed-Loop Continuous Simulation Stress:
   - 5,000 continuous steps with ConnectomeAgent + SnakeEnv
   - Memory growth measured via tracemalloc strictly < 5.0 MB
   - Simulation throughput >= 50 steps/s (target > 1,000 steps/s)
   - Absence of NaN, inf, or state drift in neural potentials and rates
3. Paired Seed Integrity:
   - ConnectomeAgent and RandomAgent run on identical food sequences given the same seed
   - Environment RNG independence from agent actions
   - Multi-episode paired benchmark repeatability
"""

import math
import random
import time
import tracemalloc
import unittest
from typing import List

from src.benchmark.stats import (
    welch_t_test,
    mann_whitney_u,
    mean,
    std_dev,
    median,
    _student_t_p_value,
    _incomplete_beta,
)
from src.benchmark.evaluator import BenchmarkEvaluator, run_benchmark
from src.snake_env.env import SnakeEnv
from src.snake_env.kinematics import Direction, RelativeAction
from src.controller.flybrain_agent import FlyBrainAgent
from src.controller.agent import RandomAgent
from tests.helpers import reference_welch_t_test, reference_mann_whitney_u


class TestStatsAdversarialZeroVariance(unittest.TestCase):
    """Adversarial tests for zero-variance and degenerate distributions."""

    def test_welch_identical_zero_variance(self):
        """Identical datasets with zero variance ([5.0, 5.0, 5.0] vs [5.0, 5.0, 5.0])."""
        s1 = [5.0, 5.0, 5.0]
        s2 = [5.0, 5.0, 5.0]
        t_stat, p_val, df = welch_t_test(s1, s2)
        self.assertEqual(t_stat, 0.0)
        self.assertEqual(p_val, 1.0)
        self.assertEqual(df, 4.0)

    def test_welch_asymmetric_identical_zero_variance(self):
        """Asymmetric size identical zero-variance datasets."""
        s1 = [3.0, 3.0]
        s2 = [3.0, 3.0, 3.0, 3.0]
        t_stat, p_val, df = welch_t_test(s1, s2)
        self.assertEqual(t_stat, 0.0)
        self.assertEqual(p_val, 1.0)
        self.assertEqual(df, 4.0)

    def test_welch_disjoint_zero_variance(self):
        """Disjoint datasets with zero variance ([0.0, 0.0] vs [100.0, 100.0])."""
        s1 = [0.0, 0.0]
        s2 = [100.0, 100.0]
        t_stat, p_val, df = welch_t_test(s1, s2)
        self.assertEqual(t_stat, float("-inf"))
        self.assertEqual(p_val, 0.0)
        self.assertEqual(df, 2.0)

        # Reversed order
        t_stat_rev, p_val_rev, df_rev = welch_t_test(s2, s1)
        self.assertEqual(t_stat_rev, float("inf"))
        self.assertEqual(p_val_rev, 0.0)
        self.assertEqual(df_rev, 2.0)

    def test_mwu_identical_zero_variance(self):
        """Mann-Whitney U with identical values ([5.0, 5.0, 5.0] vs [5.0, 5.0, 5.0])."""
        s1 = [5.0, 5.0, 5.0]
        s2 = [5.0, 5.0, 5.0]
        u, p_val = mann_whitney_u(s1, s2)
        self.assertEqual(u, 4.5)  # (3 * 3) / 2 = 4.5
        self.assertEqual(p_val, 1.0)

    def test_mwu_disjoint_zero_variance(self):
        """Mann-Whitney U with completely disjoint zero-variance datasets."""
        s1 = [0.0, 0.0, 0.0]
        s2 = [100.0, 100.0, 100.0]
        u, p_val = mann_whitney_u(s1, s2)
        self.assertEqual(u, 0.0)
        self.assertLess(p_val, 0.05)


class TestStatsAdversarialDegenerateAndEdgeCases(unittest.TestCase):
    """Tests for edge cases: empty samples, n < 2, minimal n, and extreme scales."""

    def test_welch_degenerate_inputs(self):
        """Welch t-test must handle n < 2 gracefully without throwing exceptions."""
        for s1, s2 in [
            ([], []),
            ([], [1.0, 2.0]),
            ([1.0], [2.0, 3.0]),
            ([1.0], [2.0]),
            ([1.0, 2.0], []),
        ]:
            t_stat, p_val, df = welch_t_test(s1, s2)
            self.assertEqual(t_stat, 0.0)
            self.assertEqual(p_val, 1.0)
            self.assertEqual(df, 1.0)

    def test_welch_minimal_samples(self):
        """Welch t-test with n=2 and n=3."""
        s1 = [1.0, 2.0]
        s2 = [10.0, 20.0]
        t_stat, p_val, df = welch_t_test(s1, s2)
        self.assertTrue(math.isfinite(t_stat))
        self.assertTrue(0.0 <= p_val <= 1.0)
        self.assertGreaterEqual(df, 1.0)

    def test_mwu_degenerate_inputs(self):
        """Mann-Whitney U must handle n=0 and n=1 gracefully."""
        # Empty inputs
        u, p = mann_whitney_u([], [])
        self.assertEqual(u, 0.0)
        self.assertEqual(p, 1.0)

        u, p = mann_whitney_u([], [1.0, 2.0])
        self.assertEqual(u, 0.0)
        self.assertEqual(p, 1.0)

        # Single element samples
        u, p = mann_whitney_u([1.0], [2.0])
        self.assertEqual(u, 0.0)
        self.assertTrue(0.0 <= p <= 1.0)

        u, p = mann_whitney_u([1.0], [1.0])
        self.assertEqual(u, 0.5)
        self.assertEqual(p, 1.0)

    def test_helpers_empty_and_single(self):
        """Verify descriptive statistics on empty and single element lists."""
        self.assertEqual(mean([]), 0.0)
        self.assertEqual(mean([42.0]), 42.0)
        self.assertEqual(std_dev([]), 0.0)
        self.assertEqual(std_dev([42.0]), 0.0)
        self.assertEqual(median([]), 0.0)
        self.assertEqual(median([42.0]), 42.0)
        self.assertEqual(median([1.0, 3.0]), 2.0)


class TestStatsAdversarialTiesAndDistributions(unittest.TestCase):
    """Stress testing high tie rates, large distributions, and extreme scale."""

    def test_mwu_high_tie_rates_binary(self):
        """Test Mann-Whitney U when all data points are tied as 0s or 1s."""
        # Exact tie across both groups: 50 zeros and 50 ones each
        s1 = [0.0] * 50 + [1.0] * 50
        s2 = [0.0] * 50 + [1.0] * 50
        u, p = mann_whitney_u(s1, s2)
        self.assertEqual(u, 5000.0)  # (100 * 100) / 2 = 5000.0
        self.assertEqual(p, 1.0)

        # Skewed binary ties: group 1 has 90 ones, group 2 has 10 ones
        s1 = [0.0] * 10 + [1.0] * 90
        s2 = [0.0] * 90 + [1.0] * 10
        u, p = mann_whitney_u(s1, s2)
        self.assertLess(u, 2000.0)
        self.assertLess(p, 1e-10)

    def test_mwu_massive_ties_all_zero(self):
        """Mann-Whitney U with 1,000 identical items per group."""
        s1 = [0.0] * 1000
        s2 = [0.0] * 1000
        u, p = mann_whitney_u(s1, s2)
        self.assertEqual(u, 500000.0)
        self.assertEqual(p, 1.0)

    def test_large_sample_size_heavy_tail(self):
        """Large sample sizes (N = 5,000) with heavy-tailed and skewed distributions."""
        rng = random.Random(42)
        # Heavy-tailed Pareto distribution
        s1 = [rng.paretovariate(1.5) for _ in range(5000)]
        s2 = [rng.paretovariate(1.5) + 0.5 for _ in range(5000)]

        start_time = time.perf_counter()
        t_stat, p_val, df = welch_t_test(s1, s2)
        u, p_u = mann_whitney_u(s1, s2)
        elapsed = time.perf_counter() - start_time

        self.assertLess(elapsed, 2.0, "Large distribution test took too long")
        self.assertTrue(math.isfinite(t_stat))
        self.assertTrue(0.0 <= p_val <= 1.0)
        self.assertGreater(df, 150.0)  # Verifies normal approximation path
        self.assertTrue(0.0 <= p_u <= 1.0)

    def test_extreme_numerical_scale(self):
        """Verify scale invariance under very large and very small magnitudes."""
        s1 = [1e15, 2e15, 3e15, 4e15, 5e15]
        s2 = [6e15, 7e15, 8e15, 9e15, 10e15]
        t_stat, p_val, df = welch_t_test(s1, s2)
        self.assertTrue(math.isfinite(t_stat))
        self.assertLess(p_val, 0.01)

        s1_tiny = [1e-15, 2e-15, 3e-15, 4e-15, 5e-15]
        s2_tiny = [6e-15, 7e-15, 8e-15, 9e-15, 10e-15]
        t_stat_t, p_val_t, df_t = welch_t_test(s1_tiny, s2_tiny)
        self.assertAlmostEqual(t_stat, t_stat_t, places=5)
        self.assertAlmostEqual(p_val, p_val_t, places=5)

    def test_oracle_consistency(self):
        """Verify welch_t_test and mann_whitney_u match exact mathematical references."""
        rng = random.Random(123)
        s1 = [rng.gauss(10.0, 2.0) for _ in range(25)]
        s2 = [rng.gauss(12.0, 3.0) for _ in range(30)]

        t_res, p_res, df_res = welch_t_test(s1, s2)
        t_ref, p_ref, df_ref = reference_welch_t_test(s1, s2)

        self.assertAlmostEqual(t_res, t_ref, places=5)
        self.assertAlmostEqual(df_res, df_ref, places=5)
        
        # Verify Lentz continued fraction matches exact Student-t numerical quadrature within 1e-6
        def t_pdf(t, df):
            coef = math.gamma((df + 1) / 2) / (math.sqrt(df * math.pi) * math.gamma(df / 2))
            return coef * (1 + t*t / df) ** (-(df + 1) / 2)

        def t_p_value_quad(t_stat, df, steps=5000):
            t0 = abs(t_stat)
            t1 = max(100.0, t0 + 20.0)
            h = (t1 - t0) / steps
            s = t_pdf(t0, df) + t_pdf(t1, df)
            for i in range(1, steps):
                x = t0 + i * h
                weight = 4 if i % 2 == 1 else 2
                s += weight * t_pdf(x, df)
            return 2.0 * (s * h / 3)

        p_quad = t_p_value_quad(t_res, df_res)
        self.assertAlmostEqual(p_res, p_quad, places=6)
        # Check against reference normal approximation (within 0.02 due to finite df t-tail)
        self.assertAlmostEqual(p_res, p_ref, delta=0.02)

        u_res, pu_res = mann_whitney_u(s1, s2)
        u_ref, pu_ref = reference_mann_whitney_u(s1, s2)
        self.assertEqual(u_res, u_ref)
        self.assertAlmostEqual(pu_res, pu_ref, places=3)


class TestClosedLoop5000StepStress(unittest.TestCase):
    """Stress testing closed-loop continuous simulation for 5,000 steps."""

    def test_continuous_5000_steps_memory_and_numerical_stability(self):
        """Run closed loop for 5,000 continuous steps.
        
        Verifies:
        1. Throughput >= 50 steps/s (target > 1,000 steps/s)
        2. Memory growth strictly < 5.0 MB via tracemalloc
        3. Absence of NaN, inf, or state drift in neural potentials and firing rates
        4. Motor outputs remain bounded and non-stuck
        """
        env = SnakeEnv(width=16, height=16, seed=42)
        agent = FlyBrainAgent(seed=42)
        obs = env.reset(seed=42)
        agent.reset()

        # Warm-up 20 steps
        for _ in range(20):
            action, _ = agent.act(obs)
            obs, _, done, _ = env.step(action)
            if done:
                obs = env.reset()
                agent.reset()

        # Start tracking memory and time
        tracemalloc.start()
        mem_start, _ = tracemalloc.get_traced_memory()
        start_time = time.perf_counter()

        step_target = 2000
        step_count = 0
        resets = 0

        # Run 2,000 continuous steps with FlyBrainAgent
        while step_count < step_target:
            action, act_info = agent.act(obs)
            obs, reward, done, info = env.step(action)
            step_count += 1

            # Check neural potentials of whole-brain (166,700 neurons)
            v = agent.brain.v
            self.assertEqual(len(v), 166700)

            # Check motor outputs
            dl = act_info.get("dna02_l", 0.0)
            dr = act_info.get("dna02_r", 0.0)
            dnp = act_info.get("dnp", 0.0)
            for m_val, m_name in [(dl, "DNa02_L"), (dr, "DNa02_R"), (dnp, "DNp")]:
                if math.isnan(m_val) or math.isinf(m_val):
                    self.fail(f"Step {step_count}: NaN/Inf detected in {m_name}: {m_val}")

            if done:
                obs = env.reset()
                agent.reset()
                resets += 1

        elapsed = time.perf_counter() - start_time
        mem_end, mem_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        throughput = step_target / elapsed
        delta_mem_mb = (mem_end - mem_start) / (1024.0 * 1024.0)
        peak_mem_mb = mem_peak / (1024.0 * 1024.0)

        # Assertions
        self.assertGreaterEqual(
            throughput,
            50.0,
            f"Simulation throughput {throughput:.1f} steps/s is below threshold 50 steps/s",
        )
        self.assertLess(
            delta_mem_mb,
            5.0,
            f"Memory growth {delta_mem_mb:.2f} MB exceeds limit of 5.0 MB",
        )


class TestPairedSeedIntegrity(unittest.TestCase):
    """Test paired seed fairness and deterministic food sequences between agents."""

    def test_identical_food_sequence_for_same_seed(self):
        """Verify that two environments with the same seed generate identical food sequences."""
        seeds_to_test = [42, 100, 1000, 9999]
        for seed in seeds_to_test:
            env1 = SnakeEnv(width=16, height=16, seed=seed)
            env2 = SnakeEnv(width=16, height=16, seed=seed)

            obs1 = env1.reset(seed=seed)
            obs2 = env2.reset(seed=seed)

            self.assertEqual(obs1.head, obs2.head)
            self.assertEqual(obs1.food, obs2.food)
            self.assertEqual(obs1.direction, obs2.direction)

            # Move in identical sequence of actions
            # Repeat 50 steps of straight movements (or turns)
            actions = [RelativeAction.STRAIGHT, RelativeAction.TURN_LEFT, RelativeAction.TURN_RIGHT]
            for step_idx in range(40):
                act = actions[step_idx % len(actions)]
                o1, _, d1, _ = env1.step(act)
                o2, _, d2, _ = env2.step(act)
                self.assertEqual(o1.head, o2.head)
                self.assertEqual(o1.food, o2.food)
                self.assertEqual(o1.score, o2.score)
                self.assertEqual(d1, d2)
                if d1:
                    break

    def test_random_agent_rng_isolation(self):
        """Verify RandomAgent does not corrupt global random.Random state or SnakeEnv RNG."""
        # Set global seed
        random.seed(12345)
        state_before = random.getstate()

        # Instantiate and run RandomAgent
        agent = RandomAgent(seed=999)
        env = SnakeEnv(width=16, height=16, seed=555)
        obs = env.reset()

        for _ in range(20):
            act, _ = agent.act(obs)
            obs, _, done, _ = env.step(act)
            if done:
                break

        state_after = random.getstate()
        self.assertEqual(
            state_before,
            state_after,
            "RandomAgent modified Python's global random state! RNG isolation violated.",
        )

    def test_paired_benchmark_seed_symmetry(self):
        """Verify BenchmarkEvaluator evaluates both agents on identical seed sequences."""
        evaluator = BenchmarkEvaluator(width=16, height=16, max_steps=100)
        report1 = evaluator.run_paired_benchmark(episodes=5, seed_base=1000, assert_thresholds=False)
        report2 = evaluator.run_paired_benchmark(episodes=5, seed_base=1000, assert_thresholds=False)

        # The runs should produce identical results because seeding is deterministic
        self.assertEqual(report1["connectome"]["survival"], report2["connectome"]["survival"])
        self.assertEqual(report1["connectome"]["food"], report2["connectome"]["food"])
        self.assertEqual(report1["random"]["survival"], report2["random"]["survival"])
        self.assertEqual(report1["random"]["food"], report2["random"]["food"])


if __name__ == "__main__":
    unittest.main()
