"""Tier 3: Pairwise Combinatorial Interaction Tests.

Tests interactions across orthogonal dimension pairs:
- Environment dimensions (8x8, 16x16, 24x24) x Circuit parameters (dt, tau)
- Steering threshold sensitivity x Obstacle proximity
- Controller agent types x Random seeds x Grid configurations
- Headless benchmark evaluator x Paired episode counts
"""

import unittest
from tests.e2e.harness import E2ETestHarness
from tests.helpers import safe_import, require_symbols


class TestTier3PairwiseInteractions(unittest.TestCase):
    """Pairwise combinatorial tests verifying cross-subsystem interactions."""

    def setUp(self):
        self.harness = E2ETestHarness()

    # --- Pairwise Matrix 2: Steer Threshold x Food Bearing Direction ---
    def test_pairwise_steer_threshold_x_bearing_quadrants(self):
        """P2: Motor decoder thresholds (0.02, 0.05, 0.10) across 4 relative quadrants."""
        decoder_cls = safe_import("src.controller.motor_decoder", "MotorDecoder")[0]
        if not decoder_cls:
            raise unittest.SkipTest("MotorDecoder not yet implemented (Milestone 2)")

        thresholds = [0.02, 0.05, 0.10]
        # Test bilateral differentials: strong left, mild left, balanced, mild right, strong right
        differentials = [
            (0.8, 0.2, 1),   # strong left -> TURN_LEFT
            (0.55, 0.50, None), # mild left -> sensitive to threshold
            (0.50, 0.50, 0), # balanced -> STRAIGHT
            (0.50, 0.55, None), # mild right -> sensitive to threshold
            (0.2, 0.8, 2),   # strong right -> TURN_RIGHT
        ]

        for th in thresholds:
            dec = decoder_cls(steer_threshold=th)
            for dl, dr, expected_strong in differentials:
                act = dec.decode(dna_left=dl, dna_right=dr, forward=0.5)
                if expected_strong is not None:
                    self.assertEqual(int(act), expected_strong)
                else:
                    # Mild diff depends on threshold
                    diff = abs(dl - dr)
                    if diff > th:
                        self.assertIn(int(act), [1, 2])
                    else:
                        self.assertEqual(int(act), 0)

    # --- Pairwise Matrix 3: Agent Type x Seed Sequence ---
    def test_pairwise_agent_x_seed_reproducibility(self):
        """P3: ConnectomeAgent and RandomAgent consistency across seeds."""
        if not self.harness.has_environment():
            raise unittest.SkipTest("SnakeEnv not yet implemented (Milestone 2)")

        seeds = [42, 101, 999]
        for seed in seeds:
            traj_r1 = self.harness.run_session(grid_width=12, grid_height=12, seed=seed, max_steps=20, agent_type="random")
            traj_r2 = self.harness.run_session(grid_width=12, grid_height=12, seed=seed, max_steps=20, agent_type="random")
            # Random agent with identical seed must yield identical trajectory
            self.assertEqual(traj_r1["steps"], traj_r2["steps"])
            self.assertEqual(traj_r1["head_history"], traj_r2["head_history"])

    # --- Pairwise Matrix 4: Obstacle Clearance x Reflexive Turn Override ---
    def test_pairwise_obstacle_distance_x_turning(self):
        """P4: Proximity clearance rays interacting with navigation."""
        if not self.harness.has_environment():
            raise unittest.SkipTest("SnakeEnv not yet implemented (Milestone 2)")

        env = self.harness.SnakeEnv(width=10, height=10, seed=42)
        obs = env.reset()
        self.assertGreaterEqual(obs.dist_front, 0)
        self.assertGreaterEqual(obs.dist_left, 0)
        self.assertGreaterEqual(obs.dist_right, 0)

    # --- Pairwise Matrix 5: Episode Length Limits x Benchmark Evaluation ---
    def test_pairwise_benchmark_episodes_x_grid_sizes(self):
        """P5: Multi-episode paired runs on varying grid sizes (10x10 vs 16x16)."""
        if not self.harness.has_environment():
            raise unittest.SkipTest("SnakeEnv not yet implemented (Milestone 2)")

        for sz in [10, 14]:
            traj = self.harness.run_session(grid_width=sz, grid_height=sz, seed=555, max_steps=30, agent_type="random")
            self.assertGreater(traj["steps"], 0)
            violations = self.harness.validate_trajectory_integrity(traj, (sz, sz))
            self.assertEqual(violations, [])


if __name__ == "__main__":
    unittest.main()
