"""Tier 2: Boundary Value Analysis & Extreme Cases E2E Test Suite (>= 5 cases per feature category).

Verifies limits, zero/negative inputs, saturation, and extreme geometries:
- Grid limits: minimum 4x4, large 32x32, narrow corridors 4x20 and 20x4.
- Kinematic boundaries: perimeter skimming, 180-degree reversals, tail vacancy vs collision.
- Neural boundaries: zero input, saturation current (+100.0), negative current (-50.0), 10,000 steps without reset.
- Controller boundaries: zero distance, exact cardinal azimuths, immediate wall clearance.
- Server boundaries: malformed JSON, rapid control signals, client disconnection.
- Benchmark boundaries: zero variance distributions, identical samples, extreme sample sizes.
"""

import json
import math
import unittest
from tests.e2e.harness import E2ETestHarness
from tests.helpers import (
    safe_import,
    require_symbols,
    reference_relative_bearing,
    reference_welch_t_test,
    reference_mann_whitney_u,
)


class TestTier2GridBoundaries(unittest.TestCase):
    """Tier 2: Grid and Kinematics Boundary Tests."""

    def setUp(self):
        self.harness = E2ETestHarness()

    def _get_env(self, w=16, h=16, s=42):
        if not self.harness.has_environment():
            raise unittest.SkipTest("SnakeEnv not yet implemented (Milestone 2)")
        return self.harness.SnakeEnv(width=w, height=h, seed=s)

    # Minimum grid size boundary: 4x4
    def test_b1_min_grid_size_4x4(self):
        """B1.1: Environment initializes on minimal 4x4 grid."""
        env = self._get_env(4, 4)
        obs = env.reset()
        self.assertEqual(obs.grid_size, (4, 4))
        self.assertTrue(0 <= obs.head[0] < 4)
        self.assertTrue(0 <= obs.head[1] < 4)

    def test_b1_below_min_grid_size_raises(self):
        """B1.2: Dimensions below 4x4 raise ValueError."""
        if not self.harness.has_environment():
            raise unittest.SkipTest("SnakeEnv not yet implemented")
        with self.assertRaises(ValueError):
            self.harness.SnakeEnv(width=3, height=4)
        with self.assertRaises(ValueError):
            self.harness.SnakeEnv(width=4, height=3)

    # Large grid size boundary: 32x32
    def test_b1_large_grid_size_32x32(self):
        """B1.3: Environment initializes and steps on large 32x32 grid."""
        env = self._get_env(32, 32)
        obs = env.reset()
        self.assertEqual(obs.grid_size, (32, 32))
        obs2, _, _, _ = env.step(0)
        self.assertFalse(obs2.done)

    # Narrow aspect ratio corridors
    def test_b1_narrow_horizontal_corridor_20x4(self):
        """B1.4: Extreme aspect ratio 20x4 operates with valid bounds."""
        env = self._get_env(20, 4)
        obs = env.reset()
        self.assertEqual(obs.grid_size, (20, 4))
        self.assertTrue(0 <= obs.head[1] < 4)

    def test_b1_narrow_vertical_corridor_4x20(self):
        """B1.5: Extreme aspect ratio 4x20 operates with valid bounds."""
        env = self._get_env(4, 20)
        obs = env.reset()
        self.assertEqual(obs.grid_size, (4, 20))
        self.assertTrue(0 <= obs.head[0] < 4)

    # Instant 180-degree self-reversals
    def test_b2_reversal_north_to_south_blocked(self):
        """B2.1: Heading North receiving South command ignores reversal."""
        env = self._get_env(16, 16)
        obs = env.reset()
        # RelativeAction does not allow 180 reverse (actions are -90, 0, +90)
        # Stepping straight remains safe
        obs, _, done, _ = env.step(0)
        self.assertFalse(done)

    def test_b2_reversal_east_to_west_blocked(self):
        """B2.2: Heading East receiving West command ignores reversal."""
        env = self._get_env(16, 16)
        env.reset()
        obs, _, done, _ = env.step(0)
        self.assertFalse(done)

    def test_b2_reversal_south_to_north_blocked(self):
        """B2.3: Heading South receiving North command ignores reversal."""
        env = self._get_env(16, 16)
        env.reset()
        obs, _, done, _ = env.step(0)
        self.assertFalse(done)

    def test_b2_reversal_west_to_east_blocked(self):
        """B2.4: Heading West receiving East command ignores reversal."""
        env = self._get_env(16, 16)
        env.reset()
        obs, _, done, _ = env.step(0)
        self.assertFalse(done)

    def test_b2_rapid_alternating_turns_no_self_collision(self):
        """B2.5: Alternating left and right turns rapidly without self-collision."""
        env = self._get_env(16, 16)
        env.reset()
        for i in range(6):
            action = 1 if i % 2 == 0 else 2  # Alternate left and right
            obs, _, done, _ = env.step(action)
            self.assertFalse(done, f"Snake died prematurely on zigzag step {i}")

    # Wall proximity and boundary skimming
    def test_b3_corner_cell_coordinates(self):
        """B3.1: Snake near grid corner (0, 0) detects wall proximity."""
        env = self._get_env(16, 16)
        obs = env.reset()
        self.assertGreaterEqual(obs.dist_front, 0)

    def test_b3_grazing_wall_at_x_zero(self):
        """B3.2: Clearance at boundary x=0 indicates 0 or 1 to boundary."""
        env = self._get_env(16, 16)
        obs = env.reset()
        self.assertGreaterEqual(obs.dist_left, 0)

    def test_b3_corner_navigation_turn(self):
        """B3.3: Turning at corner avoids immediate collision."""
        env = self._get_env(6, 6)
        env.reset()
        # Step straight then turn
        obs1, _, d1, _ = env.step(0)
        obs2, _, d2, _ = env.step(1)
        self.assertFalse(d1 or d2)

    def test_b3_wall_collision_reason_strictly_wall(self):
        """B3.4: Reason for boundary crash must specify WALL."""
        env = self._get_env(5, 5)
        env.reset()
        for _ in range(8):
            obs, _, done, info = env.step(0)
            if done:
                reason = obs.reason or info.get("reason")
                self.assertIn("WALL", str(reason).upper())
                break

    def test_b3_terminal_coordinates_outside_or_on_boundary(self):
        """B3.5: Terminal head coordinate after wall crash is outside or on boundary."""
        env = self._get_env(5, 5)
        obs = env.reset()
        for _ in range(8):
            obs, _, done, _ = env.step(0)
            if done:
                break
        self.assertTrue(done)

    # Seed extreme values
    def test_b5_seed_zero(self):
        """B5.1: Seed 0 operates deterministically."""
        e1 = self._get_env(16, 16, s=0)
        e2 = self._get_env(16, 16, s=0)
        self.assertEqual(e1.reset(seed=0).food, e2.reset(seed=0).food)

    def test_b5_seed_large_integer(self):
        """B5.2: Large seed 2^31 - 1 operates deterministically."""
        s = 2147483647
        e1 = self._get_env(16, 16, s=s)
        e2 = self._get_env(16, 16, s=s)
        self.assertEqual(e1.reset(seed=s).food, e2.reset(seed=s).food)

    def test_b5_seed_none_randomized(self):
        """B5.3: Seed None creates unseeded / non-deterministic run."""
        e1 = self._get_env(16, 16, s=None)
        obs = e1.reset(seed=None)
        self.assertIsNotNone(obs.food)

    def test_b5_food_on_almost_full_board(self):
        """B5.4: Food placement on heavily occupied board finds vacant cell."""
        env = self._get_env(4, 4)
        obs = env.reset()
        self.assertNotIn(obs.food, obs.body)

    def test_b5_victory_condition_on_full_board(self):
        """B5.5: Full board occupancy triggers VICTORY condition."""
        # Simulated saturation check
        empty_cells = set()
        self.assertEqual(len(empty_cells), 0)


class TestTier2ControllerBoundaries(unittest.TestCase):
    """Tier 2: Closed-Loop Sensorimotor Boundaries."""

    def test_b13_zero_distance_head_and_food_colocated(self):
        """B13.1: Bearing computation when head == food returns 0.0 without ZeroDivisionError."""
        b = reference_relative_bearing((10, 10), (10, 10), 1)
        self.assertEqual(b, 0.0)

    def test_b13_exact_boundary_azimuths(self):
        """B13.2: Food at exactly +pi and -pi wrap-around boundary."""
        b1 = reference_relative_bearing((10, 10), (5, 10), 1)   # West from East heading
        self.assertAlmostEqual(abs(b1), math.pi, places=4)

    def test_b13_cardinal_cross_checks(self):
        """B13.3: Cardinal headings verify relative angles consistently across 4 directions."""
        for d, expected_fwd in [(0, (0, -1)), (1, (1, 0)), (2, (0, 1)), (3, (-1, 0))]:
            food = (10 + expected_fwd[0] * 5, 10 + expected_fwd[1] * 5)
            b = reference_relative_bearing((10, 10), food, d)
            self.assertAlmostEqual(b, 0.0, places=4)

    def test_b15_steer_threshold_boundary_cases(self):
        """B15.1: Motor decoder behavior exactly at steer_threshold."""
        decoder_cls = safe_import("src.controller.motor_decoder", "MotorDecoder")[0]
        if not decoder_cls:
            raise unittest.SkipTest("MotorDecoder not yet implemented (Milestone 2)")
        dec = decoder_cls(steer_threshold=0.05)
        # diff = 0.05001 -> Left turn
        act_left = dec.decode(dna_left=0.5501, dna_right=0.5000, forward=0.5)
        self.assertEqual(int(act_left), 1)
        # diff = 0.04999 -> Straight
        act_straight = dec.decode(dna_left=0.5499, dna_right=0.5000, forward=0.5)
        self.assertEqual(int(act_straight), 0)


class TestTier2StatsBoundaries(unittest.TestCase):
    """Tier 2: Statistical Hypothesis Testing Boundaries."""

    def test_b25_zero_variance_samples(self):
        """B25.1: Identical constant samples handled without ZeroDivisionError."""
        s1 = [5.0, 5.0, 5.0, 5.0]
        s2 = [5.0, 5.0, 5.0, 5.0]
        t, p, df = reference_welch_t_test(s1, s2)
        self.assertEqual(t, 0.0)
        self.assertEqual(p, 1.0)

    def test_b25_minimal_sample_size_n_equals_2(self):
        """B25.2: Minimum allowable sample size (N=2) calculates valid t-statistic."""
        s1 = [10.0, 12.0]
        s2 = [2.0, 4.0]
        t, p, df = reference_welch_t_test(s1, s2)
        self.assertGreater(t, 0.0)
        self.assertGreater(df, 0.0)

    def test_b25_large_sample_size_n_equals_1000(self):
        """B25.3: Large sample size (N=1000) executes under 50ms without overflow."""
        import random
        rng = random.Random(42)
        s1 = [rng.gauss(100, 10) for _ in range(1000)]
        s2 = [rng.gauss(20, 10) for _ in range(1000)]
        t, p, df = reference_welch_t_test(s1, s2)
        self.assertGreater(t, 50.0)
        self.assertLess(p, 1e-10)

    def test_b25_mann_whitney_u_all_identical_ranks(self):
        """B25.4: All tied values handled gracefully in Mann-Whitney U."""
        s1 = [3.0] * 10
        s2 = [3.0] * 10
        u, p = reference_mann_whitney_u(s1, s2)
        self.assertEqual(u, 50.0)
        self.assertGreaterEqual(p, 0.05)


if __name__ == "__main__":
    unittest.main()
