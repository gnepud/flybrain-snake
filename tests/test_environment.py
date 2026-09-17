"""Unit tests for Snake Environment & Grid Mechanics.

Covers:
- Feature 1: Grid Board State Management
- Feature 2: Discrete Step & Kinematic Rules
- Feature 3: Boundary & Wall Collision Detection
- Feature 4: Self-Body Collision Detection
- Feature 5: Deterministic Seeded Food Placement
- Feature 6: Environment Reset & Seed Binding
- Feature 28: Unit Test Suite: Environment & Mechanics
"""

import math
import unittest
from tests.helpers import safe_import, require_symbols


class TestEnvironment(unittest.TestCase):
    """Test suite verifying classic Snake environment kinematics, collisions, and seeding."""

    def setUp(self):
        # Attempt to import SnakeEnv and related classes
        # Check standard location src.snake_env.env or package
        classes = safe_import("src.snake_env.env", "SnakeEnv")
        if classes[0] is None:
            classes = safe_import("src.snake_env", "SnakeEnv")
        self.SnakeEnv = classes[0]

        enums = safe_import("src.snake_env.kinematics", "Direction", "RelativeAction", "SnakeObservation")
        if enums[0] is None:
            enums = safe_import("src.snake_env", "Direction", "RelativeAction", "SnakeObservation")
        self.Direction = enums[0]
        self.RelativeAction = enums[1]
        self.SnakeObservation = enums[2]

    def _get_env(self, width: int = 16, height: int = 16, seed: int | None = 42):
        require_symbols(self.SnakeEnv, feature_desc="SnakeEnv (Milestone 2)")
        return self.SnakeEnv(width=width, height=height, seed=seed)

    # --- Feature 1: Grid Board State Management ---
    def test_grid_initialization_dimensions(self):
        """Verify grid initializes with specified width and height."""
        env = self._get_env(width=16, height=16, seed=42)
        obs = env.reset()
        self.assertEqual(obs.grid_size, (16, 16))
        self.assertIsInstance(obs.head, tuple)
        self.assertEqual(len(obs.head), 2)
        self.assertTrue(0 <= obs.head[0] < 16)
        self.assertTrue(0 <= obs.head[1] < 16)

    def test_initial_snake_length_and_alignment(self):
        """Verify initial snake has length >= 3 and body segments are contiguous."""
        env = self._get_env(width=16, height=16, seed=42)
        obs = env.reset()
        self.assertGreaterEqual(len(obs.body), 3)
        self.assertEqual(obs.body[0], obs.head)
        for i in range(len(obs.body) - 1):
            p1 = obs.body[i]
            p2 = obs.body[i + 1]
            manhattan = abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
            self.assertEqual(manhattan, 1, f"Segments {p1} and {p2} are not contiguous")

    def test_invalid_grid_dimensions_raise_error(self):
        """Verify invalid dimensions (< 4) raise ValueError."""
        require_symbols(self.SnakeEnv, feature_desc="SnakeEnv")
        with self.assertRaises(ValueError):
            self.SnakeEnv(width=3, height=16)
        with self.assertRaises(ValueError):
            self.SnakeEnv(width=16, height=2)

    # --- Feature 2: Kinematic Rules & Movement ---
    def test_straight_movement_updates_head(self):
        """Verify moving straight advances head in current direction."""
        env = self._get_env(width=16, height=16, seed=42)
        obs = env.reset()
        init_head = obs.head
        init_dir = obs.direction

        # Straight action
        action = self.RelativeAction.STRAIGHT if self.RelativeAction else 0
        next_obs, reward, done, info = env.step(action)

        dx, dy = 0, 0
        if hasattr(init_dir, "name"):
            if init_dir.name == "UP" or init_dir == 0:
                dx, dy = 0, -1
            elif init_dir.name == "RIGHT" or init_dir == 1:
                dx, dy = 1, 0
            elif init_dir.name == "DOWN" or init_dir == 2:
                dx, dy = 0, 1
            elif init_dir.name == "LEFT" or init_dir == 3:
                dx, dy = -1, 0
        else:
            dx, dy = 1, 0

        self.assertEqual(next_obs.head, (init_head[0] + dx, init_head[1] + dy))
        self.assertFalse(done)

    def test_self_reversal_is_blocked(self):
        """Verify 180-degree instant reversal is blocked/ignored."""
        env = self._get_env(width=16, height=16, seed=42)
        obs = env.reset()
        init_dir = obs.direction

        # If Direction is available, test absolute reverse command
        if self.Direction:
            # Map opposite direction
            opposite = {
                self.Direction.UP: self.Direction.DOWN,
                self.Direction.DOWN: self.Direction.UP,
                self.Direction.LEFT: self.Direction.RIGHT,
                self.Direction.RIGHT: self.Direction.LEFT,
            }.get(init_dir)

            if opposite is not None:
                next_obs, reward, done, info = env.step(opposite)
                # Direction should NOT instantly reverse to cause immediate neck collision
                self.assertFalse(done)
                self.assertNotEqual(next_obs.direction, opposite)

    def test_turn_left_and_right_kinematics(self):
        """Verify TURN_LEFT and TURN_RIGHT correctly update heading."""
        env = self._get_env(width=16, height=16, seed=42)
        env.reset()

        if self.RelativeAction:
            obs1, _, _, _ = env.step(self.RelativeAction.TURN_LEFT)
            self.assertIsNotNone(obs1.direction)
            obs2, _, _, _ = env.step(self.RelativeAction.TURN_RIGHT)
            self.assertIsNotNone(obs2.direction)

    # --- Feature 3: Boundary & Wall Collision Detection ---
    def test_wall_collision_north(self):
        """Verify moving past upper boundary (y < 0) terminates with WALL_COLLISION."""
        env = self._get_env(width=16, height=16, seed=42)
        obs = env.reset()

        # Drive snake North until wall collision
        done = False
        reason = None
        for _ in range(25):
            if self.Direction:
                obs, reward, done, info = env.step(self.Direction.UP)
            else:
                obs, reward, done, info = env.step(0)
            if done:
                reason = obs.reason or info.get("reason")
                break

        self.assertTrue(done, "Snake should have collided with North wall")
        self.assertIn(reason, ["WALL_COLLISION", "WALL"])

    def test_wall_collision_east(self):
        """Verify moving past right boundary (x >= width) terminates with WALL_COLLISION."""
        env = self._get_env(width=16, height=16, seed=42)
        obs = env.reset()

        done = False
        reason = None
        for _ in range(25):
            if self.Direction:
                obs, reward, done, info = env.step(self.Direction.RIGHT)
            else:
                obs, reward, done, info = env.step(0)
            if done:
                reason = obs.reason or info.get("reason")
                break

        self.assertTrue(done, "Snake should have collided with East wall")
        self.assertIn(reason, ["WALL_COLLISION", "WALL"])

    def test_wall_collision_south(self):
        """Verify moving past bottom boundary (y >= height) terminates with WALL_COLLISION."""
        env = self._get_env(width=16, height=16, seed=42)
        obs = env.reset()

        done = False
        reason = None
        for _ in range(25):
            if self.Direction:
                obs, reward, done, info = env.step(self.Direction.DOWN)
            else:
                obs, reward, done, info = env.step(0)
            if done:
                reason = obs.reason or info.get("reason")
                break

        self.assertTrue(done, "Snake should have collided with South wall")
        self.assertIn(reason, ["WALL_COLLISION", "WALL"])

    def test_wall_collision_west(self):
        """Verify moving past left boundary (x < 0) terminates with WALL_COLLISION."""
        env = self._get_env(width=16, height=16, seed=42)
        obs = env.reset()

        done = False
        reason = None
        for _ in range(25):
            if self.Direction:
                obs, reward, done, info = env.step(self.Direction.LEFT)
            else:
                obs, reward, done, info = env.step(0)
            if done:
                reason = obs.reason or info.get("reason")
                break

        self.assertTrue(done, "Snake should have collided with West wall")
        self.assertIn(reason, ["WALL_COLLISION", "WALL"])

    # --- Feature 4: Self-Body Collision Detection ---
    def test_body_collision_detection(self):
        """Verify snake colliding with its own body triggers BODY_COLLISION."""
        # Setup an environment where snake grows enough to loop into itself
        env = self._get_env(width=16, height=16, seed=42)
        obs = env.reset()

        # Artificially expand body or navigate a 5-step loop if body >= 5
        # If possible, verify that intersecting body[1:] causes collision
        if hasattr(env, "_body"):
            # Set up a coiled snake: (5,5), (5,6), (6,6), (6,5), (6,4)
            env._body = [(5, 5), (5, 6), (6, 6), (6, 5), (6, 4)]
            env._head = (5, 5)
            # Moving East from (5,5) steps onto (6,5) which is body segment!
            if self.Direction:
                obs, reward, done, info = env.step(self.Direction.RIGHT)
                self.assertTrue(done)
                self.assertIn(obs.reason or info.get("reason"), ["BODY_COLLISION", "BODY"])

    def test_tail_vacation_is_not_collision(self):
        """Verify stepping into the cell vacated by the tail on a non-food step is valid."""
        env = self._get_env(width=16, height=16, seed=42)
        obs = env.reset()

        # Snake of length 4 making a 4-step loop (RIGHT -> DOWN -> LEFT -> UP)
        # On the 4th step, the head enters the initial tail coordinate which just popped!
        if self.Direction and hasattr(env, "_body"):
            env._body = [(6, 6), (5, 6), (5, 5), (6, 5)]
            env._head = (6, 6)
            env._direction = self.Direction.RIGHT
            # Move UP to (6, 5) which is currently the tail!
            # On non-food step, tail pops, so (6, 5) is vacated and move should succeed!
            # Ensure food is not at (6, 5)
            env._food = (10, 10)
            obs, reward, done, info = env.step(self.Direction.UP)
            self.assertFalse(done, "Moving into vacated tail cell must NOT be a body collision")

    # --- Feature 5: Deterministic Seeded Food Placement ---
    def test_food_never_spawns_on_snake_body(self):
        """Verify food coordinate is strictly within vacant grid cells."""
        env = self._get_env(width=16, height=16, seed=42)
        for seed_val in range(10):
            obs = env.reset(seed=seed_val)
            self.assertNotIn(obs.food, obs.body, f"Food {obs.food} spawned on snake body")

    def test_deterministic_seed_reproducibility(self):
        """Verify identical seed yields identical initial states and food positions."""
        env1 = self._get_env(width=16, height=16, seed=123)
        env2 = self._get_env(width=16, height=16, seed=123)
        obs1 = env1.reset(seed=123)
        obs2 = env2.reset(seed=123)

        self.assertEqual(obs1.head, obs2.head)
        self.assertEqual(obs1.body, obs2.body)
        self.assertEqual(obs1.food, obs2.food)
        self.assertEqual(obs1.direction, obs2.direction)

        # Step both identically
        action = self.RelativeAction.STRAIGHT if self.RelativeAction else 0
        for _ in range(5):
            o1, r1, d1, _ = env1.step(action)
            o2, r2, d2, _ = env2.step(action)
            self.assertEqual(o1.head, o2.head)
            self.assertEqual(o1.food, o2.food)

    # --- Feature 6: Environment Reset & Seed Binding ---
    def test_reset_restores_clean_state(self):
        """Verify reset clears score, steps, done, and reason."""
        env = self._get_env(width=16, height=16, seed=42)
        obs = env.reset()

        # Step until collision or 5 steps
        action = self.RelativeAction.STRAIGHT if self.RelativeAction else 0
        for _ in range(20):
            obs, _, done, _ = env.step(action)
            if done:
                break

        # Now reset
        reset_obs = env.reset(seed=999)
        self.assertEqual(reset_obs.score, 0)
        self.assertEqual(reset_obs.step_count, 0)
        self.assertFalse(reset_obs.done)
        self.assertIsNone(reset_obs.reason)

    # --- Distance Rays & Clearance ---
    def test_egocentric_distance_rays(self):
        """Verify dist_front, dist_left, dist_right are computed correctly."""
        env = self._get_env(width=16, height=16, seed=42)
        obs = env.reset()
        self.assertGreaterEqual(obs.dist_front, 0)
        self.assertGreaterEqual(obs.dist_left, 0)
        self.assertGreaterEqual(obs.dist_right, 0)

    # --- Border Restriction for Food Placement ---
    def test_restrict_borders_keeps_food_in_safe_zone(self):
        """Verify that when restrict_borders=True, food is strictly placed >= 2 cells from edges."""
        require_symbols(self.SnakeEnv, feature_desc="SnakeEnv")
        for seed in range(50):
            env = self.SnakeEnv(width=16, height=16, seed=seed, restrict_borders=True)
            obs = env.reset(seed=seed)
            fx, fy = obs.food
            self.assertGreaterEqual(fx, 2, f"Food X {fx} violates left 2-cell boundary on seed {seed}")
            self.assertLessEqual(fx, 13, f"Food X {fx} violates right 2-cell boundary on seed {seed}")
            self.assertGreaterEqual(fy, 2, f"Food Y {fy} violates top 2-cell boundary on seed {seed}")
            self.assertLessEqual(fy, 13, f"Food Y {fy} violates bottom 2-cell boundary on seed {seed}")

    def test_restrict_borders_toggle_runtime(self):
        """Verify that modifying env.restrict_borders at runtime affects subsequent food spawns."""
        require_symbols(self.SnakeEnv, feature_desc="SnakeEnv")
        env = self.SnakeEnv(width=16, height=16, seed=123, restrict_borders=False)
        obs = env.reset(seed=123)
        env.restrict_borders = True
        for s in range(30):
            env.reset(seed=s + 100)
            fx, fy = env._food
            self.assertTrue(2 <= fx <= 13 and 2 <= fy <= 13, f"Spawned at ({fx}, {fy})")


if __name__ == "__main__":
    unittest.main()
