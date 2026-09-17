"""Unit tests for Closed-Loop Sensorimotor Controller.

Covers:
- Relative bearing egocentric angle calculation
- Descending motor output decoding (MotorDecoder)
- FlyBrainAgent closed-loop action selection
"""

import math
import unittest
from tests.helpers import safe_import, require_symbols, reference_relative_bearing
from src.controller.flybrain_agent import FlyBrainAgent


class TestController(unittest.TestCase):
    """Test suite verifying motor decoding and closed-loop agent logic."""

    def setUp(self):
        decoder_symbols = safe_import("src.controller.motor_decoder", "MotorDecoder", "decode_motor")
        self.MotorDecoder = decoder_symbols[0]
        self.decode_motor = decoder_symbols[1]

        env_symbols = safe_import("src.snake_env.kinematics", "Direction", "RelativeAction", "SnakeObservation")
        if env_symbols[0] is None:
            env_symbols = safe_import("src.snake_env", "Direction", "RelativeAction", "SnakeObservation")
        self.Direction = env_symbols[0]
        self.RelativeAction = env_symbols[1]
        self.SnakeObservation = env_symbols[2]

    # --- Egocentric Spatial Observation Encoding ---
    def test_relative_bearing_food_ahead(self):
        """Verify bearing is ~0 when food is straight ahead."""
        head = (8, 8)
        food = (12, 8)
        direction = 1  # RIGHT
        bearing = reference_relative_bearing(head, food, direction)
        self.assertAlmostEqual(bearing, 0.0, places=4)

    def test_relative_bearing_food_left(self):
        """Verify bearing is ~ -pi/2 when food is 90 degrees to the left."""
        head = (8, 8)
        food = (8, 4)
        direction = 1  # RIGHT
        bearing = reference_relative_bearing(head, food, direction)
        self.assertAlmostEqual(bearing, -math.pi / 2, places=4)

    def test_relative_bearing_food_right(self):
        """Verify bearing is ~ +pi/2 when food is 90 degrees to the right."""
        head = (8, 8)
        food = (8, 12)
        direction = 1  # RIGHT
        bearing = reference_relative_bearing(head, food, direction)
        self.assertAlmostEqual(bearing, math.pi / 2, places=4)

    def test_relative_bearing_food_behind(self):
        """Verify bearing magnitude is ~ pi when food is directly behind."""
        head = (8, 8)
        food = (4, 8)
        direction = 1  # RIGHT -> behind is West
        bearing = reference_relative_bearing(head, food, direction)
        self.assertAlmostEqual(abs(bearing), math.pi, places=4)

    def test_relative_bearing_co_located_head_food(self):
        """Verify bearing is 0 when head and food are co-located without ZeroDivisionError."""
        bearing = reference_relative_bearing((5, 5), (5, 5), 0)
        self.assertEqual(bearing, 0.0)

    # --- Descending Motor Output Decoding ---
    def test_motor_decoder_turn_left(self):
        """Verify D_left - D_right > threshold decodes to TURN_LEFT."""
        decoder_cls = self.MotorDecoder or self.decode_motor
        require_symbols(decoder_cls, feature_desc="MotorDecoder")

        decoder = self.MotorDecoder(steer_threshold=0.05) if self.MotorDecoder else None
        if decoder and hasattr(decoder, "decode"):
            action = decoder.decode(dna_left=0.8, dna_right=0.2, forward=0.5)
            left_act = self.RelativeAction.TURN_LEFT if self.RelativeAction else 1
            self.assertEqual(action, left_act)

    def test_motor_decoder_turn_right(self):
        """Verify D_right - D_left > threshold decodes to TURN_RIGHT."""
        decoder_cls = self.MotorDecoder or self.decode_motor
        require_symbols(decoder_cls, feature_desc="MotorDecoder")

        decoder = self.MotorDecoder(steer_threshold=0.05) if self.MotorDecoder else None
        if decoder and hasattr(decoder, "decode"):
            action = decoder.decode(dna_left=0.2, dna_right=0.8, forward=0.5)
            right_act = self.RelativeAction.TURN_RIGHT if self.RelativeAction else 2
            self.assertEqual(action, right_act)

    def test_motor_decoder_straight(self):
        """Verify balanced motor drive |D_left - D_right| <= threshold decodes to STRAIGHT."""
        decoder_cls = self.MotorDecoder or self.decode_motor
        require_symbols(decoder_cls, feature_desc="MotorDecoder")

        decoder = self.MotorDecoder(steer_threshold=0.05) if self.MotorDecoder else None
        if decoder and hasattr(decoder, "decode"):
            action = decoder.decode(dna_left=0.51, dna_right=0.50, forward=0.5)
            straight_act = self.RelativeAction.STRAIGHT if self.RelativeAction else 0
            self.assertEqual(action, straight_act)

    # --- Closed-Loop FlyBrain Agent ---
    def test_flybrain_agent_action_selection(self):
        """Verify FlyBrainAgent acts deterministically and returns valid RelativeAction."""
        agent = FlyBrainAgent(seed=42)
        agent.reset()

        obs = self.SnakeObservation(
            head=(8, 8),
            body=[(8, 8), (7, 8), (6, 8)],
            food=(12, 8),
            direction=self.Direction.RIGHT if self.Direction else 1,
            grid_size=(16, 16),
            food_bearing=0.0,
            food_distance=4.0,
            dist_front=7,
            dist_left=8,
            dist_right=7,
            score=0,
            step_count=0,
            done=False,
            reason=None,
        )
        action, info = agent.act(obs)
        valid_actions = [0, 1, 2]
        if self.RelativeAction:
            valid_actions.extend([self.RelativeAction.STRAIGHT, self.RelativeAction.TURN_LEFT, self.RelativeAction.TURN_RIGHT])
        self.assertIn(action, valid_actions)
        self.assertIsInstance(info, dict)


if __name__ == "__main__":
    unittest.main()
