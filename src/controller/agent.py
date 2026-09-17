"""Autonomous Agents for FlyBrain Snake.

Provides:
- FlyBrainAgent: 166,700-neuron Drosophila connectome spiking agent (pip flybrain).
- RandomAgent: Uniform pseudo-random baseline agent for statistical benchmarking.
"""

import random
from typing import Any, Dict, Optional, Tuple

from src.snake_env.kinematics import RelativeAction, SnakeObservation


class RandomAgent:
    """Baseline agent making uniform random decisions for comparative benchmarking."""

    def __init__(self, seed: Optional[int] = None):
        self._rng = random.Random(seed)

    def reset(self) -> None:
        pass

    def act(self, obs: SnakeObservation) -> Tuple[RelativeAction, Dict[str, Any]]:
        action = RelativeAction(self._rng.choice([0, 1, 2]))
        return action, {"action": action}


__all__ = ["RandomAgent"]
