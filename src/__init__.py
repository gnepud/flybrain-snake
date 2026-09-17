"""FlyBrain Snake: Whole-Brain Connectome Sensorimotor Neural Circuit Snake Game.

Autonomous snake navigation driven by simulated Drosophila melanogaster whole-brain connectome
spiking dynamics (pip flybrain) with real-time web visualization and automated evaluation.
"""

__version__ = "0.1.0"
__author__ = "FlyBrain Snake Project"

from src.snake_env import SnakeEnv, Direction, RelativeAction, SnakeObservation
from src.controller import FlyBrainAgent, RandomAgent, MotorDecoder, decode_motor
from src.web.server import SnakeServer, create_server, run_server

__all__ = [
    "__version__",
    "SnakeEnv",
    "Direction",
    "RelativeAction",
    "SnakeObservation",
    "FlyBrainAgent",
    "RandomAgent",
    "MotorDecoder",
    "decode_motor",
    "SnakeServer",
    "create_server",
    "run_server",
]
