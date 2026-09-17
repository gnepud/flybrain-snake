"""Closed-Loop Controller package.

Exports:
- MotorDecoder
- decode_motor
- FlyBrainAgent
- RandomAgent
"""

from src.controller.motor_decoder import MotorDecoder, decode_motor
from src.controller.flybrain_agent import FlyBrainAgent
from src.controller.agent import RandomAgent

__all__ = [
    "MotorDecoder",
    "decode_motor",
    "FlyBrainAgent",
    "RandomAgent",
]
