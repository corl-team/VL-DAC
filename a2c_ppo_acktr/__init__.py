"""
A2C/PPO/ACKTR implementations for VLM RL training.

This package provides:
- Modular environment support (MiniWorld, ALFWorld, WebShop, GymCards)
- Modular model adapters (Qwen2VL, Gemma3, LLaVA)
- PPO algorithm with token-level rewards
- Flexible configuration via YAML files
"""

from . import algo
from . import utils

__all__ = [
    "algo",
    "utils",
]

