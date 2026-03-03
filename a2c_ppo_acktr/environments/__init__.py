"""
Modular environment registry for VLM RL training.
Supports: MiniWorld, ALFWorld, WebShop, GymCards
"""

from .registry import EnvironmentRegistry, get_environment
from .base import BaseEnvironment, EnvironmentWrapper
from .miniworld_env import MiniWorldEnvironment
from .alfworld_env import ALFWorldEnvironment
from .webshop_env import WebShopEnvironment
from .gymcards_env import GymCardsEnvironment

__all__ = [
    "EnvironmentRegistry",
    "get_environment", 
    "BaseEnvironment",
    "EnvironmentWrapper",
    "MiniWorldEnvironment",
    "ALFWorldEnvironment",
    "WebShopEnvironment",
    "GymCardsEnvironment",
]

