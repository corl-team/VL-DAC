"""
Modular model registry for VLM RL training.
Supports: Qwen2VL, Gemma3, LLaVA
"""

from .registry import ModelRegistry, get_model
from .base import BaseVLMAdapter
from .qwen2vl import Qwen2VLAdapter
from .gemma3 import Gemma3Adapter

__all__ = [
    "ModelRegistry",
    "get_model",
    "BaseVLMAdapter",
    "Qwen2VLAdapter",
    "Gemma3Adapter",
]

