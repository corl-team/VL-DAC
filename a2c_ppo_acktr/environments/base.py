"""
Base environment class defining the interface for all environments.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
from collections import deque
import numpy as np


class BaseEnvironment(ABC):
    """
    Abstract base class for all environments in the VLM RL framework.
    Provides a unified interface for different environment types.
    """
    
    def __init__(
        self,
        env_name: str,
        seed: int,
        rank: int,
        max_episode_steps: int = 128,
        max_image_obs_len: int = 4,
        log_dir: Optional[str] = None,
        **kwargs
    ):
        self.env_name = env_name
        self.seed = seed
        self.rank = rank
        self.max_episode_steps = max_episode_steps
        self.max_image_obs_len = max_image_obs_len
        self.log_dir = log_dir
        
        self.image_observations: deque = deque(maxlen=max_image_obs_len)
        self.env = None
        self.infos: List[Dict] = []
        self.step_count = 0
        
    @abstractmethod
    def create_env(self) -> Any:
        """Create and return the underlying environment."""
        pass
    
    @abstractmethod
    def reset(self, **kwargs) -> Tuple[Any, Dict]:
        """Reset the environment and return initial observation."""
        pass
    
    @abstractmethod
    def step(self, action: Any) -> Tuple[Any, float, bool, Dict]:
        """Take a step in the environment."""
        pass
    
    @abstractmethod
    def get_observation_prompt(self, prompt_version: str = "v1") -> List[Dict]:
        """Generate the observation prompt for the VLM."""
        pass
    
    @abstractmethod
    def text_to_action(self, text_action: str) -> Any:
        """Convert VLM text output to environment action."""
        pass
    
    @abstractmethod
    def process_reward(self, reward: float, done: bool, info: Dict) -> float:
        """Process and optionally transform the reward."""
        pass
    
    @property
    @abstractmethod
    def action_space(self) -> Any:
        """Return the action space of the environment."""
        pass
    
    def clear_history(self):
        """Clear observation history (called on episode end)."""
        self.image_observations.clear()
        self.step_count = 0
        self.infos = []
    
    def add_observation(self, obs: Any):
        """Add observation to history."""
        self.image_observations.append(obs)
        
    def get_task_description(self, prompt_version: str = "v1") -> str:
        """Get task description for the environment."""
        return ""
    
    def close(self):
        """Close the environment."""
        if self.env is not None:
            self.env.close()


class EnvironmentWrapper:
    """
    Wrapper that provides a unified interface for all environments.
    Handles observation preprocessing, action postprocessing, and prompt generation.
    """
    
    def __init__(self, env: BaseEnvironment, prompt_version: str = "v1"):
        self.env = env
        self.prompt_version = prompt_version
        
    def reset(self, **kwargs) -> List[Dict]:
        """Reset and return formatted observation."""
        obs, info = self.env.reset(**kwargs)
        self.env.clear_history()
        self.env.add_observation(obs)
        return self.env.get_observation_prompt(self.prompt_version)
    
    def step(self, text_action: str) -> Tuple[List[Dict], float, bool, Dict]:
        """Step with text action and return formatted observation."""
        action = self.env.text_to_action(text_action)
        obs, reward, done, info = self.env.step(action)
        
        reward = self.env.process_reward(reward, done, info)
        
        if done:
            # Episode ended - reset environment and get new initial observation
            self.env.clear_history()
            obs, reset_info = self.env.reset()
            self.env.add_observation(obs)
        else:
            self.env.add_observation(obs)
        
        return self.env.get_observation_prompt(self.prompt_version), reward, done, info, action
    
    @property
    def action_space(self):
        return self.env.action_space
    
    def close(self):
        self.env.close()

