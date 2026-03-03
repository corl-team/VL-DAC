"""
ALFWorld environment implementation.
"""

import os
import numpy as np
from typing import Dict, List, Any, Optional, Tuple

try:
    from alfworld.agents.utils.misc import get_templated_task_desc
    import alfworld.agents.environment as environment
    HAS_ALFWORLD = True
except ImportError:
    HAS_ALFWORLD = False

from .base import BaseEnvironment
from .registry import EnvironmentRegistry


@EnvironmentRegistry.register("alfworld")
@EnvironmentRegistry.register("alfred")
class ALFWorldEnvironment(BaseEnvironment):
    """
    ALFWorld environment wrapper for VLM RL training.
    """
    
    def __init__(self, config_path: Optional[str] = None, **kwargs):
        self.config_path = config_path
        self.traj_data = None
        self.admissible_commands = set()
        self.observation_text = ""
        super().__init__(**kwargs)
        self.env = self.create_env()
        
    def create_env(self) -> Any:
        """Create ALFWorld environment."""
        if not HAS_ALFWORLD:
            raise ImportError("alfworld is not installed. Please install it first.")
        
        import yaml
        
        if self.config_path is None:
            # Use default config path
            self.config_path = "alf-config.yaml"
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        env = environment.AlfredTWEnv(config, train_eval='train')
        return env
    
    def reset(self, **kwargs) -> Tuple[Any, Dict]:
        """Reset the environment."""
        obs, info = self.env.reset()
        self.clear_history()
        
        self.observation_text = obs[0] if isinstance(obs, list) else obs
        self.admissible_commands = info.get('admissible_commands', [set()])[0]
        self.traj_data = info.get('extra.gamefile', [None])[0]
        
        info_dict = {
            'observation_text': self.observation_text,
            'admissible_commands': [self.admissible_commands],
        }
        
        return obs, info_dict
    
    def step(self, action: Any) -> Tuple[Any, float, bool, Dict]:
        """Take a step in the environment."""
        obs, reward, done, info = self.env.step([action])
        
        self.observation_text = obs[0] if isinstance(obs, list) else obs
        self.admissible_commands = info.get('admissible_commands', [set()])[0]
        
        self.step_count += 1
        
        info_dict = {
            'observation_text': self.observation_text,
            'admissible_commands': [self.admissible_commands],
        }
        
        # Handle list returns
        if isinstance(done, list):
            done = done[0]
        if isinstance(reward, list):
            reward = reward[0]
        
        return obs, reward, done, info_dict
    
    def get_task_description(self, prompt_version: str = "v1") -> str:
        """Get task description from ALFWorld."""
        if self.traj_data is not None and HAS_ALFWORLD:
            try:
                return get_templated_task_desc(self.traj_data)
            except:
                pass
        return "complete the household task"
    
    def get_observation_prompt(self, prompt_version: str = "v1") -> List[Dict]:
        """Generate observation prompt for VLM."""
        task = self.get_task_description(prompt_version)
        admissible = sorted(list(self.admissible_commands))
        formatted_actions = "\n ".join(f"'{s}'" for s in admissible)
        
        question = f"Your are an expert in the ALFRED Embodied Environment. "
        question += f"You are also given the following text description of the current scene: {self.observation_text}. "
        question += f"Your task is to {task}. "
        question += f"Your admissible actions of the current situation are: [{formatted_actions}]. "
        question += '{"thoughts": <any thoughts that will lead you to the goal>, "action": <an admissible action>}'
        
        messages = [{"type": "text", "text": question}]
        
        return [{"role": "user", "content": messages}]
    
    def text_to_action(self, text_action: str) -> str:
        """Convert VLM text output to action string."""
        try:
            parsed = eval(text_action)
            action = parsed.get("action", text_action)
            return action
        except:
            # Return raw text as action for ALFWorld
            return text_action
    
    def process_reward(self, reward: float, done: bool, info: Dict) -> float:
        """Process reward for ALFWorld."""
        return float(reward)
    
    @property
    def action_space(self) -> Any:
        """Return action space (text-based for ALFWorld)."""
        # ALFWorld uses text actions, return a dummy discrete space
        import gymnasium as gym
        return gym.spaces.Discrete(100)  # Placeholder

