"""
WebShop environment implementation.
"""

import os
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from PIL import Image

try:
    from web_agent_site.envs import WebAgentSiteEnv
    HAS_WEBSHOP = True
except ImportError:
    HAS_WEBSHOP = False

from .base import BaseEnvironment
from .registry import EnvironmentRegistry


@EnvironmentRegistry.register("webshop")
class WebShopEnvironment(BaseEnvironment):
    """
    WebShop environment wrapper for VLM RL training.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.current_instruction = ""
        self.current_page_info = {}
        self.env = self.create_env()
        
    def create_env(self) -> Any:
        """Create WebShop environment."""
        if not HAS_WEBSHOP:
            raise ImportError("WebShop is not installed. Please install it first.")
        
        env = WebAgentSiteEnv(
            self.env_name,
            render_mode="rgb_array",
            max_episode_steps=self.max_episode_steps,
        )
        return env
    
    def reset(self, **kwargs) -> Tuple[Any, Dict]:
        """Reset the environment."""
        obs = self.env.reset(**kwargs)
        self.clear_history()
        
        # WebShop returns a dict with 'image' and 'text'
        if isinstance(obs, dict):
            image_obs = obs.get('image')
            text_obs = obs.get('text', '')
            self.current_instruction = obs.get('instruction', '')
            self.current_page_info = obs
            if image_obs is not None:
                self.add_observation(image_obs)
        else:
            self.add_observation(obs)
            
        info = {'instruction': self.current_instruction, 'page_info': self.current_page_info}
        return obs, info
    
    def step(self, action: Any) -> Tuple[Any, float, bool, Dict]:
        """Take a step in the environment."""
        obs, reward, done, info = self.env.step(action)
        
        self.step_count += 1
        
        if isinstance(obs, dict):
            image_obs = obs.get('image')
            if image_obs is not None:
                self.add_observation(image_obs)
            self.current_page_info = obs
        else:
            self.add_observation(obs)
        
        return obs, reward, done, info
    
    def get_observation_prompt(self, prompt_version: str = "v1") -> List[Dict]:
        """Generate observation prompt for VLM."""
        past_images = list(self.image_observations)
        
        instruction = self.current_instruction or "Find and purchase the specified product"
        page_text = self.current_page_info.get('text', '') if isinstance(self.current_page_info, dict) else ''
        available_actions = self.current_page_info.get('available_actions', []) if isinstance(self.current_page_info, dict) else []
        
        question = f"""You are shopping on an e-commerce website.

# Your Goal:
{instruction}

# Current Page Information:
{page_text}

# Available Actions:
{', '.join(available_actions) if available_actions else 'Navigate, search, click on products, add to cart, buy'}

Based on the page screenshot and the information above, decide what action to take next to accomplish your goal.

Your response should be in the following format:
{{"thoughts": <your reasoning about what to do next>, "action": <the action to take>}}
"""
        
        messages = []
        for image in past_images:
            # Convert numpy array to PIL Image for qwen_vl_utils compatibility
            if isinstance(image, np.ndarray):
                pil_image = Image.fromarray(image.astype(np.uint8))
            else:
                pil_image = image
            messages.append({"type": "image", "image": pil_image})
        messages.append({"type": "text", "text": question})
        
        return [{"role": "user", "content": messages}]
    
    def text_to_action(self, text_action: str) -> str:
        """Convert VLM text output to action."""
        try:
            parsed = eval(text_action)
            action = parsed.get("action", text_action)
            return action
        except:
            return text_action
    
    def process_reward(self, reward: float, done: bool, info: Dict) -> float:
        """Process reward for WebShop."""
        return float(reward)
    
    @property
    def action_space(self) -> Any:
        """Return action space."""
        if hasattr(self.env, 'action_space'):
            return self.env.action_space
        import gymnasium as gym
        return gym.spaces.Discrete(100)  # Placeholder for text actions

