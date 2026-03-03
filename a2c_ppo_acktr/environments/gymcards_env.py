"""
GymCards environment implementation.
"""

import os
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from PIL import Image

try:
    import gym_cards
    HAS_GYMCARDS = True
except ImportError:
    HAS_GYMCARDS = False

import gymnasium as gym
from stable_baselines3.common.monitor import Monitor

from .base import BaseEnvironment
from .registry import EnvironmentRegistry


# Action lists for different card games
GYMCARDS_ACTIONS = {
    'gym_cards/NumberLine-v0': {
        'actions': ['+', '-'],
        'action_map': {'+': 1, '-': 0}
    },
    'gym_cards/EZPoints-v0': {
        'actions': ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '+', '*', '='],
    },
    'gym_cards/Blackjack-v0': {
        'actions': ['stand', 'hit'],
    },
    'gym_cards/Points24-v0': {
        'actions': ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '+', '-', '*', '/', '(', ')', '='],
    },
}


@EnvironmentRegistry.register("gym_cards")
@EnvironmentRegistry.register("gymcards")
class GymCardsEnvironment(BaseEnvironment):
    """
    GymCards environment wrapper for VLM RL training.
    """
    
    def __init__(self, **kwargs):
        self.formula = ""
        super().__init__(**kwargs)
        self.env = self.create_env()
        
    def create_env(self) -> Any:
        """Create GymCards environment."""
        if not HAS_GYMCARDS:
            raise ImportError("gym_cards is not installed. Please install it first.")
        
        env = gym.make(self.env_name, max_episode_steps=self.max_episode_steps)
        
        if self.log_dir is not None:
            os.makedirs(os.path.join(self.log_dir, str(self.rank)), exist_ok=True)
            env = Monitor(
                env,
                os.path.join(self.log_dir, str(self.rank)),
                allow_early_resets=True
            )
        
        return env
    
    def reset(self, **kwargs) -> Tuple[Any, Dict]:
        """Reset the environment."""
        obs, info = self.env.reset(**kwargs)
        self.clear_history()
        self.formula = ""
        
        # GymCards returns image observation
        if hasattr(obs, 'shape'):
            self.add_observation(obs)
        
        return obs, info
    
    def step(self, action: Any) -> Tuple[Any, float, bool, Dict]:
        """Take a step in the environment."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        
        self.step_count += 1
        self.infos = [info]
        
        # Update formula from info if available
        if 'Formula' in info:
            self.formula = ''.join(str(e) for e in info['Formula'])
        
        if hasattr(obs, 'shape'):
            self.add_observation(obs)
        
        return obs, reward, done, info
    
    def get_observation_prompt(self, prompt_version: str = "v1") -> List[Dict]:
        """Generate observation prompt for VLM."""
        past_images = list(self.image_observations)
        
        if self.env_name == 'gym_cards/NumberLine-v0':
            question = self._get_numberline_prompt()
        elif self.env_name == 'gym_cards/EZPoints-v0':
            question = self._get_ezpoints_prompt()
        elif self.env_name == 'gym_cards/Blackjack-v0':
            question = self._get_blackjack_prompt()
        elif self.env_name == 'gym_cards/Points24-v0':
            question = self._get_points24_prompt()
        else:
            question = "Choose the best action for this card game."
        
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
    
    def _get_numberline_prompt(self) -> str:
        question = "You are playing a game called number line. You will see a target number and a current number in the image. "
        question += "And your goal is to move the current number closer to the target by choosing either adding or subtracting one to the current number. "
        question += "You can return one of two actions: '-' or '+'. Also describe current observation and your thoughts."
        question += "The output format should be as follows: "
        question += '{"thoughts": <any thoughts that will lead you to the goal>, "action": <"+" or "-">}'
        return question
    
    def _get_ezpoints_prompt(self) -> str:
        question = "You are an expert card game player. You are observing two cards in the image. "
        question += f"You are observing the current formula: {self.formula}."
        question += "You can choose between ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '+', '*', '=']. You should choose only one symbol from given list. "
        question += "The chosen symbol will be appended to the current formula. If formula is empty choose random symbol from given list. "
        question += "Note that 'J', 'Q', and 'K' count as '10'. "
        question += 'Your goal is to output a formula that evaluates to 12, and each number can only be used once.'
        question += 'If the current formula is complete, output "=". '
        question += 'Otherwise consider which number or operator should be appended to the current formula to make it equal 12. Return your thoughts and action. Your thoughts must be concise. '
        question += 'The output format must be as follows: {"thoughts": <any thoughts that will lead you to the goal>, "action": <your chosen symbol>}'
        return question
    
    def _get_blackjack_prompt(self) -> str:
        question = "You are a blackjack player. You are observing the current game state, you can choose between ['stand', 'hit']. "
        question += "Your response should be a valid json file in the following format: \n{\n "
        question += '"thoughts": "{first describe your total points and the dealer\'s total points then think about which action to choose}", \n'
        question += '"action": "stand" or "hit" \n}'
        return question
    
    def _get_points24_prompt(self) -> str:
        current_choice = "['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '+', '-', '*', '/', '(', ')', '=']" if not self.formula[-1:].isdigit() else "['+', '-', '*', '/', '(', ')', '=']"
        
        question = "You are an expert 24 points card game player. You are observing the four cards in the image. Your goal is to make 24 using points that correspond to given cards. "
        question += f"{'You are observing the current formula: ' if self.formula else 'Right now you are on the first step to make a formula'}{self.formula}. Final formula after all steps should be in format x+o+p+q=24, where [x, o, p, q] are cards from observation and pluses might be any possible operations. "
        question += f"You can choose between {current_choice} symbols. You should choose only {'one symbol from given list that correspond to one of the cards or operator.' if not self.formula[-1:].isdigit() else 'one symbol corresponding to operator.'} "
        question += "The chosen symbol will be appended to the current formula. "
        question += "Note that 'J', 'Q', and 'K' count as '10'. "
        question += "Your goal is to output a formula that evaluates to 24, and each number (card) can only be used once. "
        question += "If the current formula equals 24, output '='. "
        question += "Otherwise consider which number or operator should be appended to the current formula to make it equal 24. Your thoughts must be concise. "
        question += "Your response must be in the following format: "
        question += '{"thoughts": <any thoughts that will lead you to the goal>, "action": <your chosen card by number or operator>}'
        return question
    
    def text_to_action(self, text_action: str) -> int:
        """Convert VLM text output to action integer."""
        try:
            parsed = eval(text_action)
            action = parsed.get("action")
            
            if self.env_name == 'gym_cards/NumberLine-v0':
                return {"+": 1, "-": 0}.get(action, np.random.choice([0, 1]))
            
            elif self.env_name == 'gym_cards/EZPoints-v0':
                action_list = GYMCARDS_ACTIONS['gym_cards/EZPoints-v0']['actions']
                if isinstance(action, int):
                    action = str(action)
                try:
                    return action_list.index(action)
                except ValueError:
                    return np.random.choice(len(action_list))
            
            elif self.env_name == 'gym_cards/Blackjack-v0':
                action_list = GYMCARDS_ACTIONS['gym_cards/Blackjack-v0']['actions']
                if isinstance(action, int):
                    action = str(action)
                try:
                    return action_list.index(action)
                except ValueError:
                    return np.random.choice(len(action_list))
            
            elif self.env_name == 'gym_cards/Points24-v0':
                action_list = GYMCARDS_ACTIONS['gym_cards/Points24-v0']['actions']
                if isinstance(action, int):
                    action = str(action)
                try:
                    return action_list.index(action)
                except ValueError:
                    return np.random.choice(len(action_list))
            
            return action
            
        except Exception as e:
            print(f"Failed to parse action '{text_action}': {e}")
            return np.random.choice([0, 1])
    
    def process_reward(self, reward: float, done: bool, info: Dict) -> float:
        """Process reward for GymCards."""
        return float(reward)
    
    @property
    def action_space(self) -> Any:
        """Return action space."""
        return self.env.action_space

