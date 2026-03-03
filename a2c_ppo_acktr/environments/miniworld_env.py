"""
MiniWorld environment implementation.
"""

import os
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from PIL import Image

try:
    import miniworld
except ImportError:
    miniworld = None

import gymnasium as gym
from stable_baselines3.common.monitor import Monitor

from .base import BaseEnvironment
from .registry import EnvironmentRegistry


# Task descriptions for MiniWorld environments
MINIWORLD_TASKS = {
    "MiniWorld-CollectHealth-v0": "to collect health kits and stay alive as long as possible",
    "MiniWorld-FourRooms-v0": {
        "v1": "to go to a red box in four rooms within as few steps as possible",
        "v2": "to go to a red box in four rooms within as few steps as possible",
        "v3": "to go to red box"
    },
    "MiniWorld-Hallway-v0": {
        "v1": "to go to a red box at the end of a hallway within as few steps as possible",
        "v2": "to go to a red box at the end of a hallway within as few steps as possible",
        "v3": "to go to red box"
    },
    "MiniWorld-Maze-v0": "to reach the red box within as few steps as possible",
    "MiniWorld-MazeS2-v0": "to reach the red box within as few steps as possible",
    "MiniWorld-MazeS3-v0": "to reach the red box within as few steps as possible",
    "MiniWorld-MazeS3Fast-v0": "to reach the red box within as few steps as possible",
    "MiniWorld-OneRoom-v0": {
        "v1": "to go to a red box randomly placed in one big room",
        "v2": "to go to a red box randomly placed in one big room",
        "v3": "to go to red box"
    },
    "MiniWorld-OneRoomS6-v0": "to go to a red box placed randomly in one big room",
    "MiniWorld-OneRoomS6Fast-v0": "to go to a red box placed randomly in one big room",
    "MiniWorld-PickupObjects-v0": "to collect as many objects as possible",
    "MiniWorld-PutNext-v0": "to put a red box next to a yellow box",
    "MiniWorld-RoomObjects-v0": "to collect as many objects as possible",
    "MiniWorld-Sidewalk-v0": "to walk on a sidewalk up to an object to be collected. Don't walk into the street. The goal is to reach the object in as few steps as possible",
    "MiniWorld-Sign-v0": "to read the sign and follow the instructions",
    "MiniWorld-TMaze-v0": "to reach the red box within as few steps as possible.",
    "MiniWorld-TMazeLeft-v0": "to reach the red box within as few steps as possible",
    "MiniWorld-TMazeRight-v0": "to reach the red box within as few steps as possible",
    "MiniWorld-ThreeRooms-v0": "to reach the red box within as few steps as possible",
    "MiniWorld-WallGap-v0": {
        "v1": "to go to a red box behind a wall within as little steps as possible",
        "v2": "to go to a red box behind a wall within as little steps as possible",
        "v3": "to go to red box"
    },
    "MiniWorld-YMaze-v0": "to go to a red box within as little steps as possible",
    "MiniWorld-YMazeLeft-v0": "to go to a red box within as little steps as possible",
    "MiniWorld-YMazeRight-v0": "to go to a red box within as little steps as possible",
}


@EnvironmentRegistry.register("miniworld")
class MiniWorldEnvironment(BaseEnvironment):
    """
    MiniWorld environment wrapper for VLM RL training.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.env = self.create_env()
        
    def create_env(self) -> Any:
        """Create MiniWorld environment."""
        if miniworld is None:
            raise ImportError("miniworld is not installed. Please install it first.")
        
        env = gym.make(
            self.env_name,
            render_mode="rgb_array",
            max_episode_steps=self.max_episode_steps
        )
        
        if self.log_dir is not None:
            os.makedirs(os.path.join(self.log_dir, str(self.rank)), exist_ok=True)
            env = Monitor(
                env,
                os.path.join(self.log_dir, str(self.rank)),
                allow_early_resets=True
            )
        
        env.reset(seed=self.seed + self.rank)
        return env
    
    def reset(self, **kwargs) -> Tuple[Any, Dict]:
        """Reset the environment."""
        obs, info = self.env.reset(**kwargs)
        self.clear_history()
        # MiniWorld returns RGB image directly
        if hasattr(obs, 'shape') and len(obs.shape) == 3:
            self.add_observation(obs)
        return obs, info
    
    def step(self, action: Any) -> Tuple[Any, float, bool, Dict]:
        """Take a step in the environment."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        
        if truncated:
            info['TimeLimit.truncated'] = True
        
        self.step_count += 1
        self.infos = [info]
        
        if hasattr(obs, 'shape') and len(obs.shape) == 3:
            self.add_observation(obs)
        
        return obs, reward, done, info
    
    def get_task_description(self, prompt_version: str = "v1") -> str:
        """Get task description for MiniWorld environment."""
        task = MINIWORLD_TASKS.get(self.env_name, "to complete the task")
        if isinstance(task, dict):
            return task.get(prompt_version, task.get("v1", "to complete the task"))
        return task
    
    def get_observation_prompt(self, prompt_version: str = "v1") -> List[Dict]:
        """Generate observation prompt for VLM."""
        task = self.get_task_description(prompt_version)
        past_images = list(self.image_observations)
        
        if prompt_version == "v1":
            question = f"""# Instructions
You are operating in a simulator. Your objective is to complete the task. To complete the task, you need to take actions. Upon completing the task, the simulation will end, and you will receive a reward. If you will not solve the task, you will get reward 0.
TASK: {task}.
Take ONE action based on the current observation. Current observation is {f'the state after {len(past_images)} previous actions' if len(past_images) > 1 else 'starting state'}. If you cannot determine how to solve the task, you may turn around or explore the environment to identify the appropriate action.
# Available actions:
0 : turn left
1 : turn right
2 : move forward
3 : move back

First, describe what you observe on the last state using a text description. Try to understand your position relative to the goal, walls, and other objects. Then, carefully consider which action will help you complete the task. Think step by step to understand the environment. After that, choose only one action. Return current scene description, thoughts, and the chosen action.

# ADDITIONAL INSTRUCTIONS:
- If you're stuck against a wall, try to turn around and explore the environment.
- If you can't see the goal, try to explore the environment.

The output format should be as follows:
{{"description": <description>, "thoughts": <thoughts>, "action": <action_number>}}
"""
        elif prompt_version in ["v2", "v3"]:
            question = f"""# Instructions
You are operating in a simulator. Your objective is to complete the task. To complete the task, you need to take actions. Upon completing the task, the simulation will end, and you will receive a reward. If you will not solve the task, you will get reward 0.
TASK: {task}.
Take ONE action based on the current observation. Current observation is {f'the state after {len(past_images)} previous actions' if len(past_images) > 1 else 'starting state'}. If you cannot determine how to solve the task, you may turn around or explore the environment to identify the appropriate action.
# Available actions:
0 : turn left
1 : turn right
2 : move forward
3 : move back

Think through current observation and choose one action to take.

# ADDITIONAL INSTRUCTIONS:
- If you're stuck against a wall, try to turn around and explore the environment.
- If you can't see the goal, try to explore the environment.

The output format should be as follows:
{{"thoughts": <any thoughts that will lead you to the goal>, "action": <action_number>}}
"""
        else:
            question = f"Task: {task}. Choose action 0-3."
        
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
    
    def text_to_action(self, text_action: str) -> int:
        """Convert VLM text output to action integer."""
        try:
            parsed = eval(text_action)
            action = parsed.get("action")
            if isinstance(action, str):
                action = int(action)
            return action
        except Exception as e:
            print(f"Failed to parse action '{text_action}': {e}")
            return np.random.choice([0, 1, 2, 3])
    
    def process_reward(self, reward: float, done: bool, info: Dict) -> float:
        """Process reward for MiniWorld."""
        if done and reward <= 0:
            return -1.0
        return float(reward)
    
    @property
    def action_space(self) -> Any:
        """Return action space."""
        return self.env.action_space

