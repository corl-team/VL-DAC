"""
Environment registry for dynamic environment loading.
"""

from typing import Dict, Type, Optional, Any
from .base import BaseEnvironment, EnvironmentWrapper


class EnvironmentRegistry:
    """
    Registry for environment types.
    Allows dynamic registration and instantiation of environments.
    """
    
    _registry: Dict[str, Type[BaseEnvironment]] = {}
    
    @classmethod
    def register(cls, name: str):
        """Decorator to register an environment class."""
        def decorator(env_class: Type[BaseEnvironment]):
            cls._registry[name.lower()] = env_class
            return env_class
        return decorator
    
    @classmethod
    def get(cls, name: str) -> Optional[Type[BaseEnvironment]]:
        """Get environment class by name."""
        # Try exact match first
        if name.lower() in cls._registry:
            return cls._registry[name.lower()]
        
        # Try partial match
        for key, env_class in cls._registry.items():
            if key in name.lower():
                return env_class
        
        return None
    
    @classmethod
    def list_environments(cls) -> list:
        """List all registered environments."""
        return list(cls._registry.keys())
    
    @classmethod
    def create(
        cls,
        env_name: str,
        seed: int,
        rank: int,
        max_episode_steps: int = 128,
        max_image_obs_len: int = 4,
        log_dir: Optional[str] = None,
        prompt_version: str = "v1",
        **kwargs
    ) -> EnvironmentWrapper:
        """Create an environment instance with wrapper."""
        env_class = cls.get(env_name)
        
        if env_class is None:
            raise ValueError(
                f"Environment '{env_name}' not found. "
                f"Available: {cls.list_environments()}"
            )
        
        env = env_class(
            env_name=env_name,
            seed=seed,
            rank=rank,
            max_episode_steps=max_episode_steps,
            max_image_obs_len=max_image_obs_len,
            log_dir=log_dir,
            **kwargs
        )
        
        return EnvironmentWrapper(env, prompt_version)


def get_environment(
    env_name: str,
    seed: int,
    rank: int,
    max_episode_steps: int = 128,
    max_image_obs_len: int = 4,
    log_dir: Optional[str] = None,
    prompt_version: str = "v1",
    **kwargs
) -> EnvironmentWrapper:
    """
    Convenience function to get an environment by name.
    
    Args:
        env_name: Name of the environment (e.g., 'MiniWorld-OneRoom-v0', 'alfworld', 'webshop')
        seed: Random seed
        rank: Process rank for distributed training
        max_episode_steps: Maximum steps per episode
        max_image_obs_len: Maximum image observation history length
        log_dir: Directory for logging
        prompt_version: Prompt version to use
        **kwargs: Additional environment-specific arguments
    
    Returns:
        EnvironmentWrapper instance
    """
    return EnvironmentRegistry.create(
        env_name=env_name,
        seed=seed,
        rank=rank,
        max_episode_steps=max_episode_steps,
        max_image_obs_len=max_image_obs_len,
        log_dir=log_dir,
        prompt_version=prompt_version,
        **kwargs
    )

